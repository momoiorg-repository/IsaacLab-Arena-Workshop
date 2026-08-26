# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assert the chuck-loading scene resets into a state a demo could actually be recorded from.

The layout sampler is unit-tested sim-free; this checks the parts of the wiring only the simulator
can answer -- that the reset event actually reaches the sim, that the workpieces are at REST rather
than mid-fall (Isaac Lab computes observations with no physics steps after the reset events, so a
dropped part would appear falling in frame 0), that the per-workpiece contact sensors resolved to
the right shape, that a target was latched, and where the arm's TCP actually starts.

    unset DISPLAY
    /isaac-sim/python.sh scripts/bdash/check_chuck_scene.py --num_envs 1 --seed 0 \\
        --n_resets 5 bdash_chuck_load --task pick --variants all
"""

import math

print("[scenecheck] launching Isaac Sim (~40s)...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--n_resets", type=int, default=5, help="how many resets to inspect")
parser.add_argument("--settle_steps", type=int, default=10, help="hold steps after reset before measuring")
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_arena.controllers.ee_control import read_ee_pose
from isaaclab_arena.utils.random import set_seed


def main():
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if args_cli.seed is not None:
        set_seed(args_cli.seed, env)

    task = env.cfg.isaaclab_arena_env.task
    names = list(task.workpiece_names)
    sensors = list(task.finger_sensor_names)
    profiles = task.profiles
    from isaaclab_arena_environments.mdp.bdash_chuck_config import load_assets_cfg

    rim_z = float(task.geom["tray_rim_height"])
    tray_z = float(task.cfg["scene"]["tray_pose"][2])
    floor_top = tray_z + float(load_assets_cfg()["tray"]["floor"])

    problems = _fixture_overlaps(task.cfg, load_assets_cfg())
    with torch.inference_mode():
        for reset_i in range(args_cli.n_resets):
            env.reset()
            # settle: the poses are analytic resting poses, so this should barely move anything
            before = torch.stack([env.scene[n].data.root_pos_w[0].clone() for n in names])
            for _ in range(args_cli.settle_steps):
                env.step(torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device))
            after = torch.stack([env.scene[n].data.root_pos_w[0] for n in names])
            drift_mm = (after - before).norm(dim=-1).max().item() * 1000.0

            target = int(env.bdash_target_idx[0].item())
            side = env.bdash_side_lying[0].tolist()
            if not 0 <= target < len(names):
                problems.append(f"reset {reset_i}: target index {target} out of range")
            if sum(side) != round(0.4 * len(names)):
                problems.append(f"reset {reset_i}: {sum(side)} side-lying, expected {round(0.4 * len(names))}")

            # contact sensors: (num_envs, bodies=1, filters=2 fingers, 3)
            shape = tuple(env.scene[sensors[0]].data.force_matrix_w.shape)
            if shape[1:] != (1, 2, 3):
                problems.append(f"reset {reset_i}: {sensors[0]}.force_matrix_w shape {shape}, expected (N,1,2,3)")

            # every part resting on the tray floor, not falling and not sunk into it
            for k, (name, prof) in enumerate(zip(names, profiles)):
                z = float(env.scene[name].data.root_pos_w[0, 2].item())
                speed = float(env.scene[name].data.root_lin_vel_w[0].norm().item())
                if not floor_top - 0.002 <= z <= floor_top + prof["max_radius"] + 0.006:
                    problems.append(
                        f"reset {reset_i}: {name} root z={z:.4f} not resting "
                        f"(floor {floor_top:.4f}, max_radius {prof['max_radius']:.4f})"
                    )
                if speed > 0.02:
                    problems.append(f"reset {reset_i}: {name} moving at {speed:.3f} m/s after settle")

            # pairwise separation of the workpiece axes, using the real horizontal footprints
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    gap = _capsule_gap(env, names, profiles, i, j)
                    if gap < 0.0:
                        problems.append(f"reset {reset_i}: {names[i]} and {names[j]} overlap by {-gap * 1000:.1f} mm")

            tcp, _ = read_ee_pose(env)
            print(
                f"[scenecheck] reset {reset_i}: target={target} ({profiles[target]['variant']}, "
                f"side={bool(side[target])})  side_lying={sum(side)}/{len(names)}  "
                f"settle_drift={drift_mm:.2f}mm  TCP=({tcp[0, 0]:.3f},{tcp[0, 1]:.3f},{tcp[0, 2]:.3f})",
                flush=True,
            )

    tray_xy = task.cfg["scene"]["tray_pose"][:2]
    print(
        f"[scenecheck] tray centre=({tray_xy[0]:.3f},{tray_xy[1]:.3f})  rim={rim_z:.3f}  floor_top={floor_top:.3f}",
        flush=True,
    )
    # Everything below MUST flush: a SystemExit inside the SimulationApp context can leave Isaac Sim
    # hanging on a GUI error dialog, and an unflushed verdict is then lost with the killed process.
    if problems:
        print(f"\nFAILED - {len(problems)} problem(s):", flush=True)
        for problem in problems[:20]:
            print("  -", problem, flush=True)
    else:
        print("CHUCK_SCENE_OK", flush=True)
    env.close()
    return 1 if problems else 0


def _fixture_overlaps(task_cfg: dict, assets_cfg: dict) -> list[str]:
    """No fixture may reach into the tray footprint.

    A workpiece scattered into an overlapping corner spawns inside a kinematic body, and PhysX then
    ejects it at up to ``max_depenetration_velocity``. That is silent -- the layout sampler is only
    aware of the other workpieces -- so it is asserted here rather than left to be noticed as
    "the parts move a lot at reset".
    """
    scene = task_cfg["scene"]
    tray = assets_cfg["tray"]["outer"]
    chuck_d = assets_cfg["chuck"]["body_diameter"]
    # The tray is YAWED, so its axis-aligned extent is not its size. Using the AABB of the rotated
    # rectangle keeps this check conservative (an AABB never under-covers the shape), which is the
    # right side to err on for a test whose whole job is to catch a fixture reaching into the tray.
    yaw = math.radians(float(scene.get("tray_yaw_deg", 0.0)))
    tray_aabb = (
        abs(tray[0] * math.cos(yaw)) + abs(tray[1] * math.sin(yaw)),
        abs(tray[0] * math.sin(yaw)) + abs(tray[1] * math.cos(yaw)),
    )
    boxes = {
        "tray": (scene["tray_pose"], tray_aabb),
        "touchoff": (scene["touchoff_pose"], tuple(assets_cfg["touchoff_block"]["size"][:2])),
        "vblock": (scene["vblock_pose"], tuple(assets_cfg["vblock"]["size"][:2])),
        "chuck": (scene["chuck_pose"], (chuck_d, chuck_d)),
    }
    problems = []
    keys = list(boxes)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (pa, sa), (pb, sb) = boxes[keys[i]], boxes[keys[j]]
            over_x = min(pa[0] + sa[0] / 2, pb[0] + sb[0] / 2) - max(pa[0] - sa[0] / 2, pb[0] - sb[0] / 2)
            over_y = min(pa[1] + sa[1] / 2, pb[1] + sb[1] / 2) - max(pa[1] - sa[1] / 2, pb[1] - sb[1] / 2)
            if over_x > 0 and over_y > 0:
                problems.append(
                    f"fixtures {keys[i]} and {keys[j]} overlap by {over_x * 1000:.0f} x {over_y * 1000:.0f} mm"
                )
    return problems


def _capsule_gap(env, names, profiles, i, j) -> float:
    """Horizontal capsule-surface gap between two workpieces (negative = overlapping)."""
    segs = []
    for k in (i, j):
        data = env.scene[names[k]].data
        pos = data.root_pos_w[0]
        quat = data.root_quat_w[0]
        axis = _rotate(quat, torch.tensor([0.0, 0.0, 1.0], device=pos.device))
        far = pos[:2] + profiles[k]["length"] * axis[:2]
        segs.append((pos[:2].tolist(), far.tolist(), profiles[k]["max_radius"]))
    (a0, a1, ra), (b0, b1, rb) = segs
    return _seg_dist(a0, a1, b0, b1) - ra - rb


def _rotate(q, v):
    w, xyz = q[0], q[1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _seg_dist(a0, a1, b0, b1) -> float:
    """Sampled segment-segment distance -- coarse but independent of the sampler's own helper."""
    best = float("inf")
    steps = 24
    for si in range(steps + 1):
        s = si / steps
        px, py = a0[0] + s * (a1[0] - a0[0]), a0[1] + s * (a1[1] - a0[1])
        for ti in range(steps + 1):
            t = ti / steps
            qx, qy = b0[0] + t * (b1[0] - b0[0]), b0[1] + t * (b1[1] - b0[1])
            best = min(best, math.hypot(px - qx, py - qy))
    return best


if __name__ == "__main__":
    try:
        _status = main()
    finally:
        simulation_app.close()
    raise SystemExit(_status)
