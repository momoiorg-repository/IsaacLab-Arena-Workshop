# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Does the tray hold still on its own? Steps the scene with a ZERO action and watches the parts.

This isolates the spawn from the teacher. ``check_chuck_teacher.py`` measured that 33% of parts
labelled upright are no longer upright by step 2 -- long before the arm arrives (a successful grasp
completes around step 97) -- and that those episodes lift 67% against 98.9% for the ones that stay
standing (Fisher p = 6e-14). With no arm motion at all, anything that still topples is the spawn.

A part cannot fall 18 degrees under gravity in one 16.6 ms env step; that needs ~19 rad/s. So the
velocities printed here are the measurement that matters: they separate "settles slowly" from
"is ejected", which is a penetration being resolved, not a topple.

    /isaac-sim/python.sh scripts/bdash/check_spawn_stability.py --num_envs 1 --seed 0 \\
        --episodes 20 bdash_chuck_load --variants all
"""

import json
import os

print("[spawncheck] launching Isaac Sim (~40s)...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--settle_steps", type=int, default=30, help="zero-action steps to watch per episode")
parser.add_argument("--jsonl", type=str, default=None)
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

    from isaaclab_arena_environments.mdp import bdash_chuck_randomization as bcr

    task = env.cfg.isaaclab_arena_env.task
    names = task.workpiece_names
    zero = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
    records = []
    print(
        f"[spawncheck] env ready; {args_cli.episodes} episodes x {args_cli.settle_steps} zero-action steps", flush=True
    )

    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            env.reset()
            tcp0 = read_ee_pose(env)[0][0].clone()
            # Every robot body, not just the TCP. The arm starts hovering over the tray, so its
            # forearm and elbow occupy tray airspace nowhere near the gripper -- a distance-to-TCP
            # test (which came back flat) cannot see them.
            bodies0 = env.scene[task.names["robot"]].data.body_pos_w[0].clone()
            upright0 = {}
            for k, name in enumerate(names):
                data = env.scene[name].data
                axis_z = bcr._rotate_z_axis(data.root_quat_w[0])[2]
                if axis_z > 0.95:  # only the standing ones are at risk of toppling
                    upright0[k] = dict(
                        axis_z=round(axis_z, 4),
                        pos=data.root_pos_w[0].clone(),
                        lin=round(float(data.root_lin_vel_w[0].norm()), 5),
                        ang=round(float(data.root_ang_vel_w[0].norm()), 5),
                    )

            worst = {
                k: dict(v, min_axis_z=v["axis_z"], max_lin=v["lin"], max_ang=v["ang"], tip=None)
                for k, v in upright0.items()
            }
            for step in range(args_cli.settle_steps):
                env.step(zero)
                for k in worst:
                    data = env.scene[names[k]].data
                    az = bcr._rotate_z_axis(data.root_quat_w[0])[2]
                    worst[k]["min_axis_z"] = min(worst[k]["min_axis_z"], round(az, 4))
                    worst[k]["max_lin"] = max(worst[k]["max_lin"], round(float(data.root_lin_vel_w[0].norm()), 5))
                    worst[k]["max_ang"] = max(worst[k]["max_ang"], round(float(data.root_ang_vel_w[0].norm()), 5))
                    if worst[k]["tip"] is None and az < 0.95:
                        worst[k]["tip"] = step

            for k, v in worst.items():
                records.append({
                    "episode": episode,
                    "piece": k,
                    "variant": task.profiles[k]["variant"],
                    "spawn_axis_z": v["axis_z"],
                    "spawn_lin": v["lin"],
                    "spawn_ang": v["ang"],
                    "min_axis_z": v["min_axis_z"],
                    "max_lin": v["max_lin"],
                    "max_ang": v["max_ang"],
                    "tip_step": v["tip"],
                    "spawn_x": round(float(v["pos"][0]), 5),
                    "spawn_y": round(float(v["pos"][1]), 5),
                    "spawn_z": round(float(v["pos"][2]), 5),
                    # Distance to the arm at reset. The sampler has no idea the robot exists, and
                    # the config already records this exact failure once: the touch-off block used
                    # to overlap the tray corner and "threw parts ~50 mm during the first few steps
                    # of every episode". If the toppling clusters near the hand, it is the same bug
                    # with the gripper as the intruder.
                    "d_tcp": round(float((v["pos"] - tcp0).norm()), 4),
                    "d_tcp_xy": round(float((v["pos"][:2] - tcp0[:2]).norm()), 4),
                    "d_body_min": round(float((bodies0 - v["pos"]).norm(dim=-1).min()), 4),
                    "d_body_xy_min": round(float((bodies0[:, :2] - v["pos"][:2]).norm(dim=-1).min()), 4),
                    "nearest_body": int((bodies0 - v["pos"]).norm(dim=-1).argmin()),
                })
            tipped = sum(1 for v in worst.values() if v["tip"] is not None)
            print(
                f"[spawncheck] ep={episode:3d} upright={len(worst)} tipped={tipped} "
                f"max_lin={max((v['max_lin'] for v in worst.values()), default=0):.3f} "
                f"max_ang={max((v['max_ang'] for v in worst.values()), default=0):.3f}",
                flush=True,
            )

    _report(records, args_cli.jsonl)
    env.close()


def _report(records, jsonl_path):
    if jsonl_path:
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        with open(jsonl_path, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"[spawncheck] wrote {jsonl_path}", flush=True)

    n = len(records)
    if not n:
        print("SPAWN_STABILITY_NO_UPRIGHT_PARTS", flush=True)
        return
    tipped = [r for r in records if r["tip_step"] is not None]
    early = [r for r in tipped if r["tip_step"] <= 2]
    print(f"\n  upright parts watched : {n}")
    print(f"  toppled with NO arm   : {len(tipped)} = {len(tipped) / n:.1%}   (of those, by step 2: {len(early)})")
    lin = sorted(r["max_lin"] for r in records)
    ang = sorted(r["max_ang"] for r in records)
    print(f"  max linear  speed     : med={lin[len(lin) // 2]:.4f}  max={lin[-1]:.4f} m/s")
    print(f"  max angular speed     : med={ang[len(ang) // 2]:.4f}  max={ang[-1]:.4f} rad/s")
    print(
        f"  spawn velocities      : lin max={max(r['spawn_lin'] for r in records):.5f}  ang"
        f" max={max(r['spawn_ang'] for r in records):.5f}"
    )
    if tipped and "d_tcp_xy" in records[0]:
        stayed = [r for r in records if r["tip_step"] is None]
        td = sorted(r["d_tcp_xy"] for r in tipped)
        sd = sorted(r["d_tcp_xy"] for r in stayed)
        print(f"  |xy| to TCP, TIPPED   : med={td[len(td) // 2]:.4f}  min={td[0]:.4f}  max={td[-1]:.4f}")
        if sd:
            print(f"  |xy| to TCP, STAYED   : med={sd[len(sd) // 2]:.4f}  min={sd[0]:.4f}  max={sd[-1]:.4f}")
    print(f"SPAWN_STABILITY {'UNSTABLE' if tipped else 'OK'} tipped={len(tipped)}/{n}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
