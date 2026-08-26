# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Can the arm actually HOLD the turned pose? Measured, after assuming it could.

The §4-2 turn ends with the tool horizontal: the part hangs vertically from a wrist that points
sideways. `verify_reerect.py` checked depth, torque and swept volume and said yes to all three -- and
none of them is reachability. The first run with the leg wired in threw the part 0.791 m with a
96.5 mm IK residual, which is what an unreachable orientation command looks like from the outside.

So this probe asks the question the earlier one did not: for each wrist yaw the turn can end at (the
grasp axis is canonicalised into the +x half-plane, so psi spans -90..+90 deg), drive the arm to the
turned pose at a candidate set-down station and measure what it converges to -- position residual,
orientation residual, and how much joint travel is left. A pose the arm cannot hold is not a pose
the open-loop turn may end at, whatever the geometry says.

Reported per (station, psi) so the output is a map, not a verdict: if the whole station fails the
station is wrong, and if only the extremes fail the turn needs its range restricted.
"""

from __future__ import annotations

import math

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--stations", default="0.50,0.00,0.20;0.45,0.00,0.25;0.55,0.00,0.30;0.45,0.15,0.25")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument(
    "--azimuths",
    default="",
    help=(
        "WORLD azimuths (deg) for the hand to end pointing at. Given these, the free yaw is solved "
        "per grasp angle as phi = beta - psi, which is what the controller does. Overrides --phis. The "
        "azimuth is the quantity that decides reachability -- measured, 60-150 deg is reachable and "
        "everything outside jams a joint -- so a sweep in phi at fixed psi is a sweep across a boundary "
        "rather than along a controlled variable."
    ),
)
parser.add_argument(
    "--phis",
    default="0",
    help=(
        "extra yaws (deg) about the part's OWN axis to try after the turn. Free by symmetry: once "
        "the part is vertical it is a cylinder about that axis, so spinning the wrist around it leaves "
        "the part's final standing pose identical while moving the ARM to a different configuration."
    ),
)
add_example_environments_cli_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul  # noqa: E402

from isaaclab_arena.controllers.ee_control import ee_pose_action, read_ee_pose  # noqa: E402
from isaaclab_arena.utils.random import set_seed  # noqa: E402


def main():
    arena_builder = get_arena_builder_from_cli(args)
    env_name, env_cfg = arena_builder.build_registered()
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    set_seed(0, env)
    uenv = env
    dev = uenv.device
    # THE REAL ee_control, loaded from the config the teacher uses. It was hand-written here first
    # and every entry was wrong: action_scale 1.0 against the true 0.5 (which "MUST match
    # FrankaIKJointRecordingActionCfg.scale"), max_rot_step 0.10 against 0.30, rot_gain 3.0 against
    # 4.0. A probe that commands the arm differently from the controller it is meant to characterise
    # measures the probe, and the first sweep run that way reported every pose unreachable.
    from isaaclab_arena_environments.mdp.bdash_chuck_config import load_controllers_cfg

    ee_cfg = load_controllers_cfg()["ee_control"]
    print(f"  ee_control: {ee_cfg}")

    stations = [tuple(float(v) for v in s.split(",")) for s in args.stations.split(";")]
    psis = [math.radians(d) for d in range(-90, 91, 30)]
    azimuths = [math.radians(float(v)) for v in args.azimuths.split(",")] if args.azimuths else None
    phis = [math.radians(float(v)) for v in args.phis.split(",")]

    print(
        f"\n  {'station':>16s} {'psi':>6s} {'phi':>5s} {'pos err':>9s} {'rot err':>9s} {'joint margin':>13s}  verdict"
    )
    print("  " + "-" * 72)
    worst = {}
    for sx, sy, sz in stations:
        for psi in psis:
            # With --azimuths the free yaw is DERIVED, exactly as the controller derives it, so the
            # row varies the thing that decides reachability instead of the thing that happens to be
            # a parameter.
            for phi in [b - psi for b in azimuths] if azimuths else phis:
                env.reset()
                z_axis = torch.zeros(1, 3, device=dev)
                z_axis[:, 2] = 1.0
                q0 = torch.zeros(1, 4, device=dev)
                q0[:, 1] = 1.0  # (0,1,0,0): tool points straight down
                q_grasp = quat_mul(quat_from_angle_axis(torch.tensor([psi], device=dev), z_axis), q0)
                local_y = torch.zeros(1, 3, device=dev)
                local_y[:, 1] = 1.0
                y_e = quat_apply(q_grasp, local_y)
                q_end = quat_mul(quat_from_angle_axis(torch.tensor([math.pi / 2.0], device=dev), y_e), q_grasp)
                # The free yaw. Applied about WORLD z, which after the turn IS the part's own axis, so
                # this rotates the arm around a part that does not care -- the one spare degree of
                # freedom the turn has, and the only knob available if the pose is otherwise unreachable.
                q_end = quat_mul(quat_from_angle_axis(torch.tensor([phi], device=dev), z_axis), q_end)

                target = torch.tensor([[sx, sy, sz]], device=dev)
                closed = torch.zeros(1, dtype=torch.bool, device=dev)
                for _ in range(args.steps):
                    action = ee_pose_action(env, target, q_end, closed, ee_cfg)
                    env.step(action)

                tcp, quat = read_ee_pose(env)
                pos_err = float((target - tcp).norm())
                dq = quat_mul(q_end, torch.tensor([[1.0, -1.0, -1.0, -1.0]], device=dev) * quat)
                rot_err = float(2.0 * torch.acos(dq[:, 0].abs().clamp(max=1.0)))

                robot = uenv.scene["robot"]
                q = robot.data.joint_pos[0, :7]
                lo = robot.data.soft_joint_pos_limits[0, :7, 0]
                hi = robot.data.soft_joint_pos_limits[0, :7, 1]
                margins = torch.minimum(q - lo, hi - q)
                margin = float(margins.min())
                # WHICH joint is jammed, not just that one is. Feasibility here is decided by a
                # single joint running out of travel, and knowing which one says whether the fix is
                # the set-down pose, the free yaw, or the arm's starting posture.
                tight = int(margins.argmin())

                ok = pos_err < 0.015 and rot_err < math.radians(5.0) and margin > 0.05
                key = (sx, sy, sz, round(math.degrees(phi)))
                worst[key] = worst.get(key, 0) + (0 if ok else 1)
                print(
                    f"  ({sx:.2f},{sy:5.2f},{sz:.2f}) {math.degrees(psi):6.0f} {math.degrees(phi):5.0f} "
                    f"{pos_err * 1e3:8.1f}m {math.degrees(rot_err):8.2f}d {margin:8.3f}@j{tight + 1}  "
                    f"{'ok' if ok else 'FAIL'}"
                )

    print("\n  summary (failures out of %d wrist angles), best first:" % len(psis))
    for key, n in sorted(worst.items(), key=lambda kv: kv[1]):
        print(f"    ({key[0]:.2f}, {key[1]:.2f}, {key[2]:.2f})  phi={key[3]:4d}  {n} fail")
    print("REERECT_REACH_DONE")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
