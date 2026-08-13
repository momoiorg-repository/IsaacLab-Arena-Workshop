# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test arm step response: fire a small EEF delta at step 10, watch settling.

Prints J2 (shoulder) and J3 (elbow) position vs target for different kp/kd values
so we can pick stiffness that settles cleanly within ~5 steps (0.1 s).
"""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_arena.embodiments.crx5ia.crx5ia as crx5ia_mod

LOG = "/tmp/step_response.txt"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


# Pairs to test
KP_KD_PAIRS = [
    (400, 40),
    (800, 80),
    (2000, 200),
    (4000, 400),
]

# Small +Z EEF delta (5 mm) fired once at step 10
PULSE = torch.zeros(1, 7)
PULSE[0, 2] = 0.005  # 5 mm in Z, gripper stays 0

arena_builder = get_arena_builder_from_cli(args_cli)

for kp, kd in KP_KD_PAIRS:
    env_name, env_cfg = arena_builder.build_registered()
    env_cfg.terminations = None
    env_cfg.recorders = {}
    env_cfg.observations.policy.concatenate_terms = False

    # Patch stiffness/damping for the arm actuators
    for cfg_obj in (crx5ia_mod._CRX5IA_CFG, crx5ia_mod._CRX5IA_ROBOTIQ85_CFG):
        cfg_obj.actuators["arm"].stiffness = float(kp)
        cfg_obj.actuators["arm"].damping = float(kd)

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    jnames = robot.data.joint_names
    j2_idx = next(i for i, n in enumerate(jnames) if n == "J2")
    j3_idx = next(i for i, n in enumerate(jnames) if n == "J3")

    log(f"\n=== kp={kp}, kd={kd} ===")
    log(f"{'Step':>4} | {'J2_pos':>8} {'J2_tgt':>8} | {'J3_pos':>8} {'J3_tgt':>8}")
    log("-" * 52)

    zero = torch.zeros(1, env.action_space.shape[-1], device=env.device)
    pulse = PULSE.clone().to(env.device)

    for step in range(50):
        act = pulse if step == 10 else zero
        env.step(act)
        if step >= 8:
            pos = robot.data.joint_pos[0]
            tgt = robot.data.joint_pos_target[0]
            marker = " <-- PULSE" if step == 10 else ""
            log(
                f"{step:4d} | {pos[j2_idx]:+8.4f} {tgt[j2_idx]:+8.4f} | {pos[j3_idx]:+8.4f} {tgt[j3_idx]:+8.4f}{marker}"
            )

    env.close()

simulation_app.close()
