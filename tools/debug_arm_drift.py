# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Debug arm drift: step crx5ia_robotiq85 with zero actions and print joint positions."""

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

arena_builder = get_arena_builder_from_cli(args_cli)
env_name, env_cfg = arena_builder.build_registered()
env_cfg.terminations = None
env_cfg.recorders = {}
env_cfg.observations.policy.concatenate_terms = False

env = gym.make(env_name, cfg=env_cfg).unwrapped
env.reset()

robot = env.scene["robot"]
joint_names = robot.data.joint_names
arm_idx = [i for i, n in enumerate(joint_names) if n.startswith("J")]

LOG = "/tmp/drift_results.txt"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


log(f"\nArm joints: {[joint_names[i] for i in arm_idx]}")
header = "Step | " + " | ".join(f"{joint_names[i]:>7}" for i in arm_idx)
log(header)
log("-" * len(header))

zero_action = torch.zeros(1, env.action_space.shape[-1], device=env.device)

for step in range(201):
    env.step(zero_action)
    if step % 20 == 0:
        pos = robot.data.joint_pos[0]
        tgt = robot.data.joint_pos_target[0]
        vals = " | ".join(f"{pos[i].item():>+7.4f}" for i in arm_idx)
        tvals = " | ".join(f"{tgt[i].item():>+7.4f}" for i in arm_idx)
        log(f"{step:4d} | {vals}  <- pos")
        log(f"     | {tvals}  <- target")

env.close()
simulation_app.close()
