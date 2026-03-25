# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return gripper finger positions as a (num_envs, 2) tensor.

    The Franka USD places panda_finger_joint1 last and panda_finger_joint2 second-to-last
    in the joint order. finger_joint_2 is negated because its joint axis is mirrored relative
    to finger_joint_1 in the USD, so both values are positive when the gripper is open.

    Args:
        env: The environment instance.
        robot_cfg: Scene entity config identifying the robot articulation.

    Returns:
        Tensor of shape (num_envs, 2) with [finger_joint_1, -finger_joint_2].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)
