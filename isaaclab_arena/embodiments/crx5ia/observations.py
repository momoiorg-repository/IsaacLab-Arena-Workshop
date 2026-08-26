# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the Robotiq 140 finger_joint position as a 1-DOF observation.

    The Robotiq 2F-140 is driven by a single ``finger_joint``; the remaining
    finger joints are mimic joints.  This function returns only the primary
    finger joint value (the last revolute DOF in the articulation).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # finger_joint is the last actuated DOF (after arm joints J1-J6)
    finger_pos = robot.data.joint_pos[:, -1:].clone()
    return finger_pos


def gripper_pos_robotiq85(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the Robotiq 85 finger_joint position as a 1-DOF observation."""
    robot: Articulation = env.scene[robot_cfg.name]
    joint_names = robot.data.joint_names
    idx = next((i for i, n in enumerate(joint_names) if n == "finger_joint"), None)
    if idx is None:
        raise ValueError("finger_joint not found in articulation joint names")
    return robot.data.joint_pos[:, idx : idx + 1].clone()
