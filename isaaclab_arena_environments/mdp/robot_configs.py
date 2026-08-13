# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Task-specific robot configurations for manipulation tasks."""

from __future__ import annotations

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# ===========================================================================================
# Franka Panda robot configuration optimized for assembly tasks (peg insert, gear mesh, etc.).
# ===========================================================================================

FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()

# Enable contact sensors for assembly tasks
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.spawn.activate_contact_sensors = True

# Disable gravity on the robot for better control stability
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True

# Increase stiffness and damping for precise positioning
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 150.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 30.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 150.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_forearm"].damping = 30.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_hand"].stiffness = 150.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_hand"].damping = 30.0

# Set initial position for tabletop tasks
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.init_state.pos = (0.0, 0.0, 0.0)


# ===========================================================================================
# Franka Panda configuration for B-DASH (scripted pick + force-control insertion).
#
# Keeps the *stiffer* HIGH_PD gains (400) for practical free-space speed during pick/transport —
# the assembly cfg above softens to 150, which makes the arm too slow (~0.04 m/s) to finish a
# pick+insert inside the episode and tanks eval throughput. Insertion compliance is provided by the
# controller's force-limited press + spiral search (configs/bdash/peg_insert/controllers.yaml), not a soft arm.
# Contact sensors are enabled (peg/finger ContactSensors) and base-link gravity disabled for control
# stability (HIGH_PD already disables gravity; set explicitly for clarity).
# ===========================================================================================
FRANKA_PANDA_BDASH_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_PANDA_BDASH_CFG.spawn.activate_contact_sensors = True
FRANKA_PANDA_BDASH_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_BDASH_CFG.init_state.pos = (0.0, 0.0, 0.0)

# Lower resting arm pose: the stock HIGH_PD home (joint2 -0.569, joint4 -2.810) holds the gripper
# high above the table. Lean the shoulder forward and open the elbow so the gripper starts lower /
# closer to the peg+socket workspace; keep the wrist (joint6) pointing down. Tune joint2/joint4 to
# raise/lower. NOTE: this changes the demo/eval start pose -> existing VLA models are off-distribution
# and need re-recording + retrain to compare fairly.
# NOTE (verified 2026-06-23): this init_state.joint_pos is in fact a NO-OP for the actual reset pose —
# the Franka embodiment's `init_franka_arm_pose` reset EVENT (FrankaEventCfg, default_pose
# [0, -0.785, -0.1107, -1.1775, 0, 0.785, 0.785, ...]) overrides it every reset. Probed post-reset
# joint_pos is the event pose regardless of the values below. Left as-is; change the event to actually
# move the start pose.
FRANKA_PANDA_BDASH_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.10,  # was -0.569: shoulder forward/down -> lowers gripper
    "panda_joint3": 0.0,
    "panda_joint4": -2.50,  # was -2.810: open elbow slightly
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,  # keep wrist pointing down
    "panda_joint7": 0.741,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
