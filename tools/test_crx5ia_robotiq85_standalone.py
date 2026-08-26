#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone Isaac Sim test for CRX5iA + Robotiq 85 USD.

Loads the local USD, enables the ROS2 bridge, and wires up:
  - /joint_states  (publisher)  — consumed by MoveIt2 for feedback
  - /isaac_joint_commands  (subscriber, sensor_msgs/JointState)  — apply joint positions from MoveIt2 bridge
  - /clock  (publisher)  — sim clock for ROS2

Run inside the isaaclab_arena-cuda_gr00t_gn16 container:
  docker exec -it isaaclab_arena-cuda_gr00t_gn16 bash -c \
    "cd /workspaces/isaaclab_arena && \
     /isaac-sim/python.sh tools/test_crx5ia_robotiq85_standalone.py"

Add --headless to run without GUI.

In a separate terminal (host, with ROS2 sourced):
  # Watch joint states
  ros2 topic echo /joint_states

  # Send joint command (open gripper, move J2)
  ros2 topic pub --once /isaac_joint_commands sensor_msgs/msg/JointState \
    '{name: [J1,J2,J3,J4,J5,J6,robotiq_85_left_knuckle_joint], \
      position: [0.0,-0.8,0.0,0.0,-1.5708,0.0,0.0]}'
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

CONFIG = {"renderer": "RaytracedLighting", "headless": args.headless}
simulation_app = SimulationApp(CONFIG)

import numpy as np  # noqa: E402

import carb  # noqa: E402
import omni.graph.core as og  # noqa: E402
import usdrt.Sdf  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils import extensions, stage, viewports  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
_USD_PATH = os.path.join(
    _REPO_ROOT,
    "isaaclab_arena/embodiments/crx5ia/usd/crx5ia_robotiq85/crx5ia_robotiq85.usd",
)
_USD_PATH = os.path.realpath(_USD_PATH)

ROBOT_STAGE_PATH = "/CRX5iA_Robotiq85"
JOINT_STATE_TOPIC = "joint_states"
JOINT_CMD_TOPIC = "isaac_joint_commands"
CLOCK_TOPIC = "clock"

# --------------------------------------------------------------------------- #
# Verify USD exists
# --------------------------------------------------------------------------- #
if not os.path.isfile(_USD_PATH):
    carb.log_error(f"USD not found: {_USD_PATH}")
    simulation_app.close()
    sys.exit(1)

print(f"\n{'='*60}")
print(f"Loading USD: {_USD_PATH}")
print(f"{'='*60}\n")

# --------------------------------------------------------------------------- #
# Enable ROS2 bridge
# --------------------------------------------------------------------------- #
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# --------------------------------------------------------------------------- #
# Stage setup
# --------------------------------------------------------------------------- #
simulation_context = SimulationContext(stage_units_in_meters=1.0)

if not args.headless:
    viewports.set_camera_view(
        eye=np.array([1.5, 1.5, 1.2]),
        target=np.array([0.0, 0.0, 0.5]),
    )

# Load local CRX5iA + Robotiq 85 USD
from isaacsim.core.utils.prims import create_prim  # noqa: E402

create_prim(
    ROBOT_STAGE_PATH,
    "Xform",
    position=np.array([0.0, 0.0, 0.0]),
    usd_path=_USD_PATH,
)

simulation_app.update()

# --------------------------------------------------------------------------- #
# Print joint info (before physics init)
# --------------------------------------------------------------------------- #
from isaacsim.core.api.articulations import Articulation  # noqa: E402

simulation_context.initialize_physics()

robot = Articulation(ROBOT_STAGE_PATH)
robot.initialize()
robot.post_reset()

print(f"\n{'='*60}")
print(f"Joint names ({robot.num_dof} DOF):")
for i, name in enumerate(robot.dof_names):
    limits = robot.dof_properties["lower"][i], robot.dof_properties["upper"][i]
    print(f"  [{i:2d}] {name:<45s} limits=[{limits[0]:.3f}, {limits[1]:.3f}]")
print(f"{'='*60}\n")

# --------------------------------------------------------------------------- #
# OmniGraph: ROS2 bridge
# --------------------------------------------------------------------------- #
try:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", ROBOT_STAGE_PATH),
                ("PublishJointState.inputs:topicName", JOINT_STATE_TOPIC),
                ("SubscribeJointState.inputs:topicName", JOINT_CMD_TOPIC),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_STAGE_PATH)]),
            ],
        },
    )
    print(f"ROS2 OmniGraph created:")
    print(f"  Publishing:  /{JOINT_STATE_TOPIC}")
    print(f"  Subscribing: /{JOINT_CMD_TOPIC}")
    print(f"  Clock:       /{CLOCK_TOPIC}\n")
except Exception as e:
    carb.log_error(f"OmniGraph setup failed: {e}")
    raise

simulation_context.play()

# --------------------------------------------------------------------------- #
# Self-test: move through a few joint poses
# --------------------------------------------------------------------------- #
_TEST_POSES = [
    # J1..J6, gripper (robotiq_85_left_knuckle_joint)
    [0.0, -0.523, 0.0, 0.0, -1.5708, 0.0, 0.0],  # home
    [0.3, -0.8, 0.2, 0.1, -1.5708, 0.0, 0.0],  # slightly rotated
    [0.0, -0.523, 0.0, 0.0, -1.5708, 0.0, 0.8],  # close gripper
    [0.0, -0.523, 0.0, 0.0, -1.5708, 0.0, 0.0],  # home + open
]

_STEPS_PER_POSE = 120  # ~2 s at 60 Hz
_step = 0
_pose_idx = 0

print("Starting simulation. Self-test poses will run automatically.")
print("Ctrl+C to exit.\n")

while simulation_app.is_running():
    simulation_context.step(render=True)
    og.Controller.set(og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True)

    # Cycle through test poses
    if _step % _STEPS_PER_POSE == 0:
        if _pose_idx < len(_TEST_POSES):
            pose = _TEST_POSES[_pose_idx]
            print(f"  → Pose [{_pose_idx}]: {pose}")
            robot.set_joint_positions(
                np.array(pose, dtype=np.float32),
                joint_indices=list(range(robot.num_dof)),
            )
            _pose_idx = (_pose_idx + 1) % len(_TEST_POSES)
    _step += 1

simulation_context.stop()
simulation_app.close()
