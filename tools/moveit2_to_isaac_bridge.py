#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
MoveIt2 → Isaac Sim bridge for CRX5iA + Robotiq 85.

Listens on the FollowJointTrajectory action server that MoveIt2 uses,
replays each trajectory point to Isaac Sim via /isaac_joint_commands
(sensor_msgs/JointState), and provides joint state feedback.

Run (host, with ROS2 sourced):
  python3 tools/moveit2_to_isaac_bridge.py

Then launch MoveIt2 (see tools/launch_crx5ia_moveit2.sh) and plan in RViz.

Architecture:
  RViz/MoveIt2
      │  FollowJointTrajectory action
      ▼
  moveit2_to_isaac_bridge.py   (this node)
      │  /isaac_joint_commands  (JointState)
      ▼
  Isaac Sim (test_crx5ia_robotiq85_standalone.py)
      │  /joint_states  (JointState)
      ▼
  MoveIt2 feedback
"""

import contextlib
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_JOINTS = ["J1", "J2", "J3", "J4", "J5", "J6"]
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]

# Topic Isaac Sim subscribes to
ISAAC_CMD_TOPIC = "/isaac_joint_commands"
# Topic Isaac Sim publishes to (for action feedback)
JOINT_STATE_TOPIC = "/joint_states"
# Action server name MoveIt2 targets (must match moveit_controllers.yaml)
ACTION_NAME = "/joint_trajectory_controller/follow_joint_trajectory"


class MoveItIsaacBridge(Node):
    def __init__(self):
        super().__init__("moveit2_isaac_bridge")

        self._cmd_pub = self.create_publisher(JointState, ISAAC_CMD_TOPIC, 10)

        self._latest_js: JointState | None = None
        self._js_sub = self.create_subscription(JointState, JOINT_STATE_TOPIC, self._js_cb, 10)

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            ACTION_NAME,
            self._execute_trajectory,
        )

        self.get_logger().info(f"Bridge ready — action server: {ACTION_NAME}")
        self.get_logger().info(f"Forwarding to: {ISAAC_CMD_TOPIC}")

    def _js_cb(self, msg: JointState):
        self._latest_js = msg

    def _execute_trajectory(self, goal_handle):
        traj = goal_handle.request.trajectory
        self.get_logger().info(f"Received trajectory: {len(traj.points)} points, joints={traj.joint_names}")

        for i, point in enumerate(traj.points):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            # Build JointState from trajectory point
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(traj.joint_names)
            msg.position = list(point.positions)

            self._cmd_pub.publish(msg)
            self.get_logger().info(f"  point [{i + 1}/{len(traj.points)}] pos={[f'{p:.3f}' for p in point.positions]}")

            # Wait for trajectory point duration
            if i < len(traj.points) - 1:
                dt_nsec = (
                    traj.points[i + 1].time_from_start.sec * 1_000_000_000
                    + traj.points[i + 1].time_from_start.nanosec
                    - point.time_from_start.sec * 1_000_000_000
                    - point.time_from_start.nanosec
                )
                sleep_sec = max(0.0, dt_nsec / 1e9)
                time.sleep(sleep_sec)

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        self.get_logger().info("Trajectory complete.")
        return result


def main():
    rclpy.init()
    node = MoveItIsaacBridge()
    try:
        with contextlib.suppress(KeyboardInterrupt):
            rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
