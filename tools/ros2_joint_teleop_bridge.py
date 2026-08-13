#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
ROS2 joint teleop bridge for CRX5iA + Robotiq 85 in Isaac Lab simulation.

Subscribes to a ROS2 sensor_msgs/JointState topic and feeds the joint positions
as actions to the Isaac Lab sim environment for demo collection.

Usage (run inside Docker with ROS2 sourced, before launching Isaac Lab):
  # Terminal 1 — this bridge writes commands to shared file
  python tools/ros2_joint_teleop_bridge.py --output /tmp/crx5ia_joint_cmd.json

  # Terminal 2 — publish joint states from e.g. joint_state_publisher_gui
  ros2 run joint_state_publisher_gui joint_state_publisher_gui

The bridge writes the latest joint positions to /tmp/crx5ia_joint_cmd.json,
which the Isaac Lab teleop script reads via --teleop_device ros2_file.

Alternatively, if rclpy is available inside the Isaac Sim container:
  docker exec isaaclab_arena-latest python tools/ros2_joint_teleop_bridge.py

Joint order (7 DOF): J1, J2, J3, J4, J5, J6, finger_joint
Topic: /joint_states (sensor_msgs/JointState)
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

JOINT_ORDER = ["J1", "J2", "J3", "J4", "J5", "J6", "finger_joint"]
DEFAULT_JOINT_POS = [0.0, -0.523, 0.0, 0.0, -1.5708, 0.0, 0.0]


def run_bridge(topic: str, output_path: str, rate_hz: float) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError:
        print("ERROR: rclpy not available. Install ROS2 or source the workspace.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(output_path)

    class JointBridgeNode(Node):
        def __init__(self):
            super().__init__("crx5ia_joint_teleop_bridge")
            self._latest = dict(zip(JOINT_ORDER, DEFAULT_JOINT_POS))
            self.create_subscription(JointState, topic, self._callback, 10)
            period = 1.0 / rate_hz
            self.create_timer(period, self._write)
            self.get_logger().info(f"Bridging {topic} → {out_path}")

        def _callback(self, msg: JointState):
            name_to_pos = dict(zip(msg.name, msg.position))
            for joint in JOINT_ORDER:
                if joint in name_to_pos:
                    self._latest[joint] = float(name_to_pos[joint])

        def _write(self):
            payload = {"joints": [self._latest[j] for j in JOINT_ORDER], "order": JOINT_ORDER}
            out_path.write_text(json.dumps(payload))

    rclpy.init()
    node = JointBridgeNode()
    try:
        with contextlib.suppress(KeyboardInterrupt):
            rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_demo_mode(output_path: str) -> None:
    """Write a static default pose for testing without a live ROS2 topic."""
    out_path = Path(output_path)
    payload = {"joints": DEFAULT_JOINT_POS, "order": JOINT_ORDER}
    out_path.write_text(json.dumps(payload))
    print(f"Demo mode: wrote default pose to {out_path}")
    print("Edit the file manually or pipe joint values from any source.")
    print("Press Ctrl+C to stop.")
    with contextlib.suppress(KeyboardInterrupt):
        while True:
            time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", default="/joint_states", help="ROS2 JointState topic to subscribe to")
    parser.add_argument("--output", default="/tmp/crx5ia_joint_cmd.json", help="Path to write joint command JSON")
    parser.add_argument("--rate", type=float, default=30.0, help="Write rate in Hz")
    parser.add_argument("--demo", action="store_true", help="Write static default pose without ROS2 (for testing)")
    args = parser.parse_args()

    if args.demo:
        run_demo_mode(args.output)
    else:
        run_bridge(args.topic, args.output, args.rate)


if __name__ == "__main__":
    main()
