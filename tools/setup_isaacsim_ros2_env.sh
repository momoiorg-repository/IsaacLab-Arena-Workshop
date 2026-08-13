#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# SPDX-License-Identifier: Apache-2.0
#
# Source this script to expose Isaac Sim's *bundled* ROS2 (Humble, built for
# Isaac Sim's Python 3.11) so that `import rclpy` works inside /isaac-sim/python.sh.
#
#   source tools/setup_isaacsim_ros2_env.sh
#
# It does NOT require a system ROS2 install. Both the sim container and the GR00T
# server container (when run from the isaaclab_arena image) use the same bundled
# stack, so DDS types/RMW match exactly.
#
# Override the distro with ISAAC_ROS2_DISTRO=jazzy if needed (humble is verified).

ISAAC_ROS2_DISTRO="${ISAAC_ROS2_DISTRO:-humble}"
ISAAC_ROS2_ROOT="/isaac-sim/exts/isaacsim.ros2.bridge/${ISAAC_ROS2_DISTRO}"

if [ ! -d "${ISAAC_ROS2_ROOT}/rclpy" ]; then
    echo "[setup_isaacsim_ros2_env] ERROR: bundled ROS2 not found at ${ISAAC_ROS2_ROOT}/rclpy" >&2
    echo "[setup_isaacsim_ros2_env] Is this an Isaac Sim image? Set ISAAC_ROS2_DISTRO if needed." >&2
    return 1 2>/dev/null || exit 1
fi

export ROS_DISTRO="${ISAAC_ROS2_DISTRO}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export LD_LIBRARY_PATH="${ISAAC_ROS2_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ISAAC_ROS2_ROOT}/rclpy:${PYTHONPATH:-}"

echo "[setup_isaacsim_ros2_env] ROS_DISTRO=${ROS_DISTRO} RMW=${RMW_IMPLEMENTATION}"
echo "[setup_isaacsim_ros2_env] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>} (set it the same on both containers)"
