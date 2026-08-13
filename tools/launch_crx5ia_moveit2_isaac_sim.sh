#!/bin/bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# SPDX-License-Identifier: Apache-2.0
#
# Launch MoveIt2 for CRX5iA + Robotiq 85 connected to Isaac Sim.
#
# Prerequisites (run each in a separate terminal):
#
#   Terminal 1 — Isaac Sim (inside isaaclab_arena-cuda_gr00t_gn16 container):
#     docker exec -it isaaclab_arena-cuda_gr00t_gn16 bash -c \
#       "cd /workspaces/isaaclab_arena && \
#        /isaac-sim/python.sh tools/test_crx5ia_robotiq85_standalone.py"
#
#   Terminal 2 — MoveIt2 bridge (host, ROS2 sourced):
#     cd /home/an/Workspace/IsaacLab-Arena-An
#     python3 tools/moveit2_to_isaac_bridge.py
#
#   Terminal 3 — THIS SCRIPT (host, ROS2 sourced):
#     bash tools/launch_crx5ia_moveit2_isaac_sim.sh
#
# After RViz opens: use the MotionPlanning panel to plan and execute goals.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source ROS2 workspace if not already sourced
if [ -z "${ROS_DISTRO:-}" ]; then
    echo "ERROR: ROS2 not sourced. Run: source /opt/ros/humble/setup.bash"
    exit 1
fi

WS_SETUP="$REPO_ROOT/ros2_ws/install/setup.bash"
if [ -f "$WS_SETUP" ]; then
    source "$WS_SETUP"
else
    echo "WARNING: ros2_ws not built yet. Run: cd ros2_ws && colcon build"
fi

echo "=== Launching CRX5iA MoveIt2 (Isaac Sim mode) ==="
echo ""
echo "Topics:"
echo "  Isaac Sim publishes: /joint_states"
echo "  MoveIt2 bridge provides: /joint_trajectory_controller/follow_joint_trajectory"
echo "  Bridge forwards to: /isaac_joint_commands"
echo ""

ros2 launch fanuc_moveit_config fanuc_moveit.launch.py \
    robot_model:=crx5ia \
    use_mock:=true \
    moveit_controller_manager_config:="$(ros2 pkg prefix fanuc_moveit_config)/share/fanuc_moveit_config/config/moveit_controllers_isaac_sim.yaml" \
    "$@"
