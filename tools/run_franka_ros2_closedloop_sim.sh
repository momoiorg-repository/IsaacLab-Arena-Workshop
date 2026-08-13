#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# SPDX-License-Identifier: Apache-2.0
#
# Sim side of the two-container ROS2 closed-loop for the Franka workshop.
# Run this INSIDE the Isaac Sim container (it uses /isaac-sim/python.sh). It
# steps the Isaac Lab env and queries the GR00T policy over ROS2 instead of
# running GR00T in-process.
#
# Start the server first (in another container), with the SAME ROS_DOMAIN_ID:
#   bash docker/run_gr00t_ros2_server.sh --domain-id 31
#
# Then, inside the Isaac Sim container:
#   bash tools/run_franka_ros2_closedloop_sim.sh --domain-id 31
#
# Like the in-process closed loop (PDF §6), this uses --embodiment franka_joint
# (direct joint control), which differs from teleoperation's --embodiment franka.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"  # avoid 31, used by the plugsim stack
NAMESPACE="${NAMESPACE:-/gr00t_policy}"
NUM_EPISODES="${NUM_EPISODES:-10}"
ENV_NAME="${ENV_NAME:-table_pick_and_place}"
EMBODIMENT="${EMBODIMENT:-franka_joint}"
OBJECT="${OBJECT:-dex_cube}"
LIVESTREAM="${LIVESTREAM:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain-id)    ROS_DOMAIN_ID="$2"; shift 2 ;;
    --namespace)    NAMESPACE="$2"; shift 2 ;;
    --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
    --env)          ENV_NAME="$2"; shift 2 ;;
    --embodiment)   EMBODIMENT="$2"; shift 2 ;;
    --object)       OBJECT="$2"; shift 2 ;;
    --livestream)   LIVESTREAM="$2"; shift 2 ;;  # 0=off, 1=WebRTC public (cloud), 2=WebRTC private (local)
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Expose Isaac Sim's bundled ROS2 so the in-process ROS2 client can import rclpy.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_isaacsim_ros2_env.sh"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"
export PYTHONUNBUFFERED=1

echo "=== Franka ROS2 closed-loop (sim side) ==="
echo "  ROS_DOMAIN_ID : ${ROS_DOMAIN_ID}"
echo "  namespace     : ${NAMESPACE}"
echo "  num_episodes  : ${NUM_EPISODES}"
echo "  env/embodiment: ${ENV_NAME} / ${EMBODIMENT} / ${OBJECT}"
echo "  livestream    : ${LIVESTREAM} (0=off, 1=WebRTC public, 2=WebRTC private)"
echo ""

cd "${REPO_ROOT}"

LIVESTREAM="${LIVESTREAM}" /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --device cpu \
  --policy_type isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy \
  --remote_transport ros2 \
  --ros2_namespace "${NAMESPACE}" \
  --ros2_domain_id "${ROS_DOMAIN_ID}" \
  --policy_device cuda \
  --enable_cameras \
  --num_episodes "${NUM_EPISODES}" \
  "${ENV_NAME}" --embodiment "${EMBODIMENT}" --object "${OBJECT}"
