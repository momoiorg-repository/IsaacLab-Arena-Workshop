#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers
# SPDX-License-Identifier: Apache-2.0
#
# Start the GR00T policy server as a ROS2 node in its own container.
#
# It reuses the isaaclab_arena GR00T image (which already ships GR00T + configs +
# Isaac Sim's bundled ROS2 Humble), so no extra build is needed. The server loads
# the GR00T model, then serves get_action/reset/... over ROS2 topics under
# --ros2_namespace, carrying the same msgpack payload as the ZeroMQ transport.
#
# Pair it with the sim side (in the Isaac Sim container):
#   bash tools/run_franka_ros2_closedloop_sim.sh
# Both must share the same ROS_DOMAIN_ID.
#
# The config's model_path (e.g. /models/workshop-franka-gr00tn1-6-checkpoints-15000)
# must exist in the container. Either put the checkpoint under --models-dir at that
# name, or point --checkpoint at any host dir and it will be mounted at the config's
# model_path automatically (no config edit needed).
#
# Usage:
#   bash docker/run_gr00t_ros2_server.sh [--domain-id N] [--namespace /gr00t_policy] \
#        [--config <path>] [--models-dir <host dir>] [--checkpoint <host dir>] [--image <name:tag>]
#
# Example (use the repo's local checkpoint):
#   bash docker/run_gr00t_ros2_server.sh --domain-id 42 --checkpoint "$PWD/models/franka-gr00t-n1-6-cube"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults (override via flags or env).
IMAGE="${IMAGE:-isaaclab_arena:cuda_gr00t_gn16}"
CONTAINER_NAME="${CONTAINER_NAME:-gr00t_ros2_server}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"  # avoid 31, used by the plugsim stack
NAMESPACE="${NAMESPACE:-/gr00t_policy}"
CONFIG="${CONFIG:-isaaclab_arena_gr00t/policy/config/franka_manip_gr00t_closedloop_config.yaml}"
POLICY_TYPE="${POLICY_TYPE:-isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy}"
MODELS_DIR="${MODELS_DIR:-$HOME/IsaacLab-Arena/models}"
CHECKPOINT="${CHECKPOINT:-}"
WORKDIR="/workspaces/isaaclab_arena"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain-id)   ROS_DOMAIN_ID="$2"; shift 2 ;;
    --namespace)   NAMESPACE="$2"; shift 2 ;;
    --config)      CONFIG="$2"; shift 2 ;;
    --policy-type) POLICY_TYPE="$2"; shift 2 ;;
    --models-dir)  MODELS_DIR="$2"; shift 2 ;;
    --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
    --image)       IMAGE="$2"; shift 2 ;;
    --name)        CONTAINER_NAME="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${MODELS_DIR}"

# Optionally bind-mount a checkpoint dir at the config's model_path.
CHECKPOINT_MOUNT=()
if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT_ABS="$(cd "${CHECKPOINT}" && pwd)"  # fails if dir doesn't exist
  MODEL_PATH_IN_CFG="$(sed -n 's/^[[:space:]]*model_path:[[:space:]]*//p' "${REPO_ROOT}/${CONFIG}" \
    | head -1 | tr -d "\"'" | sed 's/[[:space:]]*$//')"
  if [[ -z "${MODEL_PATH_IN_CFG}" ]]; then
    echo "ERROR: could not read model_path from ${REPO_ROOT}/${CONFIG}" >&2
    exit 1
  fi
  CHECKPOINT_MOUNT=(-v "${CHECKPOINT_ABS}:${MODEL_PATH_IN_CFG}:ro")
fi

echo "=== GR00T ROS2 policy server ==="
echo "  image          : ${IMAGE}"
echo "  container      : ${CONTAINER_NAME}"
echo "  ROS_DOMAIN_ID  : ${ROS_DOMAIN_ID}"
echo "  namespace      : ${NAMESPACE}"
echo "  config         : ${CONFIG}"
echo "  policy_type    : ${POLICY_TYPE}"
echo "  repo  (host)   : ${REPO_ROOT} -> ${WORKDIR}"
echo "  models(host)   : ${MODELS_DIR} -> /models"
if [[ -n "${CHECKPOINT}" ]]; then
  echo "  checkpoint     : ${CHECKPOINT_ABS} -> ${MODEL_PATH_IN_CFG} (ro)"
fi
echo ""

# Remove a stale container with the same name.
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

SERVER_CMD="\
source ${WORKDIR}/tools/setup_isaacsim_ros2_env.sh && \
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && \
export PYTHONPATH=${WORKDIR}:\${PYTHONPATH:-} && \
export PYTHONUNBUFFERED=1 && \
cd ${WORKDIR} && \
/isaac-sim/python.sh -m isaaclab_arena.remote_policy.remote_policy_server_runner \
  --transport ros2 \
  --ros2_namespace ${NAMESPACE} \
  --ros2_domain_id ${ROS_DOMAIN_ID} \
  --policy_type ${POLICY_TYPE} \
  --policy_config_yaml_path ${CONFIG} \
  --policy_device cuda"

exec docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --net=host \
  --ipc=host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -v "${REPO_ROOT}:${WORKDIR}" \
  -v "${MODELS_DIR}:/models" \
  "${CHECKPOINT_MOUNT[@]}" \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "${SERVER_CMD}"
