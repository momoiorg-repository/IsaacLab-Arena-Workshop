#!/usr/bin/env bash
# B-DASH full-pipeline demo: task text -> VLM subtask decomposition ->
# contract-based skill assignment -> gated execution -> results table.
# Usage (inside the gn17 container, or via docker exec):
#   BDASH_GEMINI_KEY_FILE=<key> ./scripts/bdash/run_bdash_pipeline.sh "タスク文" [clearance] [episodes] [extra flags...]
set -eu
cd /workspaces/isaaclab_arena
unset DISPLAY
TASK="${1:-insert the peg into the socket}"
CLEARANCE="${2:-2.0}"
EPISODES="${3:-10}"
shift $(( $# > 3 ? 3 : $# )) || true
/isaac-sim/python.sh -m isaaclab_arena.bdash.demo \
  --task "$TASK" --clearances "$CLEARANCE" --episodes "$EPISODES" \
  --out results/bdash/bdash_pipeline_table.csv "$@"
