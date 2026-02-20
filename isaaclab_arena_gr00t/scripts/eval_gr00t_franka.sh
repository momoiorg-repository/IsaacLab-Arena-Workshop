#!/bin/bash
set -e

# Add submodules to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/submodules/Isaac-GR00T

# Default variables
DATASET_PATH="/workspaces/isaaclab_arena/output/table_pick_place_cube_dataset/lerobot"
CHECKPOINT_DIR="/models/checkpoint-2000"
# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -d $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Evaluating Checkpoint: $LATEST_CHECKPOINT"

python submodules/Isaac-GR00T/gr00t/eval/open_loop_eval.py \
    --dataset-path $DATASET_PATH \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path $LATEST_CHECKPOINT \
    --traj-ids 0 \
    --action-horizon 16 \
    --steps 200 \
    --modality-keys single_arm gripper

echo "Open loop evaluation complete. Check /tmp/open_loop_eval for results."
