#!/bin/bash

# Default variables
NUM_GPUS=1
BASE_MODEL="nvidia/GR00T-N1.6-3B"
# Assumes the dataset is in the output folder relative to workspace root
DATASET_PATH="/workspaces/isaaclab_arena/output/test_demo_dataset_100/lerobot"
OUTPUT_DIR="/workspaces/isaaclab_arena/output/gr00t_franka_finetune"
MODALITY_CONFIG="isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py"

echo "Starting Gr00t Finetuning for Franka..."
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Modality Config: $MODALITY_CONFIG"

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset directory not found at $DATASET_PATH"
    echo "Please run data conversion first."
    exit 1
fi

# Run finetuning
# Note: Using python from PATH (env usually set in Docker)
python -m torch.distributed.run --nproc_per_node=1 --standalone submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path $DATASET_PATH \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py \
    --num-gpus 1 \
    --output-dir $OUTPUT_DIR \
    --save-total-limit 2 \
    --save-steps 500 \
    --max-steps 2000 \
    --tune-projector \
    --tune-diffusion-model \
    --no-tune-llm \
    --no-tune-visual \
    --use-wandb \
    --global-batch-size 1 \
    --gradient-accumulation-steps 32 \
    --dataloader-num-workers 2

echo "Finetuning complete. Checkpoints saved to $OUTPUT_DIR"
