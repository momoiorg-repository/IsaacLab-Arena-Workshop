export DATASET_DIR="/workspaces/isaaclab_arena/output"

# Record
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --device cpu \
  --enable_cameras \
  --dataset_file /workspaces/isaaclab_arena/output/table_pick_place_cube.hdf5 \
  --num_demos 3 \
  --num_success_steps 2 \
  table_pick_and_place \
  --embodiment franka \
  --teleop_device keyboard

# Replay
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/replay_demos.py --dataset_file /workspaces/isaaclab_arena/output/table_pick_place_cube.hdf5 --device cpu  table_pick_and_place

# Annotate
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/annotate_demos.py  --device cpu \
  --input_file  /workspaces/isaaclab_arena/output/table_pick_place_cube.hdf5 \
  --output_file /workspaces/isaaclab_arena/output/table_pick_place_cube_annotated.hdf5 \
  --mimic \
  --enable_cameras \
  table_pick_and_place \
  --object dex_cube \
  --embodiment franka

# Generate Dataset
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
  --device cpu \
  --enable_cameras \
  --input_file /workspaces/isaaclab_arena/output/table_pick_place_cube_annotated.hdf5 \
  --output_file /workspaces/isaaclab_arena/output/table_pick_place_cube_dataset.hdf5 \
  --num_envs 5  \
  --generation_num_trials 20 \
  --mimic \
  table_pick_and_place \
  --object dex_cube \
  --embodiment franka

# Transform to Lerobot (Custom script for Franka)
# Transform to Lerobot (Custom script for Franka)
python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py --yaml_file isaaclab_arena_gr00t/lerobot/config/franka_pick_place_config.yaml

# Train Gr00t (Must be run inside Docker)
./docker/run_docker.sh -g "bash isaaclab_arena_gr00t/scripts/train_gr00t_franka.sh"

# Evaluate Gr00t (Must be run inside Docker)
./docker/run_docker.sh -g "bash isaaclab_arena_gr00t/scripts/eval_gr00t_franka.sh"

export DATASET_PATH="/workspaces/isaaclab_arena/output/table_pick_place_cube_dataset"
export OUTPUT_DIR="/workspaces/isaaclab_arena/output/gr00t_franka_finetune"
export MODALITY_CONFIG="isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py"

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
    --use-wandb \
    --global-batch-size 8 \
    --gradient-accumulation-steps 4 \
    --dataloader-num-workers 2
