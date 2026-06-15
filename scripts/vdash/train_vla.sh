#!/usr/bin/env bash
# Fine-tune GR00T N1.6 on the V-DASH pick->handoff demos (brief v3 VLA training).
#
# Pipeline:
#   1. record_vla_demos.py        -> datasets/vdash/vla_pick_handoff.hdf5   (M9)
#   2. convert_hdf5_to_lerobot.py -> datasets/vdash/vla_pick_handoff/lerobot (GR00T-LeRobot)
#        /isaac-sim/python.sh isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
#            --yaml_file isaaclab_arena_gr00t/lerobot/config/vdash_pick_handoff_config.yaml
#   3. THIS SCRIPT                -> fine-tunes GR00T N1.6 on that dataset.
#
# Run INSIDE the GR00T container (built with `./docker/run_docker.sh -g`):
#   docker exec -u an isaaclab_arena-cuda_gr00t_gn16 bash -lc \
#     'cd /workspaces/isaaclab_arena && bash scripts/vdash/train_vla.sh'
# All knobs are overridable via env vars, e.g.:
#   MAX_STEPS=20000 GLOBAL_BATCH_SIZE=32 NUM_GPUS=2 bash scripts/vdash/train_vla.sh
#
# ── HARDWARE NOTE ──────────────────────────────────────────────────────────────
# GR00T N1.6 is a 3B VLA and launch_finetune.py loads it in fp32
# (load_bf16=False, backbone_trainable_params_fp32=True). A full fine-tune needs a
# large-VRAM GPU (A100/H100-class, ~40GB+). The local RTX 5080 (16GB) will OOM —
# run this on a bigger GPU / cloud. If you must shrink memory, lower GLOBAL_BATCH_SIZE
# (raise GRAD_ACCUM to keep the effective batch), keep TUNE_LLM/TUNE_VISUAL off, and
# reduce NUM_SHARDS_PER_EPOCH.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root (/workspaces/isaaclab_arena)
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# ── Paths ───────────────────────────────────────────────────────────────────
# Base checkpoint to start from. Default: official N1.6 base (HF, may require `huggingface-cli login`).
# Warm-start alternative (already on disk): models/franka-gr00t-n1-6-cube
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"
DATASET_PATH="${DATASET_PATH:-datasets/vdash/vla_pick_handoff/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-models/vdash-gr00t-n1-6-pick-handoff}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"   # franka modality registers under NEW_EMBODIMENT

# ── Training hyperparameters (defaults mirror the prior franka run) ───────────
MAX_STEPS="${MAX_STEPS:-15000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
NUM_GPUS="${NUM_GPUS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
USE_WANDB="${USE_WANDB:-false}"   # set "true" after `wandb login`

# What to fine-tune (matches the franka N1.6 reference run: projector + diffusion head only).
TUNE_LLM="${TUNE_LLM:-false}"
TUNE_VISUAL="${TUNE_VISUAL:-false}"
TUNE_PROJECTOR="${TUNE_PROJECTOR:-true}"
TUNE_DIFFUSION="${TUNE_DIFFUSION:-true}"

bool_flag() { [ "$2" = "true" ] && echo "--$1" || echo "--no-$1"; }

if [ ! -d "$DATASET_PATH/meta" ]; then
  echo "ERROR: dataset not found at $DATASET_PATH (expected meta/, data/, videos/)." >&2
  echo "Run the conversion first (step 2 in the header)." >&2
  exit 1
fi

echo "=== GR00T fine-tune: V-DASH pick->handoff ==="
echo "  base        : $BASE_MODEL_PATH"
echo "  dataset     : $DATASET_PATH"
echo "  output      : $OUTPUT_DIR"
echo "  embodiment  : $EMBODIMENT_TAG"
echo "  steps/bs/gpu: $MAX_STEPS / $GLOBAL_BATCH_SIZE (x$GRAD_ACCUM accum) / $NUM_GPUS"
echo "  tune        : llm=$TUNE_LLM visual=$TUNE_VISUAL projector=$TUNE_PROJECTOR diffusion=$TUNE_DIFFUSION"

exec /isaac-sim/python.sh isaaclab_arena_gr00t/scripts/launch_finetune.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --modality-config-path "$MODALITY_CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-ratio "$WARMUP_RATIO" \
  --weight-decay "$WEIGHT_DECAY" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --num-gpus "$NUM_GPUS" \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --num-shards-per-epoch "$NUM_SHARDS_PER_EPOCH" \
  "$(bool_flag tune-llm "$TUNE_LLM")" \
  "$(bool_flag tune-visual "$TUNE_VISUAL")" \
  "$(bool_flag tune-projector "$TUNE_PROJECTOR")" \
  "$(bool_flag tune-diffusion-model "$TUNE_DIFFUSION")" \
  "$(bool_flag use-wandb "$USE_WANDB")"
