#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on the B-DASH pick->handoff demos (brief v3 VLA training).
#
# Pipeline:
#   1. record_vla_demos.py        -> datasets/bdash/vla_pick_handoff.hdf5   (M9)
#   2. convert_hdf5_to_lerobot.py -> datasets/bdash/vla_pick_handoff/lerobot (GR00T-LeRobot)
#        /isaac-sim/python.sh isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
#            --yaml_file isaaclab_arena_gr00t/lerobot/config/bdash_pick_handoff_config.yaml
#   3. THIS SCRIPT                -> fine-tunes GR00T N1.6 on that dataset.
#
# Run INSIDE the GR00T N1.7 container (built with `./docker/run_docker.sh -G`):
#   docker exec -u an isaaclab_arena-cuda_gr00t_gn17 bash -lc \
#     'cd /workspaces/isaaclab_arena && bash scripts/bdash/train_vla.sh'
# All knobs are overridable via env vars, e.g.:
#   MAX_STEPS=20000 GLOBAL_BATCH_SIZE=32 NUM_GPUS=2 bash scripts/bdash/train_vla.sh
#
# ── HARDWARE NOTE ──────────────────────────────────────────────────────────────
# GR00T N1.7 is a 3B VLA and launch_finetune.py loads it in fp32
# (load_bf16=False, backbone_trainable_params_fp32=True). A full fine-tune needs a
# large-VRAM GPU (A100/H100-class, ~40GB+). The local RTX 5080 (16GB) will OOM —
# run this on a bigger GPU / cloud. If you must shrink memory, lower GLOBAL_BATCH_SIZE
# (raise GRAD_ACCUM to keep the effective batch), keep TUNE_LLM/TUNE_VISUAL off, and
# reduce NUM_SHARDS_PER_EPOCH.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root (/workspaces/isaaclab_arena)
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# ── gn17 runtime fixes (from the vdi00035-002 production runs) ───────────────
# torch 2.7.0 (cu128) needs the pip-installed NVIDIA cu12 libs on the loader
# path or `import torch` dies on libcusparseLt.so.0. Not persisted in the image.
if ls -d /isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/*/lib >/dev/null 2>&1; then
  export LD_LIBRARY_PATH="$(ls -d /isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
fi
# Local N1.7 base checkpoint (what the chuck-full250/lift247 runs used) avoids a
# gated HF download inside the container. Only a default -- env override wins.
if [ -z "${BASE_MODEL_PATH:-}" ] && [ -d /workspaces/isaaclab_arena/models/GR00T-N1.7-3B-base ]; then
  BASE_MODEL_PATH=/workspaces/isaaclab_arena/models/GR00T-N1.7-3B-base
fi

# ── Paths ───────────────────────────────────────────────────────────────────
# Base checkpoint to start from. Default: official N1.7 base (HF, may require `huggingface-cli login`).
# N1.7 uses the Cosmos-Reason2-2B (Qwen3-VL) backbone, bundled in the checkpoint.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-datasets/bdash/vla_pick_handoff_v4/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-models/bdash-gr00t-n1-7-pick-handoff-v4}"
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
# adamw_torch fits the frozen-vision recipe in 42.3/46GB (measured, A40); TUNE_VISUAL=true
# OOMs with it -- use OPTIM=adamw_bnb_8bit for visual-tuning runs (vdi00035-002 field data).
# Passed as an env var (BDASH_OPTIM) because FinetuneConfig lives in the submodule and cannot
# grow a tyro flag without touching upstream.
export BDASH_OPTIM="${OPTIM:-adamw_torch}"
USE_WANDB="${USE_WANDB:-false}"   # set "true" after `wandb login`

# What to fine-tune (projector + diffusion head only; revisit for the N1.7 Qwen backbone).
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

echo "=== GR00T fine-tune: B-DASH pick->handoff ==="
echo "  base        : $BASE_MODEL_PATH"
echo "  dataset     : $DATASET_PATH"
echo "  output      : $OUTPUT_DIR"
echo "  embodiment  : $EMBODIMENT_TAG"
echo "  steps/bs/gpu: $MAX_STEPS / $GLOBAL_BATCH_SIZE (x$GRAD_ACCUM accum) / $NUM_GPUS"
echo "  tune        : llm=$TUNE_LLM visual=$TUNE_VISUAL projector=$TUNE_PROJECTOR diffusion=$TUNE_DIFFUSION"

# Not exec: run the trainer, then hand the outputs back to the host. Training runs as root
# inside the container (no matching user there), so everything under OUTPUT_DIR lands as
# root:root with weights at mode 600 -- the host user cannot even read them, and every
# host-side step (upload, eval copy) fails with PermissionError (measured on vdi00035-001).
/isaac-sim/python.sh isaaclab_arena_gr00t/scripts/launch_finetune.py \
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
STATUS=$?
chmod -R a+rX "$OUTPUT_DIR" 2>/dev/null || true
exit $STATUS
