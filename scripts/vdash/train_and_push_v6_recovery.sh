#!/usr/bin/env bash
# Train GR00T N1.7 on the corrected (moving-frames) v5-recovery handoff dataset and
# push the result to the HF model repo. CLOUD ONLY — the 3B fp32 fine-tune OOMs the
# local RTX 5080 (16GB); run this on an A100/H100-class box.
#
# Pipeline:
#   1. pull dataset  umegan/vdash-pick-handoff-v5-recovery  (1000 demos, moving frames, camera-fixed)
#   2. fine-tune     nvidia/GR00T-N1.7-3B  (frozen ViT — unfreezing regressed; see notes)
#   3. push weights  -> umegan/vdash-gr00t-n1-7-pick-handoff-v6-recovery
#
# Prereqs on the cloud box: this repo + Isaac-GR00T (n1.7-release) checked out, the gn17
# deps available (transformers==4.57.3 etc.), and `hf auth login` as umegan (or export HF_TOKEN).
#
# Run:
#   bash scripts/vdash/train_and_push_v6_recovery.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

DATA_REPO="${DATA_REPO:-umegan/vdash-pick-handoff-v5-recovery}"
MODEL_REPO="${MODEL_REPO:-umegan/vdash-gr00t-n1-7-pick-handoff-v6-recovery}"
DATASET_PATH="${DATASET_PATH:-datasets/vdash/vla_pick_handoff_v5_recovery/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-models/vdash-gr00t-n1-7-pick-handoff-v6-recovery}"

# ── 1. Pull the dataset if it isn't already local ────────────────────────────
if [ ! -d "$DATASET_PATH/meta" ]; then
  echo "=== downloading dataset $DATA_REPO -> $DATASET_PATH ==="
  hf download "$DATA_REPO" --repo-type dataset --local-dir "$DATASET_PATH"
fi

# ── 2. Fine-tune (corrected recipe: 112,820 samples / batch32 = 3,525 steps/epoch ──
#       => 10k steps ≈ 2.8 epochs; frozen ViT, A/B the saved checkpoints, don't trust the last) ──
echo "=== fine-tuning -> $OUTPUT_DIR ==="
TUNE_VISUAL=false TUNE_LLM=false TUNE_PROJECTOR=true TUNE_DIFFUSION=true \
GLOBAL_BATCH_SIZE=32 MAX_STEPS=10000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=10 \
DATASET_PATH="$DATASET_PATH" OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/vdash/train_vla.sh

# ── 3. Push the latest checkpoint's weights to the HF model repo ──────────────
#       (uploads the highest-step checkpoint; pull a different one for A/B by hand.)
LAST_CKPT="$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
UPLOAD_DIR="${LAST_CKPT:-$OUTPUT_DIR}"   # fall back to OUTPUT_DIR root if no checkpoint-* dirs
echo "=== uploading $UPLOAD_DIR -> $MODEL_REPO ==="
hf upload "$MODEL_REPO" "$UPLOAD_DIR" . --repo-type model

echo "ALL_DONE -> pushed $UPLOAD_DIR to $MODEL_REPO"
