#!/usr/bin/env bash
# MATCHED CONTROL: same code, gates OFF (no re-align, no RCC), n=20 seed 0 -> isolates the combo effect.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
export HF_HOME=/home/an/.cache/huggingface
export HF_TOKEN="$(cat /home/an/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
M="$LD/vla_baseline_n20.log"; out="$LD/vla_baseline_n20.out"
echo "[vlabase20] START $(date -u +%FT%TZ)" | tee "$M"
BDASH_VLA_DEBUG=1 BDASH_GRASP_DIAG=1 \
  "$PY" scripts/bdash/run_eval_grid.py --policies bdash_vla_v6_recovery \
  --clearances 2.0 --levels L1 --num_episodes 20 --seed 0 \
  --log_dir logs/bdash --no_resume --out "$RD/eval_v6_baseline_n20.csv" > "$out" 2>&1
echo "[vlabase20] exit=$? $(date -u +%FT%TZ)" | tee -a "$M"
grep -E "bdash_vla_v6_recovery c=" "$out" | tee -a "$M"
tail -1 "$RD/eval_v6_baseline_n20.csv" 2>/dev/null | tee -a "$M"
echo "[vlabase20] DONE" | tee -a "$M"
