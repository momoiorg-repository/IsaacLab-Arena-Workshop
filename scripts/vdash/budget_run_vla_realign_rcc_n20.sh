#!/usr/bin/env bash
# Matched n=20 (seed 0) combo run -> directly comparable to the stored 4/20=20% baseline.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
export HF_HOME=/home/an/.cache/huggingface
export HF_TOKEN="$(cat /home/an/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY=/isaac-sim/python.sh; RD=results/vdash/repro; LD=logs/vdash/repro; mkdir -p "$RD" "$LD"
M="$LD/vla_realign_rcc_n20.log"; out="$LD/vla_realign_rcc_n20.out"
echo "[vlarr20] START $(date -u +%FT%TZ)" | tee "$M"
VDASH_VLA_DEBUG=1 VDASH_REALIGN=1 VDASH_GRIP_STIFF=200 VDASH_INS_RCC_XY_GAIN=0.005 VDASH_GRASP_DIAG=1 \
  "$PY" scripts/vdash/run_eval_grid.py --policies vdash_vla_v6_recovery \
  --clearances 2.0 --levels L1 --num_episodes 20 --seed 0 \
  --log_dir logs/vdash --no_resume --out "$RD/eval_v6_realign_rcc_n20.csv" > "$out" 2>&1
echo "[vlarr20] exit=$? $(date -u +%FT%TZ)" | tee -a "$M"
grep -E "vdash_vla_v6_recovery c=" "$out" | tee -a "$M"
tail -1 "$RD/eval_v6_realign_rcc_n20.csv" 2>/dev/null | tee -a "$M"
echo "[vlarr20] DONE" | tee -a "$M"
