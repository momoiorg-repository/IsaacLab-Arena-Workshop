#!/usr/bin/env bash
# DECISIVE: real VLA + FULL COMBO (in-hand re-align for TILT + RCC-xy for LATERAL) — the first version
# that handles BOTH grasp-error axes GENTLY (no probe touches -> should transfer, unlike the probe->0%).
# Scripted: clean 100% (no regression), tilt8 40-50%, combined lat6+tilt6 5%->~15-20%. Does it lift the
# 20% real-VLA baseline? 12 ep (~1h).
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
export HF_HOME=/home/an/.cache/huggingface
export HF_TOKEN="$(cat /home/an/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY=/isaac-sim/python.sh; RD=results/vdash/repro; LD=logs/vdash/repro; mkdir -p "$RD" "$LD"
M="$LD/vla_realign_rcc.log"; out="$LD/vla_realign_rcc.out"
echo "[vlarr] START $(date -u +%FT%TZ) token_len=${#HF_TOKEN}" | tee "$M"
VDASH_VLA_DEBUG=1 VDASH_REALIGN=1 VDASH_GRIP_STIFF=200 VDASH_INS_RCC_XY_GAIN=0.005 VDASH_GRASP_DIAG=1 \
  "$PY" scripts/vdash/run_eval_grid.py --policies vdash_vla_v6_recovery \
  --clearances 2.0 --levels L1 --num_episodes 12 --seed 0 \
  --log_dir logs/vdash --no_resume --out "$RD/eval_v6_realign_rcc.csv" > "$out" 2>&1
echo "[vlarr] exit=$? $(date -u +%FT%TZ)" | tee -a "$M"
grep -E "vdash_vla_v6_recovery c=" "$out" | tee -a "$M"
grep -m1 -aE "\[realign\] ON" "$out" | tee -a "$M"
grep -iE "gated repo|401|OfflineMode" "$out" | tail -2 | tee -a "$M"
tail -2 "$RD/eval_v6_realign_rcc.csv" 2>/dev/null | tee -a "$M"
echo "[vlarr] DONE" | tee -a "$M"
