#!/usr/bin/env bash
# DECISIVE: real VLA + in-hand RE-ALIGN (compliant grip when tip loaded on rim; stiff=200). Scripted:
# clean 100% (no regression), tilt4 70%, tilt8 50% (beats tip-shift 30%). It's GENTLE (no probe touches)
# so it should transfer (unlike the probe -> 0%). Fixes the topple (dominant real failure). Does it lift
# the 20% no-probe baseline? 12 ep for a faster signal (~1h).
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
export HF_HOME=/home/an/.cache/huggingface
export HF_TOKEN="$(cat /home/an/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY=/isaac-sim/python.sh; RD=results/vdash/repro; LD=logs/vdash/repro; mkdir -p "$RD" "$LD"
M="$LD/vla_realign.log"; out="$LD/vla_realign.out"
echo "[vlarealign] START $(date -u +%FT%TZ) token_len=${#HF_TOKEN}" | tee "$M"
VDASH_VLA_DEBUG=1 VDASH_REALIGN=1 VDASH_GRIP_STIFF=200 VDASH_GRASP_DIAG=1 \
  "$PY" scripts/vdash/run_eval_grid.py --policies vdash_vla_v6_recovery \
  --clearances 2.0 --levels L1 --num_episodes 12 --seed 0 \
  --log_dir logs/vdash --no_resume --out "$RD/eval_v6_realign.csv" > "$out" 2>&1
echo "[vlarealign] exit=$? $(date -u +%FT%TZ)" | tee -a "$M"
grep -E "vdash_vla_v6_recovery c=" "$out" | tee -a "$M"
grep -m1 -aE "\[realign\] ON" "$out" | tee -a "$M"
grep -iE "gated repo|401|OfflineMode" "$out" | tail -2 | tee -a "$M"
tail -2 "$RD/eval_v6_realign.csv" 2>/dev/null | tee -a "$M"
echo "[vlarealign] DONE" | tee -a "$M"
