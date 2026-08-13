#!/usr/bin/env bash
# Real VLA + GENTLER tactile probe (8/4 = 5x5, 25 touches; vs the 12/4=49-touch probe that destabilized
# the grasp -> 0/15 tip-overs). On the scripted test-bed 8/4 recovers the VLA-typical combined error
# (lat6+tilt6) at 17/20=85%. Question: does the gentler/smaller probe transfer to the real VLA (beat the
# 20% no-probe baseline) without toppling the grasp? 12 episodes for a faster signal (~1h).
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
export HF_HOME=/home/an/.cache/huggingface
export HF_TOKEN="$(cat /home/an/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
M="$LD/vla_gentle.log"; out="$LD/vla_gentle.out"
echo "[vlagentle] START $(date -u +%FT%TZ) token_len=${#HF_TOKEN}" | tee "$M"
BDASH_VLA_DEBUG=1 BDASH_TACTILE_CAL=1 BDASH_CAL_HALF_MM=8 BDASH_CAL_STEP_MM=4 BDASH_EPISODE_S=90 \
  BDASH_GRASP_DIAG=1 BDASH_CAL_DIAG=1 \
  "$PY" scripts/bdash/run_eval_grid.py --policies bdash_vla_v6_recovery \
  --clearances 2.0 --levels L1 --num_episodes 12 --seed 0 \
  --log_dir logs/bdash --no_resume --out "$RD/eval_v6_gentleprobe.csv" > "$out" 2>&1
echo "[vlagentle] exit=$? $(date -u +%FT%TZ)" | tee -a "$M"
grep -E "bdash_vla_v6_recovery c=" "$out" | tee -a "$M"
grep -iE "gated repo|401|OfflineMode" "$out" | tail -2 | tee -a "$M"
tail -2 "$RD/eval_v6_gentleprobe.csv" 2>/dev/null | tee -a "$M"
echo "[vlagentle] DONE" | tee -a "$M"
