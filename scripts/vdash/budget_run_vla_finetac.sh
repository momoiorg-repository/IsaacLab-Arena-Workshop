#!/usr/bin/env bash
# DECISIVE test: real GR00T VLA closed-loop with the FINER tactile probe (12/4), which lifted the
# scripted combined cell (6mm+8deg ~= 10.4mm total, ~VLA-typical) from 45% -> 90%. Compare to:
#   - seed-0 no-probe baseline 4/20 = 20% (eval_v6_recovery.csv)
#   - seed-0 default-probe (12/6) 3/20 = 15% (eval_v6_tactile.csv)
# Also log per-episode grasp GT (GRASP_DIAG) vs probe estimate (CAL_DIAG) vs result. 1 env + cameras.
# HF AUTH: the gated VLM backbone nvidia/Cosmos-Reason2-2B is fully cached (4.6G), but the GR00T loader
# makes an HF metadata API call that needs auth. Point HF at the existing token + cache (no offline:
# offline blocks that call; no-token 401s). Token is read silently (never echoed).
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
export HF_HOME=/home/an/.cache/huggingface
export HF_TOKEN="$(cat /home/an/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY=/isaac-sim/python.sh; RD=results/vdash/repro; LD=logs/vdash/repro; mkdir -p "$RD" "$LD"
M="$LD/vla_finetac.log"; out="$LD/vla_finetac.out"
echo "[vlafinetac] START $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD) token_len=${#HF_TOKEN}" | tee "$M"
VDASH_VLA_DEBUG=1 VDASH_TACTILE_CAL=1 VDASH_CAL_HALF_MM=12 VDASH_CAL_STEP_MM=4 VDASH_EPISODE_S=90 \
  VDASH_GRASP_DIAG=1 VDASH_CAL_DIAG=1 \
  "$PY" scripts/vdash/run_eval_grid.py --policies vdash_vla_v6_recovery \
  --clearances 2.0 --levels L1 --num_episodes 20 --seed 0 \
  --log_dir logs/vdash --no_resume --out "$RD/eval_v6_tactile_fine.csv" > "$out" 2>&1
ec=$?
echo "[vlafinetac] exit=$ec $(date -u +%FT%TZ)" | tee -a "$M"
grep -E "vdash_vla_v6_recovery c=" "$out" | tee -a "$M"
echo "=== auth/load errors (should be none now) ===" | tee -a "$M"
grep -iE "gated repo|401|OfflineMode|OSError.*hugging" "$out" | tail -3 | tee -a "$M"
echo "=== per-episode (grasp GT / probe estimate / result) ===" | tee -a "$M"
grep -iE "caldiag|graspdiag|decision=|ingrip" "$out" | tail -60 | tee -a "$M"
tail -2 "$RD/eval_v6_tactile_fine.csv" 2>/dev/null | tee -a "$M"
echo "[vlafinetac] DONE" | tee -a "$M"
