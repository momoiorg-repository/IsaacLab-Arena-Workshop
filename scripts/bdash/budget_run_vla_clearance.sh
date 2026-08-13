#!/usr/bin/env bash
# VLA clearance-tolerance characterization (no probe = clean baseline). The c=2 single-shot success is
# front-end-bounded (~20%); back-end can't compensate for the in-gripper grasp error. Does the VLA do
# better at a MORE FORGIVING clearance (c=3)? This maps the real VLA's success vs task tolerance (the
# scripted clearance cliff is clean-grasp only) — relevant to machine-tending where real parts vary.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
export HF_HOME=/home/an/.cache/huggingface
export HF_TOKEN="$(cat /home/an/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
M="$LD/vla_clearance.log"; out="$LD/vla_clearance.out"
echo "[vlaclr] START $(date -u +%FT%TZ) token_len=${#HF_TOKEN}" | tee "$M"
BDASH_VLA_DEBUG=1 BDASH_GRASP_DIAG=1 \
  "$PY" scripts/bdash/run_eval_grid.py --policies bdash_vla_v6_recovery \
  --clearances 2.0 3.0 --levels L1 --num_episodes 12 --seed 0 \
  --log_dir logs/bdash --no_resume --out "$RD/eval_v6_clearance.csv" > "$out" 2>&1
echo "[vlaclr] exit=$? $(date -u +%FT%TZ)" | tee -a "$M"
grep -E "bdash_vla_v6_recovery c=" "$out" | tee -a "$M"
cat "$RD/eval_v6_clearance.csv" 2>/dev/null | tee -a "$M"
echo "[vlaclr] DONE" | tee -a "$M"
