#!/usr/bin/env bash
# Pool a matched (combo, baseline) pair at a given SEED (arg1), n=20 each, to tighten the combo-vs-baseline
# estimate. Appends one row per arm to results/bdash/repro/vla_pool.csv. Combo = re-align stiff200 + RCC-xy
# 0.005; baseline = gates OFF. Real-VLA runs are non-deterministic -> pooling across seeds is valid.
set -u
SEED="${1:-1}"
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
export HF_HOME=/home/an/.cache/huggingface
export HF_TOKEN="$(cat /home/an/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
POOL="$RD/vla_pool.csv"; M="$LD/vla_seedpair_s${SEED}.log"
[ -f "$POOL" ] || echo "arm,seed,n,success,rate,jsonl" > "$POOL"
echo "[seedpair s=$SEED] START $(date -u +%FT%TZ)" | tee "$M"

run_arm() {  # arm extra_env
  local arm="$1" extra="$2"; local out="$LD/pool_${arm}_s${SEED}.out"; local csv="$RD/pool_${arm}_s${SEED}.csv"
  env $extra BDASH_VLA_DEBUG=1 BDASH_GRASP_DIAG=1 \
    "$PY" scripts/bdash/run_eval_grid.py --policies bdash_vla_v6_recovery \
    --clearances 2.0 --levels L1 --num_episodes 20 --seed "$SEED" \
    --log_dir logs/bdash --no_resume --out "$csv" > "$out" 2>&1
  local line k n j
  line=$(tail -1 "$csv"); k=$(echo "$line" | awk -F, '{print $5}'); n=$(echo "$line" | awk -F, '{print $4}')
  j=$(echo "$line" | awk -F, '{print $NF}')
  echo "${arm},${SEED},${n},${k},$(awk "BEGIN{printf \"%.3f\", ${k:-0}/${n:-20}}"),${j}" >> "$POOL"
  echo "[seedpair s=$SEED] ${arm}: ${k}/${n}" | tee -a "$M"
}

run_arm combo    "BDASH_REALIGN=1 BDASH_GRIP_STIFF=200 BDASH_INS_RCC_XY_GAIN=0.005"
run_arm baseline ""

echo "[seedpair s=$SEED] DONE $(date -u +%FT%TZ)" | tee -a "$M"
echo "=== POOL so far ===" | tee -a "$M"; cat "$POOL" | tee -a "$M"
