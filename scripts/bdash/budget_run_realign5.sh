#!/usr/bin/env bash
# Optimize RE-ALIGN + RCC-xy (both axes) on the combined error, + no-regression checks. re-align fixes
# TILT (pivot), RCC-xy recovers LATERAL (nudge tip along rim contact Fxy). Combined 5%->20% at rcc~0.002-5.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/realign5.csv"; echo "cond,off_mm,off_deg,rcc,k,n,lat,tilt,fins_N" > "$CSV"
M="$LD/realign5.log"; echo "[realign5] START $(date -u +%FT%TZ)" | tee "$M"

run() {  # cond off_mm off_deg rcc
  local cond="$1" mm="$2" dg="$3" rcc="$4"; local out="$LD/r5_${cond}.out"
  env BDASH_GRASP_OFF_MM="$mm" BDASH_GRASP_OFF_DEG="$dg" BDASH_REALIGN=1 BDASH_GRIP_STIFF=200 \
      BDASH_INS_RCC_XY_GAIN="$rcc" BDASH_GRASP_DIAG=1 \
    "$PY" scripts/bdash/run_eval_grid.py --policies bdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/r5_${cond}.csv" > "$out" 2>&1
  local kn k n lat tlt f
  kn=$(grep -E 'bdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1); k="${kn%/*}"; n="${kn#*/}"
  lat=$(grep -m1 graspdiag "$out" | grep -oE 'lat=[0-9.]+' | grep -oE '[0-9.]+')
  tlt=$(grep -m1 graspdiag "$out" | grep -oE 'tilt=[0-9.]+' | grep -oE '[0-9.]+')
  f=$(awk -F, 'NR==2{print $9}' "$RD/r5_${cond}.csv" 2>/dev/null)
  echo "${cond},${mm},${dg},${rcc},${k:-NA},${n:-NA},${lat:-NA},${tlt:-NA},${f:-NA}" >> "$CSV"
  echo "[realign5] ${cond} rcc=${rcc}: ${kn:-NA} lat=${lat:-NA} tilt=${tlt:-NA} fins=${f:-NA}N" | tee -a "$M"
}

run comb_rcc003  6 6 0.003
run comb_rcc008  6 6 0.008
run comb_rcc012  6 6 0.012
run clean_rcc005 0 0 0.005   # regression: clean grasp should stay ~100%
run tilt8_rcc005 0 8 0.005   # regression: pure tilt should stay ~50%

echo "[realign5] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
cat "$CSV" | tee -a "$M"
