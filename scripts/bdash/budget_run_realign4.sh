#!/usr/bin/env bash
# RE-ALIGN (tilt) + a GENTLE lateral lever on the combined error (lat6+tilt6). The grasp-lateral is blind
# to the controller; try (a) RCC lateral compliance (nudge tip target along rim contact force Fxy) and
# (b) a wider spiral, on top of the re-align, to recover the residual lateral. Baseline (re-align only)=5%.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/realign4.csv"; echo "cond,k,n,realized_lat,realized_tilt,fins_N" > "$CSV"
M="$LD/realign4.log"; echo "[realign4] START $(date -u +%FT%TZ)" | tee "$M"
BASE="BDASH_GRASP_OFF_MM=6 BDASH_GRASP_OFF_DEG=6 BDASH_REALIGN=1 BDASH_GRIP_STIFF=200 BDASH_GRASP_DIAG=1"

run() {  # cond extra
  local cond="$1" extra="$2"; local out="$LD/r4_${cond}.out"
  env $BASE $extra \
    "$PY" scripts/bdash/run_eval_grid.py --policies bdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/r4_${cond}.csv" > "$out" 2>&1
  local kn k n lat tlt f
  kn=$(grep -E 'bdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1); k="${kn%/*}"; n="${kn#*/}"
  lat=$(grep -m1 graspdiag "$out" | grep -oE 'lat=[0-9.]+' | grep -oE '[0-9.]+')
  tlt=$(grep -m1 graspdiag "$out" | grep -oE 'tilt=[0-9.]+' | grep -oE '[0-9.]+')
  f=$(awk -F, 'NR==2{print $9}' "$RD/r4_${cond}.csv" 2>/dev/null)
  echo "${cond},${k:-NA},${n:-NA},${lat:-NA},${tlt:-NA},${f:-NA}" >> "$CSV"
  echo "[realign4] ${cond}: ${kn:-NA} lat=${lat:-NA} tilt=${tlt:-NA} fins=${f:-NA}N" | tee -a "$M"
}

run realign_only ""
run rccxy_002    "BDASH_INS_RCC_XY_GAIN=0.002"
run rccxy_005    "BDASH_INS_RCC_XY_GAIN=0.005"
run wide_spiral  "BDASH_INS_SPIRAL_RADIUS_MAX=0.013 BDASH_INS_SPIRAL_RADIUS_RATE=0.0001"
run rccxy_wide   "BDASH_INS_RCC_XY_GAIN=0.003 BDASH_INS_SPIRAL_RADIUS_MAX=0.013"

echo "[realign4] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
cat "$CSV" | tee -a "$M"
