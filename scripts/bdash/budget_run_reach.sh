#!/usr/bin/env bash
# Probe-REACH extension. The default 12/6 tactile grid recovered the hard combined cell (6mm+8deg ~=
# 10.4mm total lateral) only 8/20 = 40%; the estimate returned 0 at >=10mm (bore-detect fails near the
# grid edge / funnel lip). Test whether a WIDER and/or FINER probe grid (BDASH_CAL_HALF_MM/STEP_MM)
# recovers that cell. If a grid lifts 40% -> high, escalate to harder cells then the real VLA.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/reach.csv"; echo "off_mm,off_deg,half_mm,step_mm,k,n,realized_lat_mm,fins_med_N" > "$CSV"
M="$LD/reach.log"; echo "[reach] START $(date -u +%FT%TZ)" | tee "$M"

run() {  # off_mm off_deg half step
  local mm="$1" dg="$2" half="$3" step="$4"
  local tag="m${mm}_d${dg}_h${half}_s${step}"; local out="$LD/reach_${tag}.out"
  env BDASH_GRASP_OFF_MM="$mm" BDASH_GRASP_OFF_DEG="$dg" BDASH_TACTILE_CAL=1 \
      BDASH_CAL_HALF_MM="$half" BDASH_CAL_STEP_MM="$step" BDASH_EPISODE_S=90 BDASH_GRASP_DIAG=1 \
      "$PY" scripts/bdash/run_eval_grid.py --policies bdash_scripted --clearances 2.0 --levels L1 \
      --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/reach_${tag}.csv" > "$out" 2>&1
  local kn k n lat f
  kn=$(grep -E 'bdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1)
  k="${kn%/*}"; n="${kn#*/}"
  lat=$(grep -m1 graspdiag "$out" | grep -oE 'lat=[0-9.]+' | grep -oE '[0-9.]+')
  f=$(awk -F, 'NR==2{print $9}' "$RD/reach_${tag}.csv" 2>/dev/null)
  echo "${mm},${dg},${half},${step},${k:-NA},${n:-NA},${lat:-NA},${f:-NA}" >> "$CSV"
  echo "[reach] ${tag}: ${kn:-NA} lat=${lat:-NA}mm fins=${f:-NA}N" | tee -a "$M"
}

# hard cell 6mm+8deg (~10.4mm total lateral); vary the probe grid
run 6 8 12 6    # baseline (default 5x5) -> expect ~40%
run 6 8 16 8    # wider, same ~25 pts (bore well inside grid)
run 6 8 12 4    # finer step (7x7), same width
run 6 8 18 6    # wider + denser (7x7)

echo "[reach] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
cat "$CSV" | tee -a "$M"
