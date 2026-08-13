#!/usr/bin/env bash
# Combined-error correction (the scripted->real transfer test). Inject BOTH in-gripper lateral
# (BDASH_GRASP_OFF_MM) AND tilt (BDASH_GRASP_OFF_DEG) at once, and apply the §2.1 tactile bore-probe
# (BDASH_TACTILE_CAL), which localizes the bore vs the assumed tip -> corrects the TOTAL lateral tip
# error (grasp lateral + tilt-induced L*sin(phi)). Question: does the probe still recover when the peg
# is also TILTED, or does the residual peg-axis tilt jam (the real-VLA failure mode, 33 N)?
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/combined.csv"; echo "cond,off_mm,off_deg,k,n,realized_lat_mm,realized_tilt_deg,fins_med_N" > "$CSV"
M="$LD/combined.log"; echo "[combined] START $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD)" | tee "$M"

run() {  # cond off_mm off_deg extra_env
  local cond="$1" mm="$2" dg="$3" extra="$4"
  local tag="${cond}_m${mm}_d${dg}"; local out="$LD/cmb_${tag}.out"
  env BDASH_GRASP_OFF_MM="$mm" BDASH_GRASP_OFF_DEG="$dg" BDASH_GRASP_DIAG=1 $extra \
    "$PY" scripts/bdash/run_eval_grid.py --policies bdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/cmb_${tag}.csv" > "$out" 2>&1
  local kn k n lat tlt f
  kn=$(grep -E 'bdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1)
  k="${kn%/*}"; n="${kn#*/}"
  lat=$(grep -m1 graspdiag "$out" | grep -oE 'lat=[0-9.]+' | grep -oE '[0-9.]+')
  tlt=$(grep -m1 graspdiag "$out" | grep -oE 'tilt=[0-9.]+' | grep -oE '[0-9.]+')
  f=$(awk -F, 'NR==2{print $9}' "$RD/cmb_${tag}.csv" 2>/dev/null)
  echo "${cond},${mm},${dg},${k:-NA},${n:-NA},${lat:-NA},${tlt:-NA},${f:-NA}" >> "$CSV"
  echo "[combined] ${tag}: ${kn:-NA} lat=${lat:-NA}mm tilt=${tlt:-NA}deg fins=${f:-NA}N" | tee -a "$M"
}

# baseline (no correction) on the combined error
run blind 6 8 ""
# probe (cal) — isolate then combine
run cal 6 0 "BDASH_TACTILE_CAL=1 BDASH_EPISODE_S=60"   # pure lateral (sanity: expect ~100%)
run cal 0 8 "BDASH_TACTILE_CAL=1 BDASH_EPISODE_S=60"   # tilt only, corrected via the probe's tip alignment
run cal 6 4 "BDASH_TACTILE_CAL=1 BDASH_EPISODE_S=60"   # combined, mild tilt
run cal 6 8 "BDASH_TACTILE_CAL=1 BDASH_EPISODE_S=60"   # combined, realistic (THE key cell)

echo "[combined] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
column -t -s, "$CSV" | tee -a "$M"
