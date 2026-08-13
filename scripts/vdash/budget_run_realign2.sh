#!/usr/bin/env bash
# In-hand RE-ALIGN follow-up (it WORKS: tilt8 5%->50% at stiff=200). Optimize softness, map the tilt
# curve, test the VLA-typical COMBINED error (lat6+tilt6), and check no regression on a clean grasp.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
PY=/isaac-sim/python.sh; RD=results/vdash/repro; LD=logs/vdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/realign2.csv"; echo "cond,off_mm,off_deg,grip_stiff,k,n,realized_lat,realized_tilt,fins_N" > "$CSV"
M="$LD/realign2.log"; echo "[realign2] START $(date -u +%FT%TZ)" | tee "$M"

run() {  # cond off_mm off_deg stiff
  local cond="$1" mm="$2" dg="$3" st="$4"
  local tag="${cond}"; local out="$LD/r2_${tag}.out"
  env VDASH_GRASP_OFF_MM="$mm" VDASH_GRASP_OFF_DEG="$dg" VDASH_REALIGN=1 VDASH_GRIP_STIFF="$st" VDASH_GRASP_DIAG=1 \
    "$PY" scripts/vdash/run_eval_grid.py --policies vdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/r2_${tag}.csv" > "$out" 2>&1
  local kn k n lat tlt f
  kn=$(grep -E 'vdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1); k="${kn%/*}"; n="${kn#*/}"
  lat=$(grep -m1 graspdiag "$out" | grep -oE 'lat=[0-9.]+' | grep -oE '[0-9.]+')
  tlt=$(grep -m1 graspdiag "$out" | grep -oE 'tilt=[0-9.]+' | grep -oE '[0-9.]+')
  f=$(awk -F, 'NR==2{print $9}' "$RD/r2_${tag}.csv" 2>/dev/null)
  echo "${cond},${mm},${dg},${st},${k:-NA},${n:-NA},${lat:-NA},${tlt:-NA},${f:-NA}" >> "$CSV"
  echo "[realign2] ${tag}: ${kn:-NA} lat=${lat:-NA} tilt=${tlt:-NA} fins=${f:-NA}N" | tee -a "$M"
}

run clean_s200   0  0 200   # no-regression check (clean grasp should stay ~high)
run tilt4_s200   0  4 200
run tilt8_s150   0  8 150   # softer
run tilt8_s100   0  8 100   # softer still
run tilt12_s200  0 12 200   # harder tilt
run comb_s200    6  6 200   # VLA-typical combined lateral+tilt

echo "[realign2] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
cat "$CSV" | tee -a "$M"
