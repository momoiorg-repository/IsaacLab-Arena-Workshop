#!/usr/bin/env bash
# RE-ALIGN + re-firm-then-spiral (crack the COMBINED lat+tilt error). Soft window straightens the tilt,
# then re-firm so the spiral search runs on a straight peg and recovers the residual LATERAL. Sweep the
# soft-window length on the VLA-typical combined error (lat6+tilt6); STEPS=0 = soft-throughout (=5%).
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/realign3.csv"; echo "cond,off_mm,off_deg,stiff,steps,k,n,realized_lat,realized_tilt,fins_N" > "$CSV"
M="$LD/realign3.log"; echo "[realign3] START $(date -u +%FT%TZ)" | tee "$M"

run() {  # cond off_mm off_deg stiff steps
  local cond="$1" mm="$2" dg="$3" st="$4" sp="$5"
  local out="$LD/r3_${cond}.out"
  env BDASH_GRASP_OFF_MM="$mm" BDASH_GRASP_OFF_DEG="$dg" BDASH_REALIGN=1 BDASH_GRIP_STIFF="$st" \
      BDASH_REALIGN_STEPS="$sp" BDASH_GRASP_DIAG=1 \
    "$PY" scripts/bdash/run_eval_grid.py --policies bdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/r3_${cond}.csv" > "$out" 2>&1
  local kn k n lat tlt f
  kn=$(grep -E 'bdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1); k="${kn%/*}"; n="${kn#*/}"
  lat=$(grep -m1 graspdiag "$out" | grep -oE 'lat=[0-9.]+' | grep -oE '[0-9.]+')
  tlt=$(grep -m1 graspdiag "$out" | grep -oE 'tilt=[0-9.]+' | grep -oE '[0-9.]+')
  f=$(awk -F, 'NR==2{print $9}' "$RD/r3_${cond}.csv" 2>/dev/null)
  echo "${cond},${mm},${dg},${st},${sp},${k:-NA},${n:-NA},${lat:-NA},${tlt:-NA},${f:-NA}" >> "$CSV"
  echo "[realign3] ${cond} (steps=${sp}): ${kn:-NA} lat=${lat:-NA} tilt=${tlt:-NA} fins=${f:-NA}N" | tee -a "$M"
}

run comb_s0   6 6 200 0    # soft throughout (baseline for combined ~5%)
run comb_s25  6 6 200 25
run comb_s40  6 6 200 40
run comb_s60  6 6 200 60
run tilt8_s40 0 8 200 40   # pure tilt still recovered with the window?

echo "[realign3] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
cat "$CSV" | tee -a "$M"
