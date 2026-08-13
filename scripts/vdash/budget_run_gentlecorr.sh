#!/usr/bin/env bash
# Gentle-correction test (autonomous). The aggressive tactile probe (49 touches) destabilized the real
# VLA grasp -> tip-over. The tilt tip-shift (VDASH_GATE_CORRECT) is GENTLE (no touching). Test, on the
# VLA-typical COMBINED error (lateral ~6mm + tilt ~6deg), whether gentle corrections recover it:
#   baseline / tip-shift / tip-shift+gentle-press / gentler-probe(8/4). If one recovers without
#   destabilizing -> candidate for a real-VLA run that could beat 20%.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
PY=/isaac-sim/python.sh; RD=results/vdash/repro; LD=logs/vdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/gentlecorr.csv"; echo "cond,off_mm,off_deg,k,n,realized_lat,realized_tilt,fins_med_N" > "$CSV"
M="$LD/gentlecorr.log"; echo "[gentlecorr] START $(date -u +%FT%TZ)" | tee "$M"
GENTLE="VDASH_INS_F_TARGET=2.0 VDASH_INS_PRESS_FORCE_BAND=1.5 VDASH_INS_CONTACT_SPEED=0.004"

run() {  # cond off_mm off_deg extra
  local cond="$1" mm="$2" dg="$3" extra="$4"
  local tag="${cond}_m${mm}_d${dg}"; local out="$LD/gc_${tag}.out"
  env VDASH_GRASP_OFF_MM="$mm" VDASH_GRASP_OFF_DEG="$dg" VDASH_GRASP_DIAG=1 $extra \
    "$PY" scripts/vdash/run_eval_grid.py --policies vdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/gc_${tag}.csv" > "$out" 2>&1
  local kn k n lat tlt f
  kn=$(grep -E 'vdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1); k="${kn%/*}"; n="${kn#*/}"
  lat=$(grep -m1 graspdiag "$out" | grep -oE 'lat=[0-9.]+' | grep -oE '[0-9.]+')
  tlt=$(grep -m1 graspdiag "$out" | grep -oE 'tilt=[0-9.]+' | grep -oE '[0-9.]+')
  f=$(awk -F, 'NR==2{print $9}' "$RD/gc_${tag}.csv" 2>/dev/null)
  echo "${cond},${mm},${dg},${k:-NA},${n:-NA},${lat:-NA},${tlt:-NA},${f:-NA}" >> "$CSV"
  echo "[gentlecorr] ${tag}: ${kn:-NA} lat=${lat:-NA} tilt=${tlt:-NA} fins=${f:-NA}N" | tee -a "$M"
}

run baseline    6 6 ""
run tipshift    6 6 "VDASH_GATE_CORRECT=1"
run tipshift_g  6 6 "VDASH_GATE_CORRECT=1 $GENTLE"
run calgentle   6 6 "VDASH_TACTILE_CAL=1 VDASH_CAL_HALF_MM=8 VDASH_CAL_STEP_MM=4 VDASH_EPISODE_S=90"
run tipshift_g  6 4 "VDASH_GATE_CORRECT=1 $GENTLE"

echo "[gentlecorr] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
cat "$CSV" | tee -a "$M"
