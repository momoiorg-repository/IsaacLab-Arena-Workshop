#!/usr/bin/env bash
# Tip-over reduction (autonomous). Dominant real-VLA failure = tilt-induced topple (35% even no-probe).
# RCC compliance (rcc_xy/rot_gain) is OFF in controllers.yaml. Sweep insertion params on the scripted
# in-gripper TILT injection (VDASH_GRASP_OFF_DEG, the topple condition) to find settings that let a
# tilted peg seat instead of tip. Baseline @8deg ~ 5% (blind) / 30% (tip-shift). Beat it via mechanics.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
PY=/isaac-sim/python.sh; RD=results/vdash/repro; LD=logs/vdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/instune.csv"; echo "cond,off_deg,k,n,realized_tilt_deg,fins_med_N" > "$CSV"
M="$LD/instune.log"; echo "[instune] START $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD)" | tee "$M"

run() {  # cond off_deg extra_env
  local cond="$1" dg="$2" extra="$3"
  local tag="${cond}_d${dg}"; local out="$LD/it_${tag}.out"
  env VDASH_GRASP_OFF_DEG="$dg" VDASH_GRASP_DIAG=1 $extra \
    "$PY" scripts/vdash/run_eval_grid.py --policies vdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/it_${tag}.csv" > "$out" 2>&1
  local kn k n tlt f
  kn=$(grep -E 'vdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1); k="${kn%/*}"; n="${kn#*/}"
  tlt=$(grep -m1 graspdiag "$out" | grep -oE 'tilt=[0-9.]+' | grep -oE '[0-9.]+')
  f=$(awk -F, 'NR==2{print $9}' "$RD/it_${tag}.csv" 2>/dev/null)
  echo "${cond},${dg},${k:-NA},${n:-NA},${tlt:-NA},${f:-NA}" >> "$CSV"
  echo "[instune] ${tag}: ${kn:-NA} realized_tilt=${tlt:-NA} fins=${f:-NA}N" | tee -a "$M"
}

D=8  # the topple point (blind 5%, tip-shift 30%)
run baseline  $D ""
run rccrot03  $D "VDASH_INS_RCC_ROT_GAIN=0.03"
run rccrot08  $D "VDASH_INS_RCC_ROT_GAIN=0.08"
run rccxy     $D "VDASH_INS_RCC_XY_GAIN=0.0008"
run rccxyrot  $D "VDASH_INS_RCC_XY_GAIN=0.0008 VDASH_INS_RCC_ROT_GAIN=0.05"
run gentle    $D "VDASH_INS_F_TARGET=2.0 VDASH_INS_PRESS_FORCE_BAND=1.5 VDASH_INS_CONTACT_SPEED=0.004"

echo "[instune] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
cat "$CSV" | tee -a "$M"
