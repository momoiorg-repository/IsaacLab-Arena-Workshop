#!/usr/bin/env bash
# Validate the §2.1 EE de-cant (tilt fix) on the scripted tilt test-bed (VDASH_GRASP_OFF_DEG induces
# in-gripper tilt+lateral). Compares blind / oracle(tip-only) / decant(tilt-only) / full(decant+lateral
# probe). GAIN (finger dz N per deg) passed via env. KEY: at high tilt the oracle (tip position only)
# cannot fix the cocked peg, but full (de-cant + lateral) should.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
PY=/isaac-sim/python.sh; LD=logs/vdash/repro; RD=results/vdash/repro; mkdir -p "$LD" "$RD"
GAIN="${GAIN:-0.5}"; SIGN="${SIGN:--1}"
M="$LD/decant.log"; echo "[decant] START $(date -u +%FT%TZ) gain=$GAIN sign=$SIGN" | tee "$M"
cl() {  # off_deg mode
  local d="$1" mode="$2" extra=""
  local out="$LD/dc_${mode}_d${d}.out"
  [ "$mode" = decant ] && extra="VDASH_DECANT=1 VDASH_DECANT_GAIN=$GAIN VDASH_DECANT_SIGN=$SIGN VDASH_CAL_DIAG=1 VDASH_EPISODE_S=40"
  [ "$mode" = full ]   && extra="VDASH_DECANT=1 VDASH_DECANT_GAIN=$GAIN VDASH_DECANT_SIGN=$SIGN VDASH_TACTILE_CAL=1 VDASH_EPISODE_S=80"
  [ "$mode" = oracle ] && extra="VDASH_CHEAT_TIP=1"
  env VDASH_GRASP_OFF_DEG="$d" VDASH_GRASP_DIAG=1 $extra \
    "$PY" scripts/vdash/run_eval_grid.py --policies vdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/dc_${mode}_d${d}.csv" > "$out" 2>&1
  local kn; kn=$(grep -E 'vdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1)
  echo "[decant] off_deg=$d $mode: ${kn:-NA}" | tee -a "$M"
}
for d in 6 10 15; do cl "$d" blind; cl "$d" oracle; cl "$d" decant; cl "$d" full; done
echo "[decant] DONE $(date -u +%FT%TZ)" | tee -a "$M"
