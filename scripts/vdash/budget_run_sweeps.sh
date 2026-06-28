#!/usr/bin/env bash
# Theory sweeps (scripted test-bed, fast, no cameras). Emits a tidy CSV:
#  A) R_widened(σ): success vs estimator lateral noise at a fixed 8mm injected offset (sets the probe spec)
#  B) R(l) blind response: success vs injected in-gripper lateral offset (the blind-channel tolerance)
#  C) R_oracle(l): success vs offset with the true tip fed (the boundary-moved ceiling)
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
PY=/isaac-sim/python.sh
RD=results/vdash/repro; LD=logs/vdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/sweeps.csv"; echo "experiment,off_mm,noise_mm,cheat,k,n,realized_lat_mm" > "$CSV"
M="$LD/sweeps.log"; echo "[sweeps] START $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD)" | tee "$M"

run() {  # exp off noise cheat(1|"")
  local exp="$1" off="$2" noise="$3" cheat="$4"
  local tag="${exp}_off${off}_n${noise}_c${cheat:-0}"; local out="$LD/sw_${tag}.out"
  env VDASH_GRASP_OFF_MM="$off" VDASH_TIP_NOISE_MM="$noise" ${cheat:+VDASH_CHEAT_TIP=1} VDASH_GRASP_DIAG=1 \
    "$PY" scripts/vdash/run_eval_grid.py --policies vdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/sw_${tag}.csv" > "$out" 2>&1
  local kn=$(grep -E 'vdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1)
  local k="${kn%/*}"; local n="${kn#*/}"
  local lat=$(grep -m1 graspdiag "$out" | grep -oE 'lat=[0-9.]+mm' | grep -oE '[0-9.]+')
  echo "${exp},${off},${noise},${cheat:-0},${k:-NA},${n:-NA},${lat:-NA}" >> "$CSV"
  echo "[sweeps] ${tag}: ${k:-NA}/${n:-NA} realized_lat=${lat:-NA}mm" | tee -a "$M"
}

echo "[sweeps] A) R_widened(σ) @ off=8 ..." | tee -a "$M"
for s in 0 1 2 3 4 5 6 8; do run rwiden 8 "$s" 1; done
echo "[sweeps] B) R(l) blind response ..." | tee -a "$M"
for o in 0 2 3 4 5 6 8 10 12; do run rblind "$o" 0 ""; done
echo "[sweeps] C) R_oracle(l) ..." | tee -a "$M"
for o in 0 4 6 8 10 12; do run roracle "$o" 0 1; done

echo "[sweeps] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
column -t -s, "$CSV" | tee -a "$M"
