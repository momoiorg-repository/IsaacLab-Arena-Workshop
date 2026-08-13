#!/usr/bin/env bash
# Calibrate the per-finger force z-asymmetry (dz) vs in-gripper tilt, to invert dz -> tilt for the
# §2.1 de-cant. Scripted test-bed, BDASH_GRASP_OFF_DEG sweep, finger + grasp diag.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
PY=/isaac-sim/python.sh; LD=logs/bdash/repro; RD=results/bdash/repro; mkdir -p "$LD" "$RD"
M="$LD/finger_calib.log"; echo "[fcal] START $(date -u +%FT%TZ)" | tee "$M"
for d in 0 2 4 6 8 12 16 20; do
  out="$LD/fcal_d${d}.out"
  env BDASH_GRASP_OFF_MM=0 BDASH_GRASP_OFF_DEG="$d" BDASH_GRASP_DIAG=1 BDASH_FINGER_DIAG=1 \
    "$PY" scripts/bdash/run_eval_grid.py --policies bdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 3 --num_envs 3 --seed 0 --log_dir "$LD" --no_resume --out "$RD/fcal_d${d}.csv" > "$out" 2>&1
  echo "[fcal] OFF_DEG=$d" | tee -a "$M"
  grep fingerdiag "$out" | sed 's/\r//g' | tee -a "$M"
done
echo "[fcal] DONE $(date -u +%FT%TZ)" | tee -a "$M"
