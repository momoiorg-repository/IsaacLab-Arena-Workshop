#!/bin/bash
# P0-3: reduced R(g) sweep -- axial grip error x approach tilt, per variant, upright, fast speeds.
#
# Grid, declared here and nowhere else:
#   axial [mm]: 0,2,5,8,12  x4 reps  (the gate's axis: grip error passes 1:1 into protrusion)
#   tilt  [deg]: 2,4,6      x3 reps  (axial=0; the second axis of the ruling, reduced)
# 29 episodes/variant, 87 total. The zero cell doubles as the fast-speed quality baseline.
# Output: logs/bdash/rg_<variant>.jsonl, fingerprint-stamped (§9).
set -e
cd /home/an/Workspace/bdash/code
AX="0,2,5,8,12,0,2,5,8,12,0,2,5,8,12,0,2,5,8,12"
TI="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
AX2="0,0,0,0,0,0,0,0,0"
TI2="2,4,6,2,4,6,2,4,6"
for V in W-A W-B W-C; do
  for PASS in 1 2; do
    if [ "$PASS" = 1 ]; then E="$AX"; T="$TI"; N=20; TAG=ax; else E="$AX2"; T="$TI2"; N=9; TAG=ti; fi
    echo "=== R(g) $V pass=$TAG n=$N ==="
    docker exec -e PYTHONUNBUFFERED=1 -e BDASH_E1_ERRORS="$E" -e BDASH_E1_TILTS="$T" \
      isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && unset DISPLAY && \
      timeout 5400 /isaac-sim/python.sh scripts/bdash/check_chuck_teacher.py --num_envs 1 --seed 90 \
      --episodes $N --max_steps 1600 --headless --enable_cameras \
      --jsonl logs/bdash/rg_${V}_${TAG}.jsonl bdash_chuck_load --task full --variants $V" \
      > /tmp/rg_${V}_${TAG}.log 2>&1 || echo "PASS FAILED: $V $TAG (see /tmp/rg_${V}_${TAG}.log)"
  done
done
echo "RG_SWEEP_DONE"
