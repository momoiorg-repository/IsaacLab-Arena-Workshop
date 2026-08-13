#!/usr/bin/env bash
# Tactile-probe validation (the make-or-break). PART 1: estimator accuracy — does the bore-edge probe
# recover the in-gripper lateral error e_xy? (compares BDASH_CAL_DIAG estimate to BDASH_GRASP_DIAG GT;
# the estimate is logged at SOLVE so it does not need the insert to finish). PART 2: closed-loop success
# with the probe (cal) vs blind vs oracle, at real magnitudes.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
PY=/isaac-sim/python.sh; RD=results/bdash/repro; LD=logs/bdash/repro; mkdir -p "$RD" "$LD"
M="$LD/probe.log"; echo "[probe] START $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD)" | tee "$M"

acc() {  # off half step
  local off="$1" half="$2" step="$3"
  local out="$LD/probe_acc_off${off}_g${step}.out"
  env BDASH_GRASP_OFF_MM="$off" BDASH_TACTILE_CAL=1 BDASH_CAL_DIAG=1 BDASH_GRASP_DIAG=1 \
      BDASH_CAL_HALF_MM="$half" BDASH_CAL_STEP_MM="$step" BDASH_EPISODE_S=60 \
      "$PY" scripts/bdash/run_eval_grid.py --policies bdash_scripted --clearances 2.0 --levels L1 \
      --num_episodes 4 --num_envs 4 --seed 0 --log_dir "$LD" --no_resume --out "$RD/probe_acc_off${off}_g${step}.csv" > "$out" 2>&1
  local est gt; est=$(grep -m1 caldiag "$out" | grep -oE 'e_xy=\([-0-9.]+,[-0-9.]+\)mm')
  gt=$(grep -m1 graspdiag "$out" | grep -oE 'lat=[0-9.]+mm')
  echo "[probe] acc off=${off} grid=${half}/${step}: cal=${est:-NONE} gt=${gt:-NONE}" | tee -a "$M"
}
echo "[probe] PART1 estimator accuracy (5x5 grid)" | tee -a "$M"
for o in 4 6 8 10 12; do acc "$o" 12 6; done
echo "[probe] PART1b estimator accuracy (3x3 grid, fast)" | tee -a "$M"
for o in 4 8 12; do acc "$o" 9 9; done

cl() {  # off mode(cal|blind|oracle)
  local off="$1" mode="$2" extra=""
  local out="$LD/probe_cl_${mode}_off${off}.out"
  [ "$mode" = cal ] && extra="BDASH_TACTILE_CAL=1 BDASH_EPISODE_S=60"
  [ "$mode" = oracle ] && extra="BDASH_CHEAT_TIP=1"
  env BDASH_GRASP_OFF_MM="$off" $extra \
      "$PY" scripts/bdash/run_eval_grid.py --policies bdash_scripted --clearances 2.0 --levels L1 \
      --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/probe_cl_${mode}_off${off}.csv" > "$out" 2>&1
  local kn; kn=$(grep -E 'bdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1)
  echo "[probe] cl off=${off} ${mode}: ${kn:-NA}" | tee -a "$M"
}
echo "[probe] PART2 closed-loop (blind vs oracle vs cal)" | tee -a "$M"
for o in 6 8 10; do cl "$o" blind; cl "$o" oracle; cl "$o" cal; done

echo "[probe] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
