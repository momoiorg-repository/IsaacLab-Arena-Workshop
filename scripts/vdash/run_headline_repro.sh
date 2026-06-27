#!/usr/bin/env bash
# V-DASH headline re-run from the frozen tree (commits be154b29 + 66775820).
# Reproduces: E2 clearance grid (scripted), E4 convergence basins (5 seeds x
# {held,recover}), and the VLA v6-recovery closed-loop eval. Outputs land in
# results/vdash/repro/ so they can be diffed against results/vdash/prefreeze_backup/.
set -u
cd /workspaces/isaaclab_arena
unset DISPLAY
PY=/isaac-sim/python.sh
RDIR=results/vdash/repro
LDIR=logs/vdash/repro
mkdir -p "$RDIR" "$LDIR"
M="$LDIR/master.log"
echo "[repro] START $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD)" | tee "$M"

# ---- Phase 1: E2 clearance grid (scripted, no cameras) ----
echo "[repro] PHASE1 E2 clearance grid $(date -u +%T)Z" | tee -a "$M"
$PY scripts/vdash/run_eval_grid.py --policies vdash_scripted \
  --clearances 2.0 1.0 0.5 0.25 --levels L0 L1 --num_episodes 50 --num_envs 50 --seed 0 \
  --log_dir logs/vdash --no_resume --out "$RDIR/e2_clearance_final.csv" > "$LDIR/e2_grid.out" 2>&1
echo "[repro] E2 exit=$?" | tee -a "$M"
grep -E "vdash_scripted c=" "$LDIR/e2_grid.out" | tee -a "$M"

# ---- Phase 2: E4 convergence basins (5 seeds x {recover,held}) ----
echo "[repro] PHASE2 E4 basins $(date -u +%T)Z" | tee -a "$M"
for MODE in recover held; do
  FLAG=""; [ "$MODE" = "recover" ] && FLAG="--recover"
  for S in 0 1 2 3 4; do
    $PY scripts/vdash/run_convergence_zone.py --policy_type vdash_scripted --num_envs 60 --seed "$S" $FLAG \
      --out "$RDIR/e4_${MODE}_c2.0_s${S}.csv" vdash_pick_insert --clearance 2.0 --level L0 \
      > "$LDIR/e4_${MODE}_s${S}.out" 2>&1
    echo "[repro] e4 ${MODE} s${S} exit=$?" | tee -a "$M"
  done
done

# ---- Phase 3: VLA v6-recovery closed-loop eval (cameras, num_envs forced to 1) ----
echo "[repro] PHASE3 VLA v6-recovery $(date -u +%T)Z" | tee -a "$M"
VDASH_VLA_DEBUG=1 $PY scripts/vdash/run_eval_grid.py --policies vdash_vla_v6_recovery \
  --clearances 2.0 --levels L1 --num_episodes 20 --seed 0 \
  --log_dir logs/vdash --no_resume --out "$RDIR/eval_v6_recovery.csv" > "$LDIR/eval_v6.out" 2>&1
echo "[repro] VLA exit=$?" | tee -a "$M"
grep -E "vdash_vla_v6_recovery c=" "$LDIR/eval_v6.out" | tee -a "$M"

echo "[repro] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
