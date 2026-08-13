#!/usr/bin/env bash
# Record the FULL pick->insert v5 dataset (--until inserted) as the END-TO-END A/B comparison
# against the handoff-cut training set (vla_pick_handoff_v5_recovery). Same composition
# (500 clean / 250 wide / 250 recovery, same seeds/stds/perturb) and the same camera render fix
# baked into record_vla_demos.py — the ONLY difference vs the handoff set is the cut point.
#
# Launch (gn17 container, detached so it survives the session):
#   docker exec -u an isaaclab_arena-cuda_gr00t_gn17 bash -lc \
#     'cd /workspaces/isaaclab_arena && nohup bash scripts/bdash/record_v5_full.sh \
#        > logs/bdash/record_v5_full.out 2>&1 &'
set -euo pipefail
cd /workspaces/isaaclab_arena
unset DISPLAY   # no-op in this container (DISPLAY already empty) but kept for parity

PY=/isaac-sim/python.sh
PARTS=datasets/bdash/_v5_full_parts
OUT=datasets/bdash/vla_pick_insert_v5_full.hdf5
LANG="Pick up the peg and insert it into the socket."
ENVARGS="bdash_pick_insert --clearance 2.0 --level L1"

rec() {  # rec <seed> <num_demos> <arm_init_std> <out.hdf5> [extra flags...]
  local seed="$1" n="$2" std="$3" out="$4"; shift 4
  $PY scripts/bdash/record_vla_demos.py --enable_cameras --num_envs 1 \
    --until inserted --language "$LANG" \
    --seed "$seed" --num_demos "$n" --arm_init_std "$std" --dataset_file "$out" "$@" \
    $ENVARGS
}

echo "=== removing old full artifacts ==="
rm -f "$PARTS"/*.hdf5 "$OUT" 2>/dev/null || true
mkdir -p "$PARTS"

echo "=== SLICE 1/3 clean (500, std 0.02, seed 0) ==="
rec 0 500 0.02 "$PARTS/full_clean.hdf5"

echo "=== SLICE 2/3 wide-start (250, std 0.15, seed 1) ==="
rec 1 250 0.15 "$PARTS/full_wide.hdf5"

echo "=== SLICE 3/3 recovery (250, std 0.05, perturb 1.0, seed 2) ==="
rec 2 250 0.05 "$PARTS/full_recovery.hdf5" --perturb_frac 1.0

echo "=== MERGE ==="
$PY scripts/bdash/merge_hdf5_demos.py \
  --inputs "$PARTS/full_clean.hdf5" "$PARTS/full_wide.hdf5" "$PARTS/full_recovery.hdf5" \
  --output "$OUT"

echo "ALL_DONE -> $OUT"
