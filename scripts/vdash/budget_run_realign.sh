#!/usr/bin/env bash
# In-hand RE-ALIGN test (the socket-edge / compliant-grip approach). Inject an in-gripper TILT (the
# topple condition) and soften the finger grip during insertion (VDASH_REALIGN + VDASH_GRIP_STIFF) so
# the peg can PIVOT in the gripper while the mouth guides it straight (wrist/press stay vertical, unlike
# de-cant). Question: does a compliant grip let a tilted peg re-align + insert? Baseline @8deg ~ 5%.
set -u
cd /workspaces/isaaclab_arena; unset DISPLAY
git config --global --add safe.directory /workspaces/isaaclab_arena 2>/dev/null || true
PY=/isaac-sim/python.sh; RD=results/vdash/repro; LD=logs/vdash/repro; mkdir -p "$RD" "$LD"
CSV="$RD/realign.csv"; echo "cond,off_deg,grip_stiff,k,n,realized_tilt,fins_med_N" > "$CSV"
M="$LD/realign.log"; echo "[realign] START $(date -u +%FT%TZ)" | tee "$M"

run() {  # cond off_deg extra
  local cond="$1" dg="$2" extra="$3"
  local tag="${cond}_d${dg}"; local out="$LD/ra_${tag}.out"
  env VDASH_GRASP_OFF_DEG="$dg" VDASH_GRASP_DIAG=1 $extra \
    "$PY" scripts/vdash/run_eval_grid.py --policies vdash_scripted --clearances 2.0 --levels L1 \
    --num_episodes 20 --num_envs 20 --seed 0 --log_dir "$LD" --no_resume --out "$RD/ra_${tag}.csv" > "$out" 2>&1
  local kn k n tlt f gs
  kn=$(grep -E 'vdash_scripted c=' "$out" | grep -oE '[0-9]+/[0-9]+' | head -1); k="${kn%/*}"; n="${kn#*/}"
  tlt=$(grep -m1 graspdiag "$out" | grep -oE 'tilt=[0-9.]+' | grep -oE '[0-9.]+')
  f=$(awk -F, 'NR==2{print $9}' "$RD/ra_${tag}.csv" 2>/dev/null)
  gs=$(echo "$extra" | grep -oE 'GRIP_STIFF=[0-9.]+' | grep -oE '[0-9.]+')
  # surface the [realign] stiffness confirmation / any failure
  grep -m1 -E "\[realign\]" "$out" | tee -a "$M" >/dev/null
  echo "${cond},${dg},${gs:-orig},${k:-NA},${n:-NA},${tlt:-NA},${f:-NA}" >> "$CSV"
  echo "[realign] ${tag} stiff=${gs:-orig}: ${kn:-NA} realized_tilt=${tlt:-NA} fins=${f:-NA}N  $(grep -m1 -oE '\[realign\] (compliant grip[^\"]*|set-stiffness failed[^\"]*)' "$out")" | tee -a "$M"
}

D=8
run baseline   $D ""
run realign800 $D "VDASH_REALIGN=1 VDASH_GRIP_STIFF=800"
run realign400 $D "VDASH_REALIGN=1 VDASH_GRIP_STIFF=400"
run realign200 $D "VDASH_REALIGN=1 VDASH_GRIP_STIFF=200"

echo "[realign] ALL_DONE $(date -u +%FT%TZ)" | tee -a "$M"
cat "$CSV" | tee -a "$M"
