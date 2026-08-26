#!/bin/bash
# P0-4: E1 -- S2 x {P1,P2,P3}, identical seed and identical injected-error sequence per policy.
#
# The injected distribution is DECLARED (the measured pilot was discarded with its dataset):
# 60% small (0-1 mm), 25% medium (2-6), 15% large (8-12), fixed shuffle, 50 values -- the same
# sequence hits every policy, so the comparison is between policies and never between draws.
set -e
cd /home/an/Workspace/bdash/code
ERRS=$(python3 - <<'PY'
import random
r=random.Random(2026)
vals=[round(r.uniform(0,1),1) for _ in range(30)]+[round(r.uniform(2,6),1) for _ in range(13)]+[round(r.uniform(8,12),1) for _ in range(7)]
r.shuffle(vals); print(",".join(str(v) for v in vals))
PY
)
echo "E1 error sequence: $ERRS"
for MODE in P1 P2 P3; do
  echo "=== E1 $MODE n=51 ==="
  docker exec -e PYTHONUNBUFFERED=1 -e BDASH_POLICY_MODE=$MODE -e BDASH_E1_ERRORS="$ERRS" \
    isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && unset DISPLAY && \
    timeout 14400 /isaac-sim/python.sh scripts/bdash/check_chuck_teacher.py --num_envs 1 --seed 90 \
    --episodes 51 --max_steps 2400 --headless --enable_cameras \
    --jsonl logs/bdash/e1_${MODE}.jsonl bdash_chuck_load --task full --variants all" \
    > /tmp/e1_${MODE}.log 2>&1 || echo "E1 $MODE FAILED (see /tmp/e1_${MODE}.log)"
done
echo "E1_DONE"
