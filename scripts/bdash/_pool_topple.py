# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import csv
import glob
import json
from collections import defaultdict

rows = list(csv.DictReader(open("results/bdash/repro/vla_pool.csv")))
agg = defaultdict(lambda: dict(n=0, succ=0, topple=0))
for r in rows:
    jname = r["jsonl"]
    hits = glob.glob("logs/bdash/**/" + jname, recursive=True)
    if not hits:
        print("  [miss]", r["arm"], "s" + r["seed"], jname)
        continue
    A = agg[r["arm"]]
    for L in open(hits[0]):
        L = L.strip()
        if not L:
            continue
        try:
            d = json.loads(L)
        except:
            continue
        if "result" not in d:
            continue
        A["n"] += 1
        if d.get("result") == "success":
            A["succ"] += 1
        ft = d.get("final_tilt_deg", 0) or 0
        if ft > 45:
            A["topple"] += 1
print("%9s %14s %14s" % ("arm", "success", "topple(>45deg)"))
for arm in ("baseline", "combo"):
    A = agg[arm]
    n = A["n"] or 1
    print(
        "%9s   %2d/%-3d = %3.0f%%   %2d/%-3d = %3.0f%%"
        % (arm, A["succ"], A["n"], 100 * A["succ"] / n, A["topple"], A["n"], 100 * A["topple"] / n)
    )
