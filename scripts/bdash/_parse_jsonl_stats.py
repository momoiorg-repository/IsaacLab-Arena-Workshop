# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os
import sys


def stats(path):
    succ = tot = topple = 0
    tilts = []
    lats = []
    for L in open(path):
        L = L.strip()
        if not L:
            continue
        try:
            d = json.loads(L)
        except:
            continue
        if "result" not in d:
            continue
        tot += 1
        if d.get("result") == "success":
            succ += 1
        ft = d.get("final_tilt_deg", 0) or 0
        if ft > 45:
            topple += 1
    return succ, tot, topple


for pat in sys.argv[1:]:
    hits = glob.glob(f"logs/bdash/**/{pat}", recursive=True)
    if not hits:
        print(f"{pat}: NOT FOUND")
        continue
    p = hits[0]
    s, t, tp = stats(p)
    print(f"{pat}: success {s}/{t} = {100*s/max(t,1):.0f}%   topple {tp}/{t} = {100*tp/max(t,1):.0f}%")
