# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
import sys


# Extract the FIRST grasp-diag (pick) tilt/lat per episode from a run .out
def picks(path):
    seen = {}
    for L in open(path, errors="ignore"):
        m = re.search(r"graspdiag.*ep=(\d+).*ingrip_tilt=([\d.]+)deg lat=([\d.]+)mm", L)
        if not m:
            continue
        ep = int(m.group(1))
        if ep not in seen:  # first (pick) grasp for this ep
            seen[ep] = (float(m.group(2)), float(m.group(3)))
    return seen


b = picks(sys.argv[1])
c = picks(sys.argv[2])
eps = sorted(set(b) | set(c))
print(f"{'ep':>3} {'base tilt/lat':>16} {'combo tilt/lat':>16}  {'match?':>6}")
nmatch = 0
ntot = 0
for e in eps:
    bb = b.get(e)
    cc = c.get(e)
    bs = f"{bb[0]:.1f}/{bb[1]:.1f}" if bb else "-"
    cs = f"{cc[0]:.1f}/{cc[1]:.1f}" if cc else "-"
    mk = ""
    if bb and cc:
        ntot += 1
        same = abs(bb[0] - cc[0]) < 0.5 and abs(bb[1] - cc[1]) < 0.5
        mk = "SAME" if same else "diff"
        if same:
            nmatch += 1
    print(f"{e:>3} {bs:>16} {cs:>16}  {mk:>6}")
print(
    f"\n{nmatch}/{ntot} episodes have matching pick grasps ->"
    f" {'DETERMINISTIC (paired)' if ntot and nmatch>=0.8*ntot else 'NON-deterministic (independent samples)'}"
)
