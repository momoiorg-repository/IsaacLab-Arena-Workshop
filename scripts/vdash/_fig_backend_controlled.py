# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib

matplotlib.use("Agg")
import math
import matplotlib.pyplot as plt
import numpy as np


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, p - (c - h), (c + h) - p  # rate, err_low, err_high


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- Panel A: scripted test-bed (rigid grasp, injected error) — mechanism works ---
labels = ["clean\ngrasp", "tilt 8°\n(pure)", "combined\nlat6+tilt6"]
base = [100, 5, 5]
corr = [100, 50, 20]
x = np.arange(len(labels))
w = 0.36
axL.bar(x - w / 2, base, w, label="back-end only", color="#c44")
axL.bar(x + w / 2, corr, w, label="+ re-align + RCC", color="#3a7")
for xi, (b, c) in enumerate(zip(base, corr)):
    axL.text(xi - w / 2, b + 2, f"{b}%", ha="center", fontsize=9)
    axL.text(xi + w / 2, c + 2, f"{c}%", ha="center", fontsize=9)
axL.set_xticks(x)
axL.set_xticklabels(labels)
axL.set_ylabel("insertion success (%)")
axL.set_ylim(0, 110)
axL.set_title(
    "(A) Scripted test-bed (injected grasp error)\nre-align (tilt) + RCC (lateral) recover the moderate band",
    fontsize=10,
)
axL.legend(fontsize=9, loc="center right")
axL.grid(axis="y", alpha=0.3)

# --- Panel B: real VLA, controlled n=40 — no transfer ---
arms = ["baseline\n(gates OFF)", "combo\n(re-align+RCC)"]
rate = [35.0, 37.5]
# wilson err from 14/40, 15/40
r0, lo0, hi0 = wilson(14, 40)
r1, lo1, hi1 = wilson(15, 40)
rates = [100 * r0, 100 * r1]
errlo = [100 * lo0, 100 * lo1]
errhi = [100 * hi0, 100 * hi1]
xb = np.arange(2)
axR.bar(xb, rates, 0.5, color=["#c44", "#3a7"], yerr=[errlo, errhi], capsize=6, ecolor="#333")
for xi, rt, k in zip(xb, rates, [14, 15]):
    axR.text(xi, rt + (100 * hi0 if xi == 0 else 100 * hi1) + 2, f"{k}/40\n{rt:.1f}%", ha="center", fontsize=9)
axR.set_xticks(xb)
axR.set_xticklabels(arms)
axR.set_ylabel("insertion success (%)")
axR.set_ylim(0, 75)
axR.set_title(
    "(B) Real VLA (GR00T N1.7), matched control, n=40/arm\n+2.5pp, p=0.82 — NO significant effect", fontsize=10
)
axR.grid(axis="y", alpha=0.3)
axR.annotate(
    "outlier grasps (lat 14–32 mm,\ntilt up to 70°) are out-of-budget\n→ gentle correction can't rescue them",
    xy=(0.5, 0.5),
    xycoords="axes fraction",
    ha="center",
    va="center",
    fontsize=8.5,
    color="#555",
    bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#bbb"),
)

fig.suptitle(
    "Back-end grasp correction: works in isolation, does not transfer to the real-VLA aggregate\n"
    "→ controlled confirmation of the precision-budget thesis (front-end grasp quality is the lever)",
    fontsize=10.5,
    y=1.02,
)
fig.tight_layout()
out = "figs/vdash/backend_correction_controlled.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
