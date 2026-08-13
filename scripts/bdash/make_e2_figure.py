# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""E2 figure — rule-based clearance success curve (brief §3 M5).

Reads the eval-grid CSV produced by ``scripts/bdash/run_eval_grid.py`` and plots success rate vs
radial clearance ``c``, one series per scene-diversity level, with 95% Wilson-CI error bars. This is
one wing of the paper's main figure; while only c=2.0 is tuned, the curve is anchored there and fills
in toward tighter clearances as M5 progresses.

Pure matplotlib (no Isaac Sim) so it runs fast and headless:

    /isaac-sim/python.sh scripts/bdash/make_e2_figure.py \\
        --csv results/bdash/e2_clearance.csv --out docs/images/bdash/e2_clearance_curve.png
"""

from __future__ import annotations

import argparse
import csv
import matplotlib
import os
from collections import defaultdict

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LEVEL_LABEL = {
    "L0": "L0 (canonical)",
    "L1": "L1 (pose/yaw)",
    "L2": "L2 (+lighting)",
    "L3": "L3 (+texture/distractors)",
}
_LEVEL_COLOR = {"L0": "#0b7285", "L1": "#5b8def", "L2": "#e8590c", "L3": "#d6336c"}


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_rows(csv_path: str, policy: str) -> list[dict]:
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r.get("policy") != policy:
                continue
            n = _f(r.get("n"), 0) or 0
            if n <= 0:
                continue
            rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser(description="E2 clearance success curve (M5)")
    p.add_argument("--csv", default=os.path.join(_REPO_ROOT, "results", "bdash", "e2_clearance.csv"))
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "docs", "images", "bdash", "e2_clearance_curve.png"))
    p.add_argument("--policy", default="bdash_scripted")
    args = p.parse_args()

    rows = load_rows(args.csv, args.policy)
    if not rows:
        raise SystemExit(f"[e2] no rows for policy={args.policy} in {args.csv}")

    # group by level -> sorted-by-clearance points
    by_level: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for level in sorted(by_level):
        pts = sorted(by_level[level], key=lambda r: _f(r["clearance_mm"]))
        cs = [_f(r["clearance_mm"]) for r in pts]
        sr = [100.0 * _f(r["success_rate"], 0.0) for r in pts]
        lo = [100.0 * (_f(r["success_rate"], 0.0) - _f(r["ci95_low"], 0.0)) for r in pts]
        hi = [100.0 * (_f(r["ci95_high"], 0.0) - _f(r["success_rate"], 0.0)) for r in pts]
        ax.errorbar(
            cs,
            sr,
            yerr=[lo, hi],
            marker="o",
            capsize=4,
            lw=1.8,
            ms=6,
            color=_LEVEL_COLOR.get(level, None),
            label=_LEVEL_LABEL.get(level, level),
        )
        # annotate n at each point
        for r, c, s in zip(pts, cs, sr):
            ax.annotate(
                f"n={int(_f(r['n']))}", (c, s), textcoords="offset points", xytext=(6, -12), fontsize=7, color="#555"
            )

    ax.set_xlabel("radial clearance c [mm]")
    ax.set_ylabel("insertion success rate [%]")
    ax.set_ylim(-3, 103)
    ax.set_xticks([0.25, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.25", "0.5", "1.0", "2.0"])
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_title(f"E2 — rule-based insertion success vs clearance ({args.policy})", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, title="scene diversity")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"[e2] wrote {args.out}  ({len(rows)} cells, levels={sorted(by_level)})")
    print("E2_FIGURE_OK")


if __name__ == "__main__":
    main()
