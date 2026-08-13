# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""E4 figure — convergence zone + handoff-error overlay (brief §3 M7).

Two panels from the convergence-zone CSV (``run_convergence_zone.py``) and the M5 handoff JSONL:
  * **left**: success-rate heatmap over (offset r, tilt θ) — the insertion controller's capture
    region (averaged over azimuth);
  * **right**: the same zone projected to the (dx,dy) handoff plane (disk of the lateral capture
    radius) with the *measured* handoff-error scatter overlaid, plus P(handoff error ∈ zone) vs the
    measured insertion success — the paper's validation that the pick-side terminal-error
    distribution lands inside the rule-based convergence zone.

Pure matplotlib (no Isaac Sim):

    /isaac-sim/python.sh scripts/bdash/make_e4_figure.py \\
        --conv_csv results/bdash/e4_convergence_c2.csv \\
        --out docs/images/bdash/e4_convergence_c2.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import matplotlib
import os
from collections import defaultdict

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_conv(csv_path: str):
    rows = list(csv.DictReader(open(csv_path)))
    clearance = float(rows[0]["clearance_mm"]) if rows else 0.0
    rs = sorted({float(r["r_mm"]) for r in rows})
    ths = sorted({float(r["tilt_deg"]) for r in rows})
    # success rate per (r, θ) averaged over azimuth
    bucket = defaultdict(list)
    for r in rows:
        bucket[(float(r["r_mm"]), float(r["tilt_deg"]))].append(int(r["success"]))
    grid = np.full((len(ths), len(rs)), np.nan)
    for i, th in enumerate(ths):
        for j, rr in enumerate(rs):
            v = bucket.get((rr, th))
            if v:
                grid[i, j] = sum(v) / len(v)
    return clearance, np.array(rs), np.array(ths), grid


def lateral_capture_radius(rs, ths, grid):
    """Largest offset r captured (>=50%) at θ≈0 — the lateral convergence radius."""
    i0 = int(np.argmin(np.abs(ths)))
    row = grid[i0]
    captured = rs[np.where(row >= 0.5)[0]]
    return float(captured.max()) if captured.size else 0.0


def load_handoff(jsonl_glob: str):
    files = sorted(glob.glob(jsonl_glob), key=os.path.getmtime)
    if not files:
        return None
    dx, dy = [], []
    for line in open(files[-1]):
        d = json.loads(line)
        if d.get("handoff_dx_mm") is not None and d.get("t_handoff_s") is not None:
            dx.append(d["handoff_dx_mm"])
            dy.append(d["handoff_dy_mm"])
    return (np.array(dx), np.array(dy), os.path.basename(files[-1])) if dx else None


def main():
    p = argparse.ArgumentParser(description="E4 convergence zone + handoff overlay (M7)")
    p.add_argument("--conv_csv", default=os.path.join(_REPO_ROOT, "results", "bdash", "e4_convergence_c2.csv"))
    p.add_argument(
        "--handoff_jsonl",
        default=os.path.join(_REPO_ROOT, "logs", "bdash", "bdash_handoff_grid_bdash_scripted_c2.0_L1_*.jsonl"),
    )
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "docs", "images", "bdash", "e4_convergence_c2.png"))
    args = p.parse_args()

    clearance, rs, ths, grid = load_conv(args.conv_csv)
    r_cap = lateral_capture_radius(rs, ths, grid)
    handoff = load_handoff(args.handoff_jsonl)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # --- left: (r, θ) capture heatmap ---
    im = axL.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        extent=[rs.min() - 0.5, rs.max() + 0.5, ths.min() - 1, ths.max() + 1],
    )
    axL.set_xlabel("lateral offset r [mm]")
    axL.set_ylabel("tilt θ [deg]")
    axL.set_title(f"convergence zone — capture rate (c={clearance:.2f} mm)", fontsize=10)
    axL.set_xticks(rs)
    axL.set_yticks(ths)
    for i, th in enumerate(ths):
        for j, rr in enumerate(rs):
            if not np.isnan(grid[i, j]):
                axL.text(
                    rr,
                    th,
                    f"{grid[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#222" if grid[i, j] > 0.4 else "#fff",
                )
    fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04, label="capture rate")

    # --- right: (dx,dy) zone disk + handoff scatter ---
    axR.add_patch(
        Circle(
            (0, 0),
            r_cap,
            facecolor="#b2f2bb",
            edgecolor="#2f9e44",
            lw=1.5,
            alpha=0.6,
            label=f"convergence zone (r≤{r_cap:.0f} mm, θ≈0)",
        )
    )
    axR.plot(0, 0, "+", color="#2f9e44", ms=10)
    lim = max(r_cap + 2, 6)
    p_in = None
    if handoff is not None:
        dx, dy, fname = handoff
        rr = np.hypot(dx, dy)
        inside = rr <= r_cap
        p_in = float(inside.mean())
        axR.scatter(dx[inside], dy[inside], s=22, color="#1971c2", label=f"handoff ∈ zone ({inside.sum()})")
        axR.scatter(
            dx[~inside], dy[~inside], s=22, color="#e8590c", marker="x", label=f"handoff ∉ zone ({(~inside).sum()})"
        )
        lim = max(lim, float(np.abs(np.concatenate([dx, dy])).max()) + 1)
    axR.set_xlim(-lim, lim)
    axR.set_ylim(-lim, lim)
    axR.set_aspect("equal")
    axR.axhline(0, color="#ccc", lw=0.6)
    axR.axvline(0, color="#ccc", lw=0.6)
    axR.set_xlabel("handoff dx [mm]")
    axR.set_ylabel("handoff dy [mm]")
    title = "handoff terminal error vs convergence zone"
    if p_in is not None:
        title += f"\nP(handoff ∈ zone) = {p_in:.0%}"
    axR.set_title(title, fontsize=10)
    axR.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"[e4] lateral capture radius (θ≈0): {r_cap:.1f} mm")
    if p_in is not None:
        print(f"[e4] P(handoff ∈ zone) = {p_in:.1%}  ({handoff[2]})")
    print(f"[e4] wrote {args.out}")
    print("E4_FIGURE_OK")


if __name__ == "__main__":
    main()
