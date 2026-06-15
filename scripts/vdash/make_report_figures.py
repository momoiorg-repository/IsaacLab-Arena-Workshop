# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Generate the figures embedded in docs/vdash_M1-M3_report_ja.md.

Pure matplotlib/trimesh (no Isaac Sim) so it runs fast and headless:
  * peg 3D render (from assets/vdash/peg.stl) — shows the cylindrical grip,
  * peg+socket cross-section schematic (from configs/vdash/assets.yaml) — bore, funnel, clearance c,
  * clearance-series bores (c = 2.0/1.0/0.5/0.25 mm),
  * M3 success-rate progression bar chart.

Run inside the container:
    /isaac-sim/python.sh scripts/vdash/make_report_figures.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import yaml
from matplotlib.patches import Polygon as MplPolygon

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ASSETS = os.path.join(_ROOT, "assets", "vdash")
_OUT = os.path.join(_ROOT, "docs", "images", "vdash")
os.makedirs(_OUT, exist_ok=True)

with open(os.path.join(_ROOT, "configs", "vdash", "assets.yaml")) as f:
    CFG = yaml.safe_load(f)


def fig_peg_3d():
    """3D render of the peg mesh (cylindrical grip + shaft + chamfered tip)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = trimesh.load(os.path.join(_ASSETS, "peg.stl"))
    v, faces = mesh.vertices * 1000.0, mesh.faces  # mm
    fig = plt.figure(figsize=(3.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    tris = Poly3DCollection(v[faces], alpha=0.9, facecolor="#5b8def", edgecolor="none")
    ax.add_collection3d(tris)
    for lo, hi, setlim in ((v[:, 0].min(), v[:, 0].max(), ax.set_xlim),
                           (v[:, 1].min(), v[:, 1].max(), ax.set_ylim),
                           (v[:, 2].min(), v[:, 2].max(), ax.set_zlim)):
        setlim(lo, hi)
    ax.set_box_aspect((v[:, 0].ptp(), v[:, 1].ptp(), v[:, 2].ptp()))
    ax.view_init(elev=12, azim=35)
    ax.set_title("peg (Ø8mm shaft, Ø20mm cyl grip)", fontsize=9)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "peg_3d.png"), dpi=130)
    plt.close(fig)


def _peg_profile(p):
    """Right-half (r>=0) outline of the peg in the x-z plane (mm)."""
    r = p["shaft_diameter"] / 2 * 1000
    ch = p["tip_chamfer"] * 1000
    L = p["shaft_length"] * 1000
    g = p["grip_size"] * 1000
    # tip(0) -> chamfer -> shaft -> grip cylinder
    return [(0, 0), (r - ch, 0), (r, ch), (r, L), (g / 2, L), (g / 2, L + g), (0, L + g)]


def _socket_profile(s, c_mm, peg_r_mm):
    """Right-half outline of the socket in x-z (mm), including bore + conical funnel."""
    outer = s["outer_xy"] / 2 * 1000
    h = s["height"] * 1000
    depth = s["bore_depth"] * 1000
    bore_r = peg_r_mm + c_mm
    csd = s.get("countersink_depth", s["mouth_chamfer"]) * 1000
    csr = s.get("countersink_radius", s["mouth_chamfer"]) * 1000
    # outer wall up, across the top to the funnel rim, down the funnel, down the bore, across the floor
    return [
        (0, 0), (outer, 0), (outer, h), (bore_r + csr, h),
        (bore_r, h - csd), (bore_r, h - depth), (0, h - depth),
    ]


def fig_cross_section():
    """Peg + socket cross-section schematic with the c=2.0 socket and a zoom on the funnel."""
    peg_r = CFG["peg"]["shaft_diameter"] / 2 * 1000
    pp = np.array(_peg_profile(CFG["peg"]))
    sp = np.array(_socket_profile(CFG["socket"], 2.0, peg_r))

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(8.2, 4.6), gridspec_kw={"width_ratios": [1.7, 1]})
    # --- left: full cross-section (mirror to both halves) ---
    for poly, color, lab in ((sp, "#9aa7b5", "socket (c=2.0mm)"), (pp, "#5b8def", "peg")):
        full = np.vstack([poly, poly[::-1] * [-1, 1]])
        ax.add_patch(MplPolygon(full, closed=True, facecolor=color, edgecolor="#33414f", lw=1.0,
                                label=lab, alpha=0.85))
    # show the peg dropped into the bore (tip at depth 20mm) as a dashed overlay
    pp_in = pp.copy(); pp_in[:, 1] += (30 - 20)  # mouth at z=30, place tip at z=10
    full_in = np.vstack([pp_in, pp_in[::-1] * [-1, 1]])
    ax.add_patch(MplPolygon(full_in, closed=True, facecolor="none", edgecolor="#d6336c", lw=1.3,
                            ls="--", label="peg inserted"))
    ax.set_xlim(-32, 32); ax.set_ylim(-2, 72); ax.set_aspect("equal")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title("peg / socket cross-section", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    # --- right: zoom on the mouth funnel ---
    for poly, color in ((sp, "#9aa7b5"), (pp, "#5b8def")):
        full = np.vstack([poly, poly[::-1] * [-1, 1]])
        axz.add_patch(MplPolygon(full, closed=True, facecolor=color, edgecolor="#33414f", lw=1.0, alpha=0.85))
    axz.annotate("3mm funnel\n(lead-in)", xy=(peg_r + 1.5, 28.5), xytext=(9, 33),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="#d6336c"))
    axz.annotate(f"clearance c\n(bore r = {peg_r:.0f}+c)", xy=(peg_r + 2, 18), xytext=(9, 12),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="#0b7285"))
    axz.set_xlim(0, 16); axz.set_ylim(8, 34); axz.set_aspect("equal")
    axz.set_xlabel("x [mm]"); axz.set_title("mouth zoom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "cross_section.png"), dpi=130)
    plt.close(fig)


def fig_clearance_series():
    """The four bores (c = 2.0/1.0/0.5/0.25 mm) drawn to scale at the mouth."""
    peg_r = CFG["peg"]["shaft_diameter"] / 2 * 1000
    cs = CFG["clearances_mm"]
    fig, axes = plt.subplots(1, len(cs), figsize=(9.0, 2.6))
    for ax, c in zip(axes, cs):
        bore_r = peg_r + c
        ax.add_patch(plt.Circle((0, 0), bore_r, facecolor="#dfe6ee", edgecolor="#33414f"))
        ax.add_patch(plt.Circle((0, 0), peg_r, facecolor="#5b8def", edgecolor="#33414f"))
        ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"c={c}mm\n(radial gap {c}mm)", fontsize=9)
    fig.suptitle("clearance series — peg (blue) in bore (grey), to scale", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "clearance_series.png"), dpi=130)
    plt.close(fig)


def fig_success_progression():
    """M3 success-rate progression on L1 / c=2.0 (50 trials)."""
    labels = ["cyl grip\n+ basic", "+ light press\nspiral/retry", "+ two-stage", "+ tilt-retry", "+ funnel"]
    vals = [62, 74, 70, 68, 68]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(range(len(vals)), vals, color="#5b8def", edgecolor="#33414f")
    ax.axhline(80, color="#d6336c", ls="--", lw=1.2)
    ax.text(len(vals) - 0.5, 81, "target 80%", color="#d6336c", fontsize=8, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("success rate [%]"); ax.set_ylim(0, 100)
    ax.set_title("M3 insertion success — L1, c=2.0mm, 50 trials", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "m3_success_progression.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_peg_3d()
    fig_cross_section()
    fig_clearance_series()
    fig_success_progression()
    print("FIGURES_OK ->", _OUT)
