#!/usr/bin/env python3
"""Generate the V-DASH insertion-diagnosis figures (M11) from the CSVs / diagnostic logs on disk.

Run in the gn17 container:
    /isaac-sim/python.sh scripts/vdash/plot_paper_figs.py
Outputs -> figs/vdash/{e4_basin,ingripper,cheat,e2_clearance}.png
Each figure is skipped (with a note) if its source data is missing, so it is safe to re-run as the
E2 grid completes.
"""
from __future__ import annotations

import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results", "vdash")
LOGS = os.path.join(ROOT, "logs", "vdash")
FIGS = os.path.join(ROOT, "figs", "vdash")
os.makedirs(FIGS, exist_ok=True)


def _basin_rows(mode):
    """Prefer the multi-seed per-seed CSVs (e4_{mode}_c2.0_s*.csv); else the single 1-trial CSV."""
    seedfiles = sorted(glob.glob(os.path.join(RES, f"e4_{mode}_c2.0_s*.csv")))
    if seedfiles:
        rows = []
        for f in seedfiles:
            rows += list(csv.DictReader(open(f)))
        return rows, len(seedfiles)
    single = os.path.join(RES, "e4_recover_c2.0.csv" if mode == "recover" else "e4_convergence_c2.0.csv")
    return (list(csv.DictReader(open(single))) if os.path.exists(single) else []), 1


def _basin_grid_rows(rows):
    rs = sorted({float(r["r_mm"]) for r in rows})
    ts = sorted({float(r["tilt_deg"]) for r in rows})
    g = np.full((len(ts), len(rs)), np.nan)
    acc = {}
    for r in rows:
        k = (float(r["tilt_deg"]), float(r["r_mm"]))
        acc.setdefault(k, []).append(int(r["success"]))
    for i, t in enumerate(ts):
        for j, rr in enumerate(rs):
            if (t, rr) in acc:
                g[i, j] = 100.0 * np.mean(acc[(t, rr)])
    return rs, ts, g


def fig_e4_basin():
    held_rows, nh = _basin_rows("held")
    rec_rows, nr = _basin_rows("recover")
    if not (held_rows and rec_rows):
        print("skip e4_basin (missing basin CSVs)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, rows, ncell, title in ((axes[0], held_rows, nh, "Held (static tolerance)"),
                                   (axes[1], rec_rows, nr, "Recovery (faithful basin)")):
        rs, ts, g = _basin_grid_rows(rows)
        im = ax.imshow(g, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=100,
                       extent=[min(rs) - 0.5, max(rs) + 0.5, min(ts) - 1, max(ts) + 1])
        ax.set_xlabel("handoff lateral offset r (mm)")
        ax.set_ylabel("handoff tilt θ (deg)")
        ov = int(round(np.nanmean(g)))
        ax.set_title(f"{title}\noverall {ov}%")
        for i, t in enumerate(ts):
            for j, rr in enumerate(rs):
                if not np.isnan(g[i, j]):
                    ax.text(rr, t, f"{int(g[i, j])}", ha="center", va="center",
                            color="w" if g[i, j] < 55 else "k", fontsize=7)
    fig.colorbar(im, ax=axes, label="insert success (%)", shrink=0.8)
    fig.suptitle("E4 convergence basin (vdash_scripted, c=2.0, L0 — deterministic; each cell = mean "
                 "over tilt azimuths)\nthe controller recovers offset to 5 mm & tilt to ~8°", fontsize=10)
    out = os.path.join(FIGS, "e4_basin.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def _parse_ingrip(logpath, pol):
    if not os.path.exists(logpath):
        return [], []
    txt = open(logpath, errors="ignore").read()
    tilt, lat = [], []
    for m in re.finditer(rf"graspdiag\] pol={pol} .*?HANDOFF ingrip_tilt=([0-9.]+)deg lat=([0-9.]+)mm", txt):
        tilt.append(float(m.group(1)))
        lat.append(float(m.group(2)))
    return tilt, lat


def fig_ingripper():
    st, sl = _parse_ingrip(os.path.join(LOGS, "graspdiag_scripted.out"), "scripted")
    vt, vl = _parse_ingrip(os.path.join(LOGS, "graspdiag_vla.out"), "vla")
    if not (st and vt):
        print("skip ingripper (missing graspdiag logs)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for ax, sval, vval, lbl in ((axes[0], st, vt, "in-gripper tilt (deg)"),
                                (axes[1], sl, vl, "in-gripper lateral (mm)")):
        ax.bar([0, 1], [np.mean(sval), np.mean(vval)], width=0.5,
               color=["#4c9f70", "#c0504d"], zorder=1)
        ax.scatter(np.zeros(len(sval)), sval, color="k", s=14, zorder=2)
        ax.scatter(np.ones(len(vval)), vval, color="k", s=14, zorder=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"scripted\n(98% insert)", f"VLA v6\n(20% insert)"])
        ax.set_ylabel(lbl)
        ax.text(0, np.mean(sval), f" {np.mean(sval):.1f}", va="bottom", ha="center")
        ax.text(1, np.mean(vval), f" {np.mean(vval):.1f}", va="bottom", ha="center")
    fig.suptitle("In-gripper grasp quality at handoff — the VLA holds the peg cocked "
                 "(§2.1-blind controller can't correct it)", fontsize=10)
    out = os.path.join(FIGS, "ingripper.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def fig_cheat():
    labels = ["nominal\ngrasp", "8 mm in-gripper\noffset", "offset +\ntrue tip (cheat)"]
    vals = [98, 0, 55]
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    bars = ax.bar(labels, vals, color=["#4c9f70", "#c0504d", "#4f81bd"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}%", ha="center", va="bottom")
    ax.set_ylabel("insert success (%)  (c=2.0/L1)")
    ax.set_ylim(0, 105)
    ax.set_title("The failure IS the tip-estimate error\n(give the controller the true tip → 0% → 55%)",
                 fontsize=10)
    out = os.path.join(FIGS, "cheat.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def fig_e2():
    path = os.path.join(RES, "e2_clearance_final.csv")
    if not os.path.exists(path):
        print("skip e2_clearance (grid not finished)")
        return
    rows = [r for r in csv.DictReader(open(path)) if r.get("success_rate") not in (None, "")]
    if not rows:
        print("skip e2_clearance (csv empty)")
        return
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for level, color in (("L0", "#999999"), ("L1", "#4f81bd")):
        pts = sorted([(float(r["clearance_mm"]), 100 * float(r["success_rate"]),
                       100 * float(r["ci95_low"]), 100 * float(r["ci95_high"]))
                      for r in rows if r["level"] == level])
        if not pts:
            continue
        c, sr, lo, hi = zip(*pts)
        ax.errorbar(c, sr, yerr=[np.array(sr) - np.array(lo), np.array(hi) - np.array(sr)],
                    marker="o", capsize=3, color=color, label=level)
    ax.set_xlabel("radial clearance c (mm)")
    ax.set_ylabel("insert success (%)")
    ax.set_title("E2 clearance curve (vdash_scripted, n=50, Wilson 95%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.invert_xaxis()
    ax.legend()
    out = os.path.join(FIGS, "e2_clearance.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_e4_basin()
    fig_ingripper()
    fig_cheat()
    fig_e2()
    print("done. figs ->", FIGS)
