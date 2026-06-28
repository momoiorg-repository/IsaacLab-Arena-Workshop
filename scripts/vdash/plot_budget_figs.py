# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Precision-budget analysis + figures for V-DASH (docs/vdash_precision_budget_plan.md).

Consumes the scripted-test-bed sweeps (``results/vdash/repro/sweeps.csv``) and the VLA in-gripper
grasp distribution (``results/vdash/repro/grasp_diag_vla_v6.csv``) and produces:
  * R(l) blind-channel response, R_oracle(l), and R_widened(σ) curves
  * the success-tail separation scatter (VLA grasps coloured by outcome)
  * the NON-CIRCULAR convolution prediction: P_pred = mean_g R(e(g)) over the VLA's measured grasp
    distribution (e(g) = lateral + grip_offset·sin tilt), compared to the observed VLA success.

Run (host or container): python plot_budget_figs.py
"""
from __future__ import annotations

import csv
import math
import numpy as np
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results", "vdash", "repro")
FIGS = os.path.join(REPO, "figs", "vdash")
GRIP_OFFSET_MM = 60.0  # tip -> grip-cube centre; effective tip bias e(g) = lat + grip_offset·sin(tilt)


def _dictrows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _readlines(path):
    with open(path) as fh:
        return fh.readlines()


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load_sweeps():
    rows = []
    p = os.path.join(RES, "sweeps.csv")
    if not os.path.exists(p):
        return rows
    for r in _dictrows(p):
        try:
            r["k"] = int(r["k"])
            r["n"] = int(r["n"])
        except (ValueError, KeyError):
            continue
        for f in ("off_mm", "noise_mm"):
            r[f] = float(r[f])
        try:
            r["realized_lat_mm"] = float(r["realized_lat_mm"])
        except (ValueError, KeyError):
            r["realized_lat_mm"] = r["off_mm"]
        # sanitize: a dropped env-0 grasp gives a nonsense grasp-diag lateral; fall back to the knob
        if r["realized_lat_mm"] <= 0 or r["realized_lat_mm"] > r["off_mm"] + 6:
            r["realized_lat_mm"] = r["off_mm"]
        rows.append(r)
    return rows


def load_grasp():
    rows = []
    p = os.path.join(RES, "grasp_diag_vla_v6.csv")
    if not os.path.exists(p):
        return rows
    for r in _dictrows(p):
        if not r["ingrip_lat_mm"] or r["ingrip_lat_mm"] in ("None", ""):
            continue
        rows.append({
            "lat": float(r["ingrip_lat_mm"]),
            "tilt": float(r["ingrip_tilt_deg"]),
            "result": r["result"],
        })
    return rows


def eff_err(lat_mm, tilt_deg):
    return lat_mm + GRIP_OFFSET_MM * math.sin(math.radians(tilt_deg))


def main():
    os.makedirs(FIGS, exist_ok=True)
    sw = load_sweeps()
    grasp = load_grasp()
    blind = sorted([r for r in sw if r["experiment"] == "rblind"], key=lambda r: r["realized_lat_mm"])
    oracle = sorted([r for r in sw if r["experiment"] == "roracle"], key=lambda r: r["off_mm"])
    widen = sorted([r for r in sw if r["experiment"] == "rwiden"], key=lambda r: r["noise_mm"])

    # ---- blind-channel response R as a fn of realized in-gripper lateral (the test-bed has ~0 tilt) ----
    bx = [r["realized_lat_mm"] for r in blind]
    by = [r["k"] / r["n"] for r in blind]
    print("R(l) blind response:", [(round(x, 1), round(y, 2)) for x, y in zip(bx, by)])
    print("R_oracle(l):", [(r["off_mm"], round(r["k"] / r["n"], 2)) for r in oracle])
    print("R_widened(σ):", [(r["noise_mm"], round(r["k"] / r["n"], 2)) for r in widen])

    # ---- NON-CIRCULAR convolution prediction ----
    pred_line = ""
    if grasp and blind:
        handoffs = [g for g in grasp]  # rows with a measured grasp = reached handoff
        # R interpolated at each VLA grasp's EFFECTIVE error e(g); test-bed x ~ effective error (tilt~0)
        xs = np.array(bx)
        ys = np.array(by)

        def R_at(e):
            return float(np.interp(e, xs, ys, left=ys[0], right=ys[-1]))

        per_handoff = [R_at(eff_err(g["lat"], g["tilt"])) for g in handoffs]
        per_handoff_lat = [R_at(g["lat"]) for g in handoffs]  # lateral-only variant (ignores tilt)
        n_eps = 20  # episodes in the VLA eval
        handoff_rate = len(handoffs) / n_eps
        p_e2e_eff = handoff_rate * float(np.mean(per_handoff))
        p_e2e_lat = handoff_rate * float(np.mean(per_handoff_lat))
        obs_k = sum(1 for g in grasp if g["result"] == "success")
        obs = obs_k / n_eps
        pred_line = (
            "CONVOLUTION PREDICTION (P = handoff_rate * mean_g R(e(g))):\n"
            f"  handoff_rate={handoff_rate:.2f} ({len(handoffs)}/{n_eps})\n"
            f"  P_pred (effective err e=lat+60·sinθ) = {p_e2e_eff:.3f}\n"
            f"  P_pred (lateral only)                = {p_e2e_lat:.3f}\n"
            f"  OBSERVED VLA success                 = {obs:.3f} ({obs_k}/{n_eps})  "
            f"CI95={tuple(round(v, 3) for v in wilson(obs_k, n_eps)[1:])}"
        )
        print(pred_line)

    # ---- figures ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); printed results only.")
        return

    # Fig 1: R(l), R_oracle(l), R_widened(σ)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if blind:
        ax[0].plot(bx, by, "o-", label="blind R(l)  (no correction)", color="crimson")
    if oracle:
        ox = [r["off_mm"] for r in oracle]
        oy = [r["k"] / r["n"] for r in oracle]
        ax[0].plot(ox, oy, "s--", label="oracle R(l)  (true tip fed)", color="seagreen")
    if grasp:
        ax0b = ax[0].twinx()
        ax0b.hist([g["lat"] for g in grasp], bins=8, range=(0, 16), alpha=0.18, color="navy")
        ax0b.set_ylabel("VLA grasps (count)", color="navy")
    ax[0].set_xlabel("in-gripper lateral (mm)")
    ax[0].set_ylabel("insert success")
    ax[0].set_title("Blind vs oracle response (with VLA p_grasp)")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend(loc="upper right", fontsize=8)
    if widen:
        wx = [r["noise_mm"] for r in widen]
        wy = [r["k"] / r["n"] for r in widen]
        ax[1].plot(wx, wy, "o-", color="darkorange")
        ax[1].set_xlabel("estimator lateral RMS σ (mm)")
        ax[1].set_ylabel("insert success @ 8mm offset")
        ax[1].set_title("R_widened(σ): probe accuracy spec")
        ax[1].set_ylim(-0.02, 1.02)
        ax[1].axhline(0.55, ls=":", color="seagreen", label="oracle 55%")
        ax[1].axhline(0.0, ls=":", color="crimson", label="blind 0%")
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "budget_response.png"), dpi=130)
    print("wrote", os.path.join(FIGS, "budget_response.png"))

    # Fig 2: success-tail separation
    if grasp:
        fig2, axx = plt.subplots(figsize=(5.5, 4.5))
        for res, col, mk in [
            ("success", "seagreen", "*"),
            ("insertion_failed", "crimson", "x"),
            ("timeout", "gray", "."),
            ("dropped", "black", "+"),
        ]:
            pts = [(g["lat"], g["tilt"]) for g in grasp if g["result"] == res]
            if pts:
                axx.scatter([p[0] for p in pts], [p[1] for p in pts], c=col, marker=mk, s=90, label=res)
        axx.set_xlabel("in-gripper lateral (mm)")
        axx.set_ylabel("in-gripper tilt (deg)")
        axx.set_title("VLA grasp quality vs outcome (tail separation)")
        axx.legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(os.path.join(FIGS, "budget_tail_separation.png"), dpi=130)
        print("wrote", os.path.join(FIGS, "budget_tail_separation.png"))


def fig_widening():
    """Grouped bars: blind vs oracle (per-step true tip) vs tactile probe, per injected offset.
    Parsed from logs/vdash/repro/probe.log ('[probe] cl off=X mode: k/n')."""
    import re

    p = os.path.join(REPO, "logs", "vdash", "repro", "probe.log")
    if not os.path.exists(p):
        return
    data = {}
    for ln in _readlines(p):
        m = re.search(r"cl off=(\d+) (\w+): (\d+)/(\d+)", ln)
        if m:
            data[(int(m.group(1)), m.group(2))] = (int(m.group(3)), int(m.group(4)))
    if not data:
        return
    offs = sorted({o for o, _ in data})
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    modes = [("blind", "crimson"), ("oracle", "seagreen"), ("cal", "darkorange")]
    x = np.arange(len(offs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (mode, col) in enumerate(modes):
        vals = [(data.get((o, mode), (0, 1))[0] / data.get((o, mode), (0, 1))[1]) for o in offs]
        ax.bar(x + (i - 1) * w, vals, w, label=mode, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o} mm" for o in offs])
    ax.set_xlabel("injected in-gripper offset")
    ax.set_ylabel("insert success")
    ax.set_ylim(0, 1.05)
    ax.set_title("Interface-widening: §2.1 tactile probe vs blind vs oracle")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGS, "budget_interface_widening.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def fig_decant():
    """Grouped bars: blind / oracle(tip-only) / decant(tilt-only) / full(de-cant + lateral probe) vs
    injected tilt. Parsed from logs/vdash/repro/decant.log ('[decant] off_deg=X mode: k/n')."""
    import re

    p = os.path.join(REPO, "logs", "vdash", "repro", "decant.log")
    if not os.path.exists(p):
        return
    data = {}
    for ln in _readlines(p):
        m = re.search(r"off_deg=(\d+) (\w+): (\d+)/(\d+)", ln)
        if m:
            data[(int(m.group(1)), m.group(2))] = (int(m.group(3)), int(m.group(4)))
    if not data:
        return
    degs = sorted({d for d, _ in data})
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    modes = [("blind", "crimson"), ("oracle", "seagreen"), ("decant", "mediumpurple"), ("full", "darkorange")]
    x = np.arange(len(degs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (mode, col) in enumerate(modes):
        vals = [(data.get((d, mode), (0, 1))[0] / data.get((d, mode), (0, 1))[1]) for d in degs]
        ax.bar(x + (i - 1.5) * w, vals, w, label=mode, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}° tilt\n(~{d}mm lat)" for d in degs])
    ax.set_xlabel("injected in-gripper cock (tilt + coupled lateral)")
    ax.set_ylabel("insert success")
    ax.set_ylim(0, 1.05)
    ax.set_title("Tilt fix: finger-force de-cant + bore-edge probe (§2.1) vs oracle/blind")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGS, "budget_tilt_fix.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
    fig_widening()
    fig_decant()
