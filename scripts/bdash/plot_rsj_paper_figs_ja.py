# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate the RSJ2026 B-DASH paper figures with Japanese labels.

Labels follow the paper terminology exactly (切り替え時 / 横ずれ / 傾き /
嵌合成功率 / 把持応答 R / 収束域 B / 微調整VLA / go・no-go / 観測補正).
In-figure titles are omitted; the captions carry the description.

Run inside the container (IPAexGothic required):
  /isaac-sim/python.sh scripts/bdash/plot_rsj_paper_figs_ja.py
Outputs -> figs/bdash/{multiseed_v6,e4_basin,budget_response,clearance_cliff,budget_gate,grasp_compare}.png
"""

import csv
import json
import matplotlib
import os
from collections import defaultdict

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.basename(REPO) not in ("isaaclab_arena", "IsaacLab-Arena-An"):
    REPO = "/workspaces/isaaclab_arena"
RES = os.path.join(REPO, "results/bdash/repro")
FIGS = os.path.join(REPO, "figs/bdash")

for cand in (
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/truetype/ipaexfont-gothic/ipaexg.ttf",
):
    if os.path.exists(cand):
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        JP_FONT = cand
        break
else:
    raise RuntimeError("IPAexGothic not found; run inside the container")
plt.rcParams.update({"font.size": 13, "axes.unicode_minus": False})


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(row for row in f if not row.startswith("#"))]


def fig_multiseed():
    rates = []
    for s in range(5):
        r = read_rows(os.path.join(RES, f"eval_v6_s{s}.csv"))[0]
        rates.append(float(r["success_rate"]) * 100)
    mean, sd = float(np.mean(rates)), float(np.std(rates, ddof=1))
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.axhspan(13, 29, color="tab:green", alpha=0.10, label="95%信頼区間 13–29%")
    ax.axhspan(mean - sd, mean + sd, color="tab:red", alpha=0.10, label=f"平均±標準偏差 {mean:.0f}%±{sd:.1f}%")
    ax.bar(range(5), rates, width=0.55, color="steelblue", label="シード別，各20試行")
    for i, v in enumerate(rates):
        ax.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=12)
    ax.axhline(mean, color="tab:red", lw=2)
    ax.set_xlabel("乱数シード")
    ax.set_ylabel("嵌合成功率 (%)")
    ax.set_ylim(0, 42)
    ax.set_xticks(range(5))
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "multiseed_v6.png"), dpi=150)
    plt.close(fig)
    print("wrote multiseed_v6.png", rates)


def fig_basin():
    # extended sweep (r to 10 mm, tilt to 12 deg); deterministic placement,
    # each cell = mean over 12 error azimuths (origin cell = 1 trial)
    acc = defaultdict(list)
    for r in read_rows(os.path.join(REPO, "results/bdash/e4_basin_az12.csv")):
        acc[(float(r["r_mm"]), float(r["tilt_deg"]))].append(float(r["success"]))
    rs = sorted({k[0] for k in acc})
    ts = sorted({k[1] for k in acc})
    grid = np.array([[100 * np.mean(acc[(r, t)]) for r in rs] for t in ts])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=100, aspect="auto")
    for i in range(len(ts)):
        for j in range(len(rs)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10.5, color="white" if v < 55 else "black")
    ax.set_xlabel("切り替え時の横ずれ r (mm)")
    ax.set_ylabel("切り替え時の傾き θ (deg)")
    ax.set_xticks(range(len(rs)), [f"{r:.0f}" for r in rs])
    ax.set_yticks(range(len(ts)), [f"{t:.0f}" for t in ts])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("嵌合成功率 (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "e4_basin.png"), dpi=150)
    plt.close(fig)
    print("wrote e4_basin.png overall", grid.mean().round(1))


def _sweep(prefix, suffix, offs):
    xs, ys = [], []
    for o in offs:
        path = os.path.join(RES, f"{prefix}{o}{suffix}.csv")
        if not os.path.exists(path):
            continue
        r = read_rows(path)[0]
        xs.append(o)
        ys.append(float(r["success_rate"]))
    return xs, ys


def fig_response():
    bx, by = _sweep("sw_rblind_off", "_n0_c0", [0, 2, 3, 4, 5, 6, 8, 10, 12])
    ox, oy = _sweep("sw_roracle_off", "_n0_c1", [0, 4, 6, 8, 10, 12])
    with open(os.path.join(RES, "pg_multiseed.json")) as f:
        lat = json.load(f)["lat"]
    plt.rcParams["font.size"] = 16  # near-final-size canvas: ~8pt effective at 0.73 columnwidth
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax2 = ax.twinx()
    ax2.hist(lat, bins=np.arange(0, 19, 1.5), color="lightsteelblue", alpha=0.55, zorder=0)
    ax2.set_ylabel("VLAの把持数", color="steelblue")
    ax2.tick_params(axis="y", colors="steelblue")
    ax.plot(bx, by, "o-", color="tab:red", lw=2.5, zorder=5, label="把持応答 $R$")
    ax.plot(ox, oy, "s--", color="tab:blue", lw=2, zorder=5, label="オラクル")
    ax.set_xlabel("グリッパ内の横ずれ l (mm)")
    ax.set_ylabel("嵌合成功率")
    ax.set_ylim(-0.03, 1.05)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "budget_response.png"), dpi=150)
    plt.close(fig)
    plt.rcParams["font.size"] = 13
    print("wrote budget_response.png", list(zip(bx, by)))


def fig_cliff():
    acc = defaultdict(list)
    for s in range(5):
        for r in read_rows(os.path.join(RES, f"clearance_sweep_s{s}.csv")):
            # c <= 1.0 is uniformly 0% and adds no information; keep the informative range
            if float(r["clearance_mm"]) >= 1.25:
                acc[float(r["clearance_mm"])].append(float(r["success_rate"]) * 100)
    cs = sorted(acc)
    means = [np.mean(acc[c]) for c in cs]
    sds = [np.std(acc[c], ddof=1) for c in cs]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.axvspan(1.25, 1.5, color="tab:red", alpha=0.10)
    ax.errorbar(cs, means, yerr=sds, fmt="o-", color="steelblue", lw=2.5, capsize=4, markersize=7)
    ax.annotate("崖 1.25–1.5 mm", (1.375, 88), ha="center", fontsize=12, color="firebrick")
    ax.annotate(
        "1.25 mm 以下では0%", (1.27, 8), fontsize=11, xytext=(1.42, 22), arrowprops=dict(arrowstyle="->", color="gray")
    )
    ax.annotate(
        "限界 c≈1.5 mm",
        (1.5, float(means[cs.index(1.5)])),
        fontsize=11,
        xytext=(1.62, 55),
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.set_xlabel("クリアランス c (mm)")
    ax.set_ylabel("嵌合成功率 (%)")
    ax.set_xticks(cs)
    ax.set_ylim(-5, 108)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "clearance_cliff.png"), dpi=150)
    plt.close(fig)
    print("wrote clearance_cliff.png", list(zip(cs, np.round(means, 1))))


def fig_gate():
    rows = read_rows(os.path.join(RES, "budget_gate_sweep.csv"))
    tilt = [float(r["injected_tilt_deg"]) for r in rows]
    naive = [float(r["naive_success_pct"]) for r in rows]
    nogo = [float(r["gate_nogo_rate_pct"]) for r in rows]
    corr = [float(r["observable_corrected_pct"]) for r in rows]

    def panel_a(ax):
        th = np.degrees(np.arcsin(5.0 / 60.0))  # budget 5 mm <-> tilt threshold (L = 60 mm)
        ax.axvspan(-1.5, th, color="tab:green", alpha=0.08)
        ax.axvspan(th, 13.5, color="tab:red", alpha=0.08)
        ax.axvline(th, color="#555555", ls="--", lw=1.6)
        ax.set_ylim(0, 114)
        ax.text(th + 0.25, 104, "設計しきい値 4.8°", fontsize=14, color="#444444", ha="left")
        ax.bar(tilt, naive, width=1.6, color="lightblue", edgecolor="steelblue")
        ax.plot(tilt, nogo, "o-", color="tab:orange", lw=3, markersize=8)
        ax.annotate(
            "嵌合成功率",
            (0.9, 82),
            color="steelblue",
            fontsize=14,
            arrowprops=dict(arrowstyle="->", color="steelblue"),
            xytext=(2.6, 90),
        )
        ax.annotate(
            "no-go 率",
            (8.05, 87),
            color="tab:orange",
            fontsize=14,
            xytext=(8.8, 60),
            arrowprops=dict(arrowstyle="->", color="tab:orange"),
        )
        ax.text(1.2, 6, "go", color="green", fontsize=17, fontweight="bold")
        ax.text(9.7, 6, "no-go", color="firebrick", fontsize=17, fontweight="bold")
        ax.set_xlabel("注入した把持傾き (deg)")
        ax.set_ylabel("割合 (%)")
        ax.set_xticks(tilt)
        ax.set_xlim(-1.5, 13.5)

    def panel_b(ax):
        ax.plot(tilt, naive, "o--", color="gray", lw=2, markersize=8, label="補正なし")
        ax.plot(tilt, corr, "o-", color="tab:green", lw=3, markersize=8, label="観測補正あり")
        for x, y0, y1 in ((4, 55, 100), (8, 5, 30)):
            ax.annotate(
                "", (x, y1 - 2), xytext=(x, y0 + 2), arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.8)
            )
            dy = 0 if x == 4 else -14
            ax.annotate(
                f"{y0}→{y1}%", (x, (y0 + y1) / 2), fontsize=14, color="tab:green", xytext=(x + 0.35, (y0 + y1) / 2 + dy)
            )
        ax.set_xlabel("注入した把持傾き (deg)")
        ax.set_ylabel("嵌合成功率 (%)")
        ax.set_xticks(tilt)
        ax.set_ylim(-5, 108)
        ax.legend(loc="lower left", fontsize=12)

    # standalone panels for the paper (near-final size: ~8pt effective at 0.73 columnwidth)
    plt.rcParams["font.size"] = 16
    for name, draw in (("budget_gate_a.png", panel_a), ("budget_gate_b.png", panel_b)):
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        draw(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, name), dpi=150)
        plt.close(fig)
        print(f"wrote {name}")

    # combined two-panel version (slides)
    plt.rcParams["font.size"] = 13
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    panel_a(axes[0])
    axes[0].set_title("(a) ゲート判定", fontsize=13)
    panel_b(axes[1])
    axes[1].set_title("(b) 観測補正による回復", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "budget_gate.png"), dpi=150)
    plt.close(fig)
    print("wrote budget_gate.png")


def fig_grasp_headers():
    """Rewrite the two header lines of grasp_compare.png (keep the renders)."""
    from PIL import Image, ImageDraw, ImageFont

    path = os.path.join(FIGS, "grasp_compare.png")
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    # the header text sits above the photos on white; the photo's top-left
    # corner is dark background, so the first run of dark rows at x=25 marks
    # the top of the photo panels
    band, run = 0, 0
    for y in range(h):
        if sum(px[25, y]) < 350:
            run += 1
            if run >= 5:
                band = y - run + 1
                break
        else:
            run = 0
    if not 20 <= band <= h // 3:
        raise RuntimeError(f"header band detection failed (band={band}px)")
    # enlarge the header band so the labels print at ~8pt (width-limited: keep labels short)
    grow = 24
    canvas = Image.new("RGB", (w, h + grow), "white")
    canvas.paste(im, (0, grow))
    im = canvas
    h = h + grow
    band = band + grow
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, w, band - 1], fill="white")
    left = "真っ直ぐな把持"
    right = "約20°傾いた把持"
    size = min(int(band * 0.72), int(w * 0.46 / max(len(left), len(right))))
    font = ImageFont.truetype(JP_FONT, size)
    for text, cx, color in ((left, w * 0.25, (27, 118, 42)), (right, w * 0.75, (178, 24, 24))):
        tw = draw.textlength(text, font=font)
        draw.text((cx - tw / 2, (band - size) * 0.45), text, fill=color, font=font)
    im.save(path)
    print(f"rewrote grasp_compare.png headers (band={band}px, font={size}px)")


if __name__ == "__main__":
    fig_multiseed()
    fig_basin()
    fig_response()
    fig_cliff()
    fig_gate()
    fig_grasp_headers()


def fig_vision():
    """図1: B-DASH全体像 (日本語・本文の術語と1対1)."""
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14.2, 5.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x0, y0, x1, y1, fc="#ececec", ec="#555555", lw=1.4, ls="-", r=1.2, z=2):
        b = FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            ec=ec,
            fc=fc,
            lw=lw,
            linestyle=ls,
            boxstyle=f"round,pad=0,rounding_size={r}",
            zorder=z,
        )
        ax.add_patch(b)

    def arrow(p0, p1, color="#37474f", lw=2.2, ls="-", z=4, style="-|>", ms=16):
        ax.add_patch(
            FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls, zorder=z)
        )

    def txt(x, y, s, size=12, weight="normal", color="#212121", ha="center", va="center", z=5):
        ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z)

    # ---- top row ----------------------------------------------------------
    box(4, 52, 17.5, 94)
    txt(10.75, 86, "VLMプランナ", 13, "bold")
    txt(10.75, 80, "(Organizer)", 11)
    txt(10.75, 66, "タスクを\nサブタスクへ分割", 12)
    arrow((0.2, 73), (3.8, 73))
    txt(1.8, 79, "タスク\n指示", 11)

    box(21, 52, 33.5, 94, fc="#f5f5f5")
    txt(27.25, 88.5, "サブタスク列", 13, "bold")
    for i, s in enumerate(["1. 搬送", "2. 把持・搬送", "3. 精密投入"]):
        box(22.3, 79 - 9.5 * i - 5.5, 32.2, 79 - 9.5 * i + 1.5, fc="white")
        txt(27.25, 77 - 9.5 * i, s, 11.5)
    arrow((17.7, 73), (20.8, 73))

    box(37.5, 52, 56.5, 94, fc="#f5f5f5")
    txt(47, 88.5, "スキルライブラリ", 13, "bold")
    cells = [
        ("VLA", 38.8, 75.0, 46.3, 85.5),
        ("強化学習\n方策", 47.7, 75.0, 55.2, 85.5),
        ("ルールベース制御", 38.8, 63.0, 55.2, 73.5),
    ]
    for s, x0, y0, x1, y1 in cells:
        box(x0, y0, x1, y1, fc="white")
        txt((x0 + x1) / 2, (y0 + y1) / 2, s, 10.5)
    txt(47, 57.5, "仕様 = 実測した許容誤差・\n残留誤差・完了条件", 9.5, color="#333333")
    arrow((33.7, 73), (37.3, 73))
    txt(35.5, 79, "スキル\n割り当て", 10.5)

    # executed sequence
    txt(78.5, 97.5, "実行されるスキル列", 13, "bold")
    txt(93.5, 97.4, "machine tending の例", 10.5, color="#555555")
    seq = [
        ("搬送", "ルールベース\n制御", 59.5, 68.5),
        ("把持・搬送", "VLA", 72.5, 81.5),
        ("精密投入", "ルールベース\n力制御", 85.5, 97.5),
    ]
    for name, impl, x0, x1 in seq:
        box(x0, 62, x1, 86, fc="#ececec")
        txt((x0 + x1) / 2, 79, name, 12.5, "bold")
        txt((x0 + x1) / 2, 69.5, impl, 10.5)
    aspect = 14.2 / 5.9
    for gx in (70.5, 83.5):
        ax.add_patch(Ellipse((gx, 74), 2.3, 2.3 * aspect, fc="#f5a623", ec="#8a5a00", lw=1.3, zorder=6))
        txt(gx, 74, "G", 12, "bold", color="white", z=7)
        arrow((gx - 2.0, 74), (gx - 0.4, 74), ms=11)
        arrow((gx + 0.4, 74), (gx + 2.0, 74), ms=11)
    arrow((56.7, 73), (59.3, 73))
    txt(58, 78.5, "実行", 10.5)
    # this-work dashed frame
    box(72.0, 58.5, 99.2, 89.5, fc="none", ec="#8b1a1a", lw=2.0, ls=(0, (5, 3)), z=3)
    txt(85.6, 91.5, "本稿の範囲", 12, "bold", color="#8b1a1a")
    txt(78.5, 50.5, "各境界の残留誤差 = 補正できる誤差 + 後段非観測誤差", 10.5, color="#555555")

    # ---- bottom band: inside gate G ---------------------------------------
    box(4, 4, 99.2, 42, fc="#fdf6dc", ec="#8b6d1f", lw=1.6, r=2)
    txt(51.5, 37, "スキル連結調整ゲート G の内部", 13.5, "bold")
    flow = [("力覚などの\n観測信号", 7, 24), ("後段非観測誤差を\n推定", 28, 45), ("実効誤差が\n許容誤差以内か", 49, 66)]
    for s, x0, x1 in flow:
        box(x0, 12, x1, 30, fc="white")
        txt((x0 + x1) / 2, 21, s, 11.5)
    arrow((24.2, 21), (27.8, 21))
    arrow((45.2, 21), (48.8, 21))
    box(70.5, 23, 97.5, 32, fc="#dcefdc", ec="#2e7d32")
    txt(84, 27.5, "go — 目的値を調整し切り替え", 12, "bold", color="#1b5e20")
    box(70.5, 9, 97.5, 18, fc="#f7dcdc", ec="#b23b3b")
    txt(84, 13.5, "no-go — 前段をやり直す", 12, "bold", color="#8b1a1a")
    arrow((66.2, 23.5), (70.3, 27), color="#2e7d32")
    arrow((66.2, 18.5), (70.3, 14), color="#b23b3b")
    # replan feedback
    arrow((70.3, 11), (10.75, 11), color="#8b1a1a", ls=(0, (5, 3)), lw=1.8)
    arrow((10.75, 11), (10.75, 51.6), color="#8b1a1a", ls=(0, (5, 3)), lw=1.8)
    txt(38, 7.6, "やり直しても収まらなければ再計画", 11, color="#8b1a1a")
    # dotted link: gate G in sequence -> band
    arrow((83.5, 71.9), (83.5, 42.4), color="#8a5a00", ls=(0, (2, 3)), lw=1.5, style="-|>", ms=10)

    fig.savefig(os.path.join(FIGS, "bdash_vision.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote bdash_vision.png")
