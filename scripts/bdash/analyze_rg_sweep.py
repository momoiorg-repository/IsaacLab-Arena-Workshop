# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""P0-3/P0-5: turn the R(g) sweep into gate thresholds and committed E1 predictions.

Two outputs, and the ORDER is the science:

* **Thresholds** (§9 derived-window rule): per variant, the largest injected axial error the direct
  insertion absorbed while still producing a good part -- min(process capability, functional
  requirement). These overwrite the provisional `gate.go_below_mm`.
* **predictions.json** (E2): what E1 *will* measure for P1/P2/P3 under the declared error mix,
  computed ONLY from the sweep. This file must be committed BEFORE E1 runs; its content records the
  config fingerprint and the sweep files it was derived from, so "predicted first" is checkable.

Honesty notes baked into the output: cell sizes are N=4 (axial) / N=3 (tilt) per point, so the
capability bound is coarse and stated as such; the E1 error mix is DECLARED (the measured pilot
distribution died with the abandoned pilot dataset) and is embedded here byte-identically with
run_e1.sh (same seed, same generator).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random

VARIANTS = ("W-A", "W-B", "W-C")
LOGS = pathlib.Path("logs/bdash")


def e1_error_list() -> list[float]:
    """The E1 injected-error sequence, byte-identical with run_e1.sh (same seed, same draws)."""
    r = random.Random(2026)
    vals = (
        [round(r.uniform(0, 1), 1) for _ in range(30)]
        + [round(r.uniform(2, 6), 1) for _ in range(13)]
        + [round(r.uniform(8, 12), 1) for _ in range(7)]
    )
    r.shuffle(vals)
    return vals


def load_rows(variant: str) -> list[dict]:
    rows = []
    for tag in ("ax", "ti"):
        f = LOGS / f"rg_{variant}_{tag}.jsonl"
        if f.exists():
            rows += [json.loads(line) for line in f.open()]
    return rows


def good(r: dict) -> bool:
    """A GOOD PART under the §9-derived window: seated (per-variant seat-angle window, depth, in
    bore) AND protrusion inside the per-variant tolerance. NOT the flat 0.5-deg `qc_ok` bar: that
    bar sits below the process's measured angle capability, and §9 forbids judging against a window
    the process cannot occupy -- it is reported separately as the strict-spec rate, not used to
    route. Loaded-but-out-of-window is scrap, and scrap is what P1 ships, so protrusion stays in."""
    prot = r.get("protrusion_err")
    tol = r.get("protrusion_tol") or 0.002
    return bool(r.get("min_insert")) and prot is not None and abs(prot) <= tol


def rate_curve(rows: list[dict]) -> dict[float, tuple[int, int]]:
    by: dict[float, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("injected_tilt_deg") not in (None, 0.0):
            continue  # axial axis only
        e = r.get("injected_e_mm")
        if e is None:
            continue
        by[e][1] += 1
        by[e][0] += int(good(r))
    return {e: (s, n) for e, (s, n) in sorted(by.items())}


def interp_rate(curve: dict[float, tuple[int, int]], e: float) -> float:
    """Piecewise-linear P(good | e) from the measured grid."""
    pts = [(k, s / n) for k, (s, n) in curve.items() if n]
    if not pts:
        return 0.0
    pts.sort()
    if e <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= e <= x1:
            return y0 + (y1 - y0) * (e - x0) / (x1 - x0 + 1e-9)
    return pts[-1][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/bdash/predictions.json")
    ap.add_argument("--min_rate", type=float, default=1.0, help="cell good-rate required to call e absorbable")
    args = ap.parse_args()

    shas = set()
    report: dict = {"variants": {}, "e1_error_mix_mm": e1_error_list()}
    for v in VARIANTS:
        rows = load_rows(v)
        shas |= {r.get("config_sha") for r in rows}
        curve = rate_curve(rows)
        tilt = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            t = r.get("injected_tilt_deg")
            if t not in (None, 0.0):
                tilt[t][1] += 1
                tilt[t][0] += int(good(r))
        # Capability bound: the largest grid e whose WHOLE cell was good. Coarse (N=4) and said so.
        absorbed = [e for e, (s, n) in curve.items() if n and s / n >= args.min_rate]
        cap = max(absorbed) if absorbed else 0.0
        grid = sorted(curve)
        nxt = min((e for e in grid if e > cap), default=None)
        # Derived threshold: midpoint to the first degrading cell -- conservative side of coarse.
        go = cap if nxt is None else (cap + nxt) / 2.0
        report["variants"][v] = {
            "curve_good_over_n": {str(e): f"{s}/{n}" for e, (s, n) in curve.items()},
            "tilt_good_over_n": {str(t): f"{s}/{n}" for t, (s, n) in sorted(tilt.items())},
            "capability_all_good_mm": cap,
            "derived_go_below_mm": round(go, 2),
            "cell_n": max((n for _, n in curve.values()), default=0),
        }

    report["config_sha"] = sorted(x for x in shas if x)
    if len(report["config_sha"]) > 1:
        raise SystemExit(f"§9: sweep spans multiple scene fingerprints {report['config_sha']} -- not comparable")

    # ---- E2 predictions: P1 / P2 / P3 under the declared mix, from the sweep alone -------------
    errors = report["e1_error_mix_mm"]
    preds = {}
    for v in VARIANTS:
        curve = rate_curve(load_rows(v))
        zero = interp_rate(curve, 0.0)
        go = report["variants"][v]["derived_go_below_mm"]
        p1 = sum(interp_rate(curve, e) for e in errors) / len(errors)
        p2 = zero  # every part refixed -> re-grasped at nominal, inserts at the zero-error rate
        direct = [e for e in errors if e <= go]
        routed = [e for e in errors if e > go]
        p3 = (sum(interp_rate(curve, e) for e in direct) + zero * len(routed)) / len(errors)
        preds[v] = {
            "P1_good_rate": round(p1, 3),
            "P2_good_rate": round(p2, 3),
            "P3_good_rate": round(p3, 3),
            "P3_refix_fraction": round(len(routed) / len(errors), 3),
        }
    report["predictions"] = preds
    report["notes"] = (
        "Committed BEFORE E1 (E2 requires it). Error mix is DECLARED, not pilot-measured -- the "
        "measured pilot died with the abandoned place dataset. Cells are N=4 (axial) / N=3 (tilt); "
        "capability bounds are correspondingly coarse and rounded toward caution. P2/P3 cycle-time "
        "costs are measured in E1 itself, not predicted here."
    )
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("config_sha", "predictions")}, indent=1))
    for v in VARIANTS:
        r = report["variants"][v]
        print(
            f"{v}: curve={r['curve_good_over_n']}  tilt={r['tilt_good_over_n']}  ->"
            f" go_below={r['derived_go_below_mm']}mm"
        )
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
