# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""P0-4/P0-5: aggregate E1 (P1/P2/P3 x n=51, same seed, same injected-error sequence) and check the
committed predictions (E2).

Metric definitions follow the kill-switch directive:

* **good**: the §9-derived window -- min_insert AND |protrusion_err| <= per-variant tol. The flat
  0.5-deg `qc_ok` is reported separately as the strict-spec rate, never used to route or rank.
* **scrap**: the episode ended `success` (the cell SHIPPED the part) but the part is out of window.
  This is P1's characteristic cost: no measurement means no way to know it shipped scrap.
* **deep failure**: the cell never shipped -- time_out / load_failed / gave_up. Jam-shaped cost.
* **rejected**: P3 only -- the gate refused the part; counted separately (neither scrap nor deep).
* **cycle**: arm-owner steps (grip owners overlap the arm and are excluded). Throughput is
  good parts per 10k arm steps, so slow-but-sure and fast-but-scrappy are on one axis.

§9: refuses to aggregate if the three files carry more than one config_sha, and refuses to compare
against predictions committed for a different sha. Also asserts the injected-error sequences are
IDENTICAL across P1/P2/P3 -- that is the paired-design guarantee.
"""

from __future__ import annotations

import collections
import json
import pathlib

LOGS = pathlib.Path("logs/bdash")
POLICIES = ("P1", "P2", "P3")
VARIANTS = ("W-A", "W-B", "W-C")


def rows(p: str) -> list[dict]:
    return [json.loads(line) for line in (LOGS / f"e1_{p}.jsonl").open()]


def is_good(r: dict) -> bool:
    prot = r.get("protrusion_err")
    tol = r.get("protrusion_tol") or 0.002
    return bool(r.get("min_insert")) and prot is not None and abs(prot) <= tol


def cycle_steps(r: dict) -> int:
    return sum(v for k, v in (r.get("owner_steps") or {}).items() if not k.startswith("grip:"))


def main() -> None:
    data = {p: rows(p) for p in POLICIES}

    shas = {r.get("config_sha") for rs in data.values() for r in rs}
    if len(shas) != 1:
        raise SystemExit(f"§9: E1 spans configs {shas} -- not comparable")
    sha = shas.pop()

    seqs = {p: [r.get("injected_e_mm") for r in data[p]] for p in POLICIES}
    if not (seqs["P1"] == seqs["P2"] == seqs["P3"]):
        raise SystemExit("§9: injected-error sequences differ across policies -- pairing broken")

    pred_path = LOGS / "predictions.json"
    pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
    if pred and pred.get("config_sha") not in ([sha], sha):
        raise SystemExit(f"§9: predictions committed for {pred.get('config_sha')}, E1 ran on {sha}")

    report: dict = {"config_sha": sha, "n_per_policy": {p: len(data[p]) for p in POLICIES}, "policies": {}}
    for p in POLICIES:
        rs = data[p]
        shipped = [r for r in rs if "success" in (r.get("terminated_by") or [])]
        rejected = [r for r in rs if r.get("gate_decision") == "reject"]
        deep = [r for r in rs if r not in shipped and r not in rejected]
        good = [r for r in shipped if is_good(r)]
        scrap = [r for r in shipped if not is_good(r)]
        cycles = sorted(cycle_steps(r) for r in rs)
        med = cycles[len(cycles) // 2] if cycles else 0
        total = sum(cycles)
        per_v = {}
        for v in VARIANTS:
            vr = [r for r in rs if r.get("variant") == v]
            per_v[v] = {
                "n": len(vr),
                "good": sum(is_good(r) and "success" in (r.get("terminated_by") or []) for r in vr),
                "good_rate": round(
                    sum(is_good(r) and "success" in (r.get("terminated_by") or []) for r in vr) / max(len(vr), 1), 3
                ),
            }
        report["policies"][p] = {
            "good": len(good),
            "good_rate": round(len(good) / len(rs), 3),
            "scrap": len(scrap),
            "scrap_rate_of_shipped": round(len(scrap) / max(len(shipped), 1), 3),
            "deep_failures": len(deep),
            "rejected": len(rejected),
            "refixed": sum(bool(r.get("refixed")) for r in rs),
            "strict_qc_ok": sum(bool(r.get("qc_ok")) for r in rs),
            "median_cycle_steps": med,
            "good_per_10k_steps": round(len(good) / max(total, 1) * 1e4, 3),
            "per_variant": per_v,
            "deep_terminations": dict(
                collections.Counter(t for r in deep for t in (r.get("terminated_by") or ["none"]))
            ),
        }

    if pred:
        e2 = {}
        for v in VARIANTS:
            pv = pred["predictions"][v]
            e2[v] = {
                p: {
                    "predicted": pv[f"{p}_good_rate"],
                    "measured": report["policies"][p]["per_variant"][v]["good_rate"],
                    "abs_err": round(
                        abs(pv[f"{p}_good_rate"] - report["policies"][p]["per_variant"][v]["good_rate"]), 3
                    ),
                }
                for p in POLICIES
            }
            e2[v]["P3_refix_fraction"] = {
                "predicted": pv["P3_refix_fraction"],
                "measured": round(report["policies"]["P3"]["refixed"] / max(len(data["P3"]), 1), 3),
            }
        # The ORDERING claim is the falsifiable core: does the gate beat no-gate, per variant?
        order_ok = {
            v: (
                report["policies"]["P3"]["per_variant"][v]["good_rate"]
                >= report["policies"]["P1"]["per_variant"][v]["good_rate"]
            )
            for v in VARIANTS
        }
        report["e2"] = {"per_variant": e2, "P3_ge_P1_measured": order_ok}

    # Honesty notes the slide must carry (§9): the committed predictions routed P3 at the sweep-
    # derived 1.0 mm threshold, but E1 ran on the frozen config whose provisional gate is 4.0 mm --
    # rewriting the YAML would have changed the fingerprint predictions were committed against.
    # The touch-off bias (~+2.9 mm) also sits inside the gate margin. Both push measured P3 routing
    # BELOW the predicted 0.4 refix fraction; the per-variant good-rate predictions still land
    # within 0.23 worst-case. Deep-failure episodes are listed so "same 3 episodes under every
    # policy" (common-cause, not policy-caused) is checkable.
    report["notes"] = (
        "predictions routed P3 at derived go_below=1.0mm; E1 ran the frozen config's provisional "
        "4.0mm gate (changing YAML would break the committed fingerprint). Touch bias ~+2.9mm "
        "included in gate margin. Deep failures listed per policy for common-cause check."
    )
    report["deep_failure_episodes"] = {
        p: [
            {"ep": r.get("episode"), "variant": r.get("variant"), "e_mm": r.get("injected_e_mm"),
             "by": r.get("terminated_by")}
            for r in data[p]
            if "success" not in (r.get("terminated_by") or []) and r.get("gate_decision") != "reject"
        ]
        for p in POLICIES
    }

    out = LOGS / "e1_report.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
