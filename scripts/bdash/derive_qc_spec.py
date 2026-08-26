# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Derive the QC acceptance window from what the scripted loader actually achieves.

spec §2.3 states a good part as "seat angle <= 0.5 deg, protrusion within the family target
+-1.5 mm". Those numbers are marked provisional in ``assets.yaml`` ("per-variant acceptance (filled
from the R(g) sweep, §6)"), and §9 forbids hand-tuning them: a tolerance may only come from measured
data. This script is that derivation for the LOADER's own contribution, ahead of the full R(g) sweep.

**Process-capability framing.** Treat the loader as a process and the QC window as its specification.
For a window of half-width ``T`` around nominal, a centred process with spread ``sigma`` has

    Cp  = T / (3 sigma)              capability if it were centred
    Cpk = (T - |mu|) / (3 sigma)     capability as it actually sits, ``mu`` = mean offset

Read the other way, the window a given capability demands is ``T = |mu| + 3 sigma * Cpk``. Quoting
the window this way says what the process can hold rather than what anyone hoped for, and it
separates the two failure modes a single pass/fail rate hides: a window missed because the process
is OFF CENTRE (fix the command) and one missed because it is TOO WIDE (fix the mechanism).

Sample selection matters and is deliberate: **only episodes the loader completed** feed the fit. A
dropped or jammed part is not a capability datum, it is a different failure, and folding it in would
inflate sigma with events the window was never meant to cover.

    /isaac-sim/python.sh scripts/bdash/derive_qc_spec.py logs/bdash/<run>.jsonl
"""

import argparse
import collections
import json
import math
import statistics as st

CPK_TARGET = 1.33  # the usual "capable" bar: ~63 ppm one-sided if centred and normal


def _fit(values):
    """(mean, sample sigma) for a list, or (mean, 0) when there is only one point."""
    mean = st.mean(values)
    sigma = st.stdev(values) if len(values) > 1 else 0.0
    return mean, sigma


def _capability(values, half_width):
    mu, sigma = _fit(values)
    if sigma <= 0.0:
        return mu, sigma, float("inf"), float("inf")
    cp = half_width / (3.0 * sigma)
    cpk = (half_width - abs(mu)) / (3.0 * sigma)
    return mu, sigma, cp, cpk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="per-episode records from check_chuck_teacher.py --task full")
    parser.add_argument("--cpk", type=float, default=CPK_TARGET)
    args = parser.parse_args()

    with open(args.jsonl) as handle:
        rows = [json.loads(line) for line in handle]
    loaded = [r for r in rows if r.get("min_insert") and r.get("protrusion_err") is not None]
    print(f"records {len(rows)}   loaded (capability sample) {len(loaded)}")
    if not loaded:
        print("QC_SPEC_NO_SAMPLE")
        return

    by_variant = collections.defaultdict(list)
    for r in loaded:
        by_variant[r["variant"]].append(r)

    print("\n=== protrusion error (mm), against the current window ===")
    print(f"{'variant':8} {'n':>3} {'mean':>8} {'sigma':>8} {'now':>7} {'Cp':>6} {'Cpk':>6} {'T@Cpk':>8}")
    derived = {}
    for variant in sorted(by_variant):
        sample = by_variant[variant]
        errs = [r["protrusion_err"] * 1000.0 for r in sample]
        now = sample[0]["protrusion_tol"] * 1000.0
        mu, sigma, cp, cpk = _capability(errs, now)
        need = abs(mu) + 3.0 * sigma * args.cpk
        derived[variant] = need
        print(f"{variant:8} {len(errs):3d} {mu:8.3f} {sigma:8.3f} {now:7.2f} {cp:6.2f} {cpk:6.2f} {need:8.3f}")

    angles = [r["qc_angle_deg"] for r in loaded if r.get("qc_angle_deg") is not None]
    if angles:
        mu, sigma = _fit(angles)
        p95 = sorted(angles)[max(0, int(0.95 * len(angles)) - 1)]
        need = mu + 3.0 * sigma * args.cpk
        print(f"\n=== seat angle (deg) ===  n={len(angles)} mean={mu:.3f} sigma={sigma:.3f} p95={p95:.3f}")
        print(f"  spec §2.3 bar 0.500   window at Cpk {args.cpk}: {need:.3f}")

    print("\n=== what to write into assets.yaml ===")
    for variant, need in derived.items():
        print(f"  {variant}: protrusion_mm: {math.ceil(need * 10) / 10:.1f}   # derived, Cpk>={args.cpk}")
    print("QC_SPEC_DERIVED")


if __name__ == "__main__":
    main()
