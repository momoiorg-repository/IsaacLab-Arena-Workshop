# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure the loader's residual depth offset and write it back to the config as a calibration.

spec §9 forbids hand-tuning constants; the standing amendment allows a MEASUREMENT ROUTINE to write
them, provided the write carries metadata saying where the number came from. This is that routine
for one constant: ``overshoot_comp``, the per-variant depth offset that remains after the predictive
trigger has cancelled what it can.

The distinction that makes this worth doing rather than nudging a number by hand: a capability study
separates a window missed because the process is OFF CENTRE from one missed because it is TOO WIDE.
Measured on the loader, W-B sat at Cp 1.77 / Cpk 0.70 -- its spread was comfortably inside the
window and only its centre was out. A centre offset is exactly what a calibration constant is for;
spread is not, and no amount of calibration will move it.

Two offsets get folded into one number here, deliberately, because the loader cannot tell them apart
from proprioception alone:

* control lag the trigger's extrapolation does not fully cancel (~0.2-0.9 mm, all variants)
* a geometric shortfall where the seat cannot be reached while the part is held. W-C is the case:
  its Ø45 flange seats at 72 mm, but the fingertip clears the chuck face only for a grip station
  beyond 80.9 mm on an 80 mm part, so it stops ~5.4 mm short every time. That is a constant, not a
  drift, and the measurement finds it without needing to know which of the two it is.

Usage -- run the loader, then calibrate from what it did:

    /isaac-sim/python.sh scripts/bdash/check_chuck_teacher.py --num_envs 1 --seed 0 \\
        --episodes 12 --max_steps 900 --jsonl logs/bdash/calib.jsonl \\
        bdash_chuck_load --task full --variants all
    /isaac-sim/python.sh scripts/bdash/calibrate_overshoot.py logs/bdash/calib.jsonl --write
"""

import argparse
import collections
import json
import os
import statistics as st
import time

MIN_SAMPLES = 3


def _measure(rows):
    """Per-variant mean protrusion error, over episodes the loader actually completed."""
    by_variant = collections.defaultdict(list)
    for record in rows:
        if record.get("min_insert") and record.get("protrusion_err") is not None:
            by_variant[record["variant"]].append(record["protrusion_err"])
    return by_variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="records from check_chuck_teacher.py --task full")
    parser.add_argument("--config", default="configs/bdash/chuck_load/controllers.yaml")
    parser.add_argument("--write", action="store_true", help="write the block back into the config")
    parser.add_argument("--allow_stacked", action="store_true", help="calibrate even with one already applied")
    args = parser.parse_args()

    # REFUSE to calibrate on data taken with a calibration already applied. A compensation
    # measured through an existing compensation is not the process's offset, it is the residual --
    # writing it back applies the correction twice. Measured when this guard was missing: W-A's
    # comp went to +2.74 mm on top of a live -0.86 mm and the variant came out at +3.29 mm with
    # 0/14 in window, while W-B and W-C (already near centre) looked fine and hid it.
    with open(args.config) as handle:
        live = handle.read()
    if "overshoot_comp:" in live and not args.allow_stacked:
        print(
            "REFUSING: a calibration is already active in "
            f"{args.config}.\n"
            "  A run made with it applied measures the RESIDUAL, not the offset; writing that back\n"
            "  compensates twice. Remove the AUTO-WRITTEN block, re-run the loader, then calibrate.\n"
            "  (--allow_stacked overrides, for deliberately iterating a residual.)"
        )
        print("CALIB_REFUSED")
        return

    with open(args.jsonl) as handle:
        rows = [json.loads(line) for line in handle]
    by_variant = _measure(rows)
    if not by_variant:
        print("CALIB_NO_SAMPLE")
        return

    print(f"{'variant':8} {'n':>3} {'mean err':>10} {'sigma':>8} {'comp':>10}")
    comp = {}
    for variant in sorted(by_variant):
        errs = by_variant[variant]
        mean = st.mean(errs)
        sigma = st.stdev(errs) if len(errs) > 1 else 0.0
        # protrusion error = measured - nominal. A NEGATIVE error means the part sits deeper than
        # nominal, so the commanded depth must come DOWN by that much: comp = +mean. Signs are the
        # single easiest thing to get backwards here, so the arithmetic is written out.
        comp[variant] = round(mean, 5)
        flag = "" if len(errs) >= MIN_SAMPLES else "  (UNDER-SAMPLED)"
        print(
            f"{variant:8} {len(errs):3d} {mean * 1000:9.3f}mm {sigma * 1000:7.3f}mm {comp[variant] * 1000:9.3f}mm{flag}"
        )

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    source = os.path.basename(args.jsonl)
    block = ["", "# AUTO-WRITTEN by scripts/bdash/calibrate_overshoot.py -- do not hand-edit (spec §9)."]
    block += [
        f"#   source: {source}",
        "#   samples: " + ", ".join(f"{v}={len(by_variant[v])}" for v in sorted(by_variant)),
        f"#   measured: {stamp}",
        "# Added to the commanded insertion depth, per variant, in metres. Folds control lag and",
        "# any geometric shortfall the loader cannot reach into one measured constant.",
        "overshoot_comp:",
    ]
    block += [f"  {v}: {comp[v]:.5f}" for v in sorted(comp)]
    text = "\n".join(block)
    print("\n" + text)

    if not args.write:
        print("\nCALIB_DRY_RUN (pass --write to apply)")
        return

    with open(args.config) as handle:
        config = handle.read()
    marker = "# AUTO-WRITTEN by scripts/bdash/calibrate_overshoot.py"
    if marker in config:
        head, _, tail = config.partition(marker)
        # drop the previous block: everything up to the next top-level key
        rest = tail.split("\n")
        keep = []
        for i, line in enumerate(rest):
            if i and line and not line.startswith((" ", "#")):
                keep = rest[i:]
                break
        config = head.rstrip("\n") + "\n" + text + "\n\n" + "\n".join(keep)
    else:
        config = config.rstrip("\n") + "\n" + text + "\n"
    with open(args.config, "w") as handle:
        handle.write(config)
    print(f"\nCALIB_WRITTEN {args.config}")


if __name__ == "__main__":
    main()
