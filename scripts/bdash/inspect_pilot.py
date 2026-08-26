# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect a recorded pilot before committing GPU-days to the main run.

The four checks are the ones a bad dataset actually fails, and each exists because something like
it has already gone wrong here:

* **frozen frames** -- the cameras only refresh on a render-coupled physics pump. Without it every
  frame in an episode is byte-identical and the dataset looks perfectly well-formed. Measured as the
  per-episode minimum absolute frame difference: a zero means at least one pair of consecutive
  frames was identical.
* **camera §2.1** -- the three views the policy consumes must all be present and non-degenerate. A
  view that is missing, constant, or a different size is a view the model will learn to ignore.
* **continuous-value channels** -- `axis_cond` and `appearance` carry the conditioning. All-zero or
  non-finite means the wiring broke: `appearance` shipped five zeros for a while because the MDL
  path has no scalar colour, and nothing downstream noticed.
* **the instruction** -- now per episode. Checked against the material the target was ACTUALLY drawn
  with, because the sentence is the label and a label that names the wrong part is worse than a
  constant one.

Reads the HDF5 and the attempts sidecar; renders nothing and needs no GPU.
"""

from __future__ import annotations

import argparse
import collections
import h5py
import json
import numpy as np
import pathlib

CAMERAS = ("wrist_cam_rgb", "left_cam_rgb", "right_cam_rgb")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--frame_sample", type=int, default=24, help="frames per episode for the freeze check")
    args = ap.parse_args()

    path = pathlib.Path(args.hdf5)
    side = pathlib.Path(args.sidecar or path.with_name(path.stem + "_attempts.jsonl"))
    fails: list[str] = []

    with h5py.File(path, "r") as f:
        demos = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))
        lengths = [f["data"][k]["actions"].shape[0] for k in demos]
        print(f"demos={len(demos)}  frames: mean={np.mean(lengths):.0f} min={min(lengths)} max={max(lengths)}")

        # --- cameras present, correctly shaped, and not constant -------------------------------
        obs0 = f["data"][demos[0]]["obs"]
        for cam in CAMERAS:
            if cam not in obs0:
                fails.append(f"camera {cam} missing")
                continue
            shape = obs0[cam].shape
            print(f"  {cam:15s} {shape}")
            if len(shape) != 4 or shape[1] < 64 or shape[2] < 64:
                fails.append(f"camera {cam} has shape {shape}")

        # --- frozen frames ---------------------------------------------------------------------
        frozen = []
        for k in demos:
            o = f["data"][k]["obs"]
            n = o[CAMERAS[0]].shape[0]
            idx = np.unique(np.linspace(0, n - 1, min(args.frame_sample, n)).astype(int))
            for cam in CAMERAS:
                if cam not in o:
                    continue
                frames = o[cam][idx].astype(np.float32)
                d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))
                if d.size and float(d.min()) == 0.0:
                    frozen.append((k, cam))
        print(f"  frozen frame pairs: {len(frozen)} (over {len(demos)} demos x {len(CAMERAS)} views)")
        if frozen:
            fails.append(f"frozen frames in {len(frozen)} episode/view pairs, e.g. {frozen[:3]}")

        # --- continuous channels ---------------------------------------------------------------
        for key in ("axis_cond", "appearance"):
            if key not in obs0:
                fails.append(f"channel {key} missing")
                continue
            vals = np.concatenate([f["data"][k]["obs"][key][:] for k in demos[:50]])
            finite = np.isfinite(vals).all()
            allzero = not np.any(vals)
            print(f"  {key:15s} finite={finite} all_zero={allzero} range=[{vals.min():.3f}, {vals.max():.3f}]")
            if not finite or allzero:
                fails.append(f"channel {key}: finite={finite} all_zero={allzero}")

        for key in ("joint_pos", "eef_pos"):
            if key in obs0:
                vals = np.concatenate([f["data"][k]["obs"][key][:] for k in demos[:50]])
                if not np.isfinite(vals).all():
                    fails.append(f"channel {key} has non-finite values")

    # --- instruction vs the material actually drawn ---------------------------------------------
    if side.exists():
        rows = [json.loads(line) for line in side.open()]
        ok = [r for r in rows if r.get("language")]
        mism = [r for r in ok if r.get("target_material") != (r.get("appearance") or [{}])[r["target"]].get("name")]
        dup = [
            r for r in ok if len({a.get("name") for a in (r.get("appearance") or [])}) != len(r.get("appearance") or [])
        ]
        phrases = collections.Counter(r["language"] for r in ok)
        print(f"\ninstructions: {len(ok)} labelled, {len(phrases)} distinct")
        for text, n in phrases.most_common():
            print(f"  {n:4d}  {text}")
        print(f"  instruction/material mismatches: {len(mism)}")
        print(f"  episodes with a duplicated material: {len(dup)}")
        if mism:
            fails.append(f"{len(mism)} demos labelled with the wrong material")
        if dup:
            fails.append(f"{len(dup)} demos have two parts sharing a material -- the instruction is ambiguous")
        if len(phrases) < 2:
            fails.append("every demo carries the same sentence; the instruction selects nothing")
    else:
        fails.append(f"sidecar {side} not found -- cannot check the labels")

    print()
    if fails:
        print("INSPECT_FAIL")
        for x in fails:
            print(f"  - {x}")
        raise SystemExit(1)
    print("INSPECT_OK")


if __name__ == "__main__":
    main()
