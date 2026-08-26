# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Derive a lifted-cut slice from a full-cycle chuck demo HDF5 -- no re-recording.

The teacher is deterministic, so truncating a full demo at the moment the `lifted` predicate
would have fired is byte-equivalent (same trajectories) to having recorded with `--until lifted`.
Cut condition mirrors bdash_chuck_predicates.lifted: target part risen ~60 mm above its start
height while the fingers hold it (width inside the grasp band). The sidecar's per-episode sentence
gets its verb phrase swapped ("load it into the chuck" -> "lift it clear") so the label describes
what the truncated demo actually does -- the v5 lesson about mislabeled cut points, applied.
"""

from __future__ import annotations

import argparse
import h5py
import json
import numpy as np
import pathlib

VKEY = {"W-A": "bdash_workpiece_wa_0", "W-B": "bdash_workpiece_wb_0", "W-C": "bdash_workpiece_wc_0"}


def cut_frame(demo: h5py.Group, variant: str) -> int | None:
    wp_z = demo[f"states/rigid_object/{VKEY[variant]}/root_pose"][:, 2]
    q = demo["states/articulation/robot/joint_position"]
    width = q[:, -2] + q[:, -1]
    risen = wp_z > wp_z[0] + 0.060
    # Band accepts every variant's grasp: W-A/W-B hold the 25 mm shaft (width ~25 mm), W-C holds
    # its 45 mm flange (measured 48.2 mm) -- and still excludes the fully-open hand (80 mm).
    held = (width > 0.015) & (width < 0.065)
    idx = np.nonzero(risen & held)[0]
    return int(idx[0]) + 1 if idx.size else None  # noqa: E226


def truncate(src: h5py.Group, dst: h5py.Group, t: int) -> None:
    for k, item in src.items():
        if isinstance(item, h5py.Group):
            g = dst.create_group(k)
            for a in item.attrs:
                g.attrs[a] = item.attrs[a]
            truncate(item, g, t)
        else:
            dst.create_dataset(k, data=item[:t])
    for a in src.attrs:
        dst.attrs[a] = src.attrs[a]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = pathlib.Path(args.hdf5)
    out = pathlib.Path(args.out)
    rows = [json.loads(line) for line in path.with_name(path.stem + "_attempts.jsonl").open()]
    exported = [r for r in rows if r.get("exported")]

    lens = []
    with h5py.File(path) as f, h5py.File(out, "w") as g:
        data = g.create_group("data")
        for a in f["data"].attrs:
            data.attrs[a] = f["data"].attrs[a]
        for i, meta in enumerate(exported):
            src = f[f"data/demo_{i}"]
            t = cut_frame(src, meta["variant"])
            if t is None:
                raise SystemExit(f"demo_{i}: lifted condition never met -- refusing to guess")
            dst = data.create_group(f"demo_{i}")
            truncate(src, dst, t)
            if "num_samples" in dst.attrs:
                dst.attrs["num_samples"] = t
            lens.append(t)
    for r in rows:
        if r.get("language"):
            r["language"] = r["language"].replace("load it into the chuck", "lift it clear")
    out.with_name(out.stem + "_attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"WROTE {out}: {len(lens)} demos, cut len min/med/max = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")


if __name__ == "__main__":
    main()
