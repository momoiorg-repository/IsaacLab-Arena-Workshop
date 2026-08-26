# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Post-recording audit for LIFT-cut chuck demos (data spec v2 §3-2).

The lift slice's quality bar differs from the seating audit: an episode is GOOD when, at its final
frame, the TARGET is risen and upright in the fingers -- and NO OTHER part got knocked over on the
way in. The "no toppled neighbours" clause is new with v2: the lift247 post-mortem showed the model
imitating contact-heavy approaches, and a demo that bulldozes a neighbour while succeeding teaches
exactly that.

Criteria (final frame of `states`):
  target:     risen >= 50 mm above its episode-start height, axis within ~26 deg of vertical
              (axis_z > 0.9), finger width inside the holding band (15-65 mm)
  neighbours: every non-target part axis_z > 0.7 (not toppled)

Run after EVERY recording batch; `--out` writes a cleaned copy (dense renumber + sidecar fix).
"""

from __future__ import annotations

import argparse
import h5py
import json
import numpy as np
import pathlib

VKEY = {"W-A": "bdash_workpiece_wa_0", "W-B": "bdash_workpiece_wb_0", "W-C": "bdash_workpiece_wc_0"}


def axis_z(q: np.ndarray) -> np.ndarray:
    x, y = q[..., 1], q[..., 2]
    return 1 - 2 * (x * x + y * y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--out", help="write a cleaned copy dropping NG demos")
    args = ap.parse_args()
    path = pathlib.Path(args.hdf5)
    rows = [json.loads(line) for line in path.with_name(path.stem + "_attempts.jsonl").open()]
    exported = [r for r in rows if r.get("exported")]

    bad = []
    with h5py.File(path) as f:
        for i, meta in enumerate(exported):
            st = f[f"data/demo_{i}/states"]
            q = st["articulation/robot/joint_position"]
            width = float(q[-1, -2] + q[-1, -1])
            tkey = VKEY[meta["variant"]]
            reasons = []
            for name, key in VKEY.items():
                wp = st[f"rigid_object/{key}/root_pose"]
                az = float(axis_z(wp[-1, 3:7]))
                if key == tkey:
                    rise = float(wp[-1, 2] - wp[0, 2])
                    if rise < 0.050:
                        reasons.append(f"rise={rise * 1000:.0f}mm")
                    if az < 0.9:
                        reasons.append(f"target_axis={az:.2f}")
                    if not (0.015 < width < 0.065):
                        reasons.append(f"width={width * 1000:.0f}mm")
                elif az < 0.7:
                    reasons.append(f"toppled:{name}(az={az:.2f})")
            if reasons:
                bad.append(i)
                print(f"NG demo_{i} {meta['variant']}: {', '.join(reasons)}")
    print(f"AUDIT: {len(bad)} NG / {len(exported)} demos")

    if args.out and bad:
        badset = set(bad)
        out = pathlib.Path(args.out)
        with h5py.File(path) as src, h5py.File(out, "w") as dst:
            grp = dst.create_group("data")
            for a in src["data"].attrs:
                grp.attrs[a] = src["data"].attrs[a]
            j = 0
            for i in range(len(exported)):
                if i in badset:
                    continue
                src.copy(f"data/demo_{i}", grp, name=f"demo_{j}")
                j += 1
        k = -1
        for r in rows:
            if r.get("exported"):
                k += 1
                if k in badset:
                    r["exported"] = False
                    r["language"] = None
                    r["dropped_by_audit"] = True
        out.with_name(out.stem + "_attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"WROTE {out} ({j} demos) + sidecar")


if __name__ == "__main__":
    main()
