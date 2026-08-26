# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Post-recording audit for chuck full-cycle demo HDF5s -- run it after EVERY recording.

The recorder's cut (`chuck_closed`) is the COMMITMENT signal, not a verification: the jaws close
when the teacher believes it is done, so a part that slipped in the fingers and ended lying on top
of the chuck still exports as a "success". Measured on chuck_full250: 3 of 250 exported demos were
exactly that (found first by human eyeballing; this audit reproduces the same 3 from state data,
and no others). Criteria per demo, on the final frame of `states`:

  seated  =  part axis within ~18 deg of vertical (axis_z > 0.95)
          AND part bottom >= 20 mm below the chuck face
          AND part axis within 10 mm of the bore axis

Optionally rebuilds a cleaned HDF5 (+ sidecar with the dropped demos' language nulled, which keeps
the converter's recording-order sentence list aligned with the densified demo numbering).

Numbering note: LeRobot episode numbers follow h5py's ALPHABETICAL key order (demo_0, demo_1,
demo_10, demo_100, ...), not recording order -- `--map` prints the translation.
"""

from __future__ import annotations

import argparse
import h5py
import json
import numpy as np
import pathlib

VKEY = {"W-A": "bdash_workpiece_wa_0", "W-B": "bdash_workpiece_wb_0", "W-C": "bdash_workpiece_wc_0"}
HEIGHT = {"W-A": 0.09, "W-B": 0.10, "W-C": 0.08}


def axis_z(q: np.ndarray) -> np.ndarray:
    """World-z component of the part's local +z axis, from a wxyz quaternion."""
    x, y = q[..., 1], q[..., 2]
    return 1 - 2 * (x * x + y * y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--out", help="write a cleaned copy here, dropping NG demos (with fixed sidecar)")
    ap.add_argument("--map", action="store_true", help="print recording-order -> lerobot-episode map")
    args = ap.parse_args()

    path = pathlib.Path(args.hdf5)
    sidecar = path.with_name(path.stem + "_attempts.jsonl")
    rows = [json.loads(line) for line in sidecar.open()]
    exported = [r for r in rows if r.get("exported")]

    bad = []
    with h5py.File(path) as f:
        n = len(f["data"])
        for i, meta in enumerate(exported):
            st = f[f"data/demo_{i}/states/rigid_object"]
            wp = st[VKEY[meta["variant"]]]["root_pose"]
            chuck = st["bdash_chuck_body"]["root_pose"][-1]
            face = chuck[2] + 0.08
            p = wp[-1]
            az = float(axis_z(p[3:7]))
            depth = (face - (p[2] - HEIGHT[meta["variant"]] / 2 * az)) * 1000
            xy = float(np.hypot(p[0] - chuck[0], p[1] - chuck[1]) * 1000)
            ok = az > 0.95 and depth > 20 and xy < 10
            if not ok:
                bad.append(i)
                print(f"NG demo_{i} {meta['variant']} axis_z={az:.3f} depth={depth:.1f}mm xy={xy:.1f}mm")
        if args.map:
            keys = sorted(f"demo_{i}" for i in range(n))
            for ep, k in enumerate(keys):
                if int(k.split("_")[1]) in bad:
                    print(f"lerobot ep{ep} = {k}  <- NG")
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
        # Null the dropped rows' language IN PLACE OF removing them: the converter builds its
        # sentence list from truthy languages in file order, so this keeps sentence k = demo_k.
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
