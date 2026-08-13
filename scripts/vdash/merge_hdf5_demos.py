# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Merge several Isaac Lab demo HDF5s into one (renumber demos, fix the ``data`` attrs).

The VLA recorder writes one ``data/demo_{i}`` group per successful episode plus
``data.attrs={env_args, total}`` (``total`` = summed step count). The recovery dataset is recorded as
several slices (clean / wide-start / recovery), so concatenate them for the single-file LeRobot
converter:

    /isaac-sim/python.sh scripts/vdash/merge_hdf5_demos.py \\
        --inputs datasets/vdash/_v5_parts/v5_{clean,wide,recovery}.hdf5 \\
        --output datasets/vdash/vla_pick_handoff_v5_recovery.hdf5
"""

import argparse
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    total = 0
    env_args = None
    idx = 0
    with h5py.File(args.output, "w") as out:
        g = out.create_group("data")
        for path in args.inputs:
            with h5py.File(path, "r") as f:
                d = f["data"]
                if env_args is None:
                    env_args = d.attrs.get("env_args")
                names = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
                for nm in names:
                    f.copy(d[nm], g, name=f"demo_{idx}")
                    total += int(g[f"demo_{idx}"]["obs"]["joint_pos"].shape[0])
                    idx += 1
            print(f"[merge] {path}: cumulative {idx} demos", flush=True)
        if env_args is not None:
            g.attrs["env_args"] = env_args
        g.attrs["total"] = total
    print(f"[merge] WROTE {args.output}: {idx} demos, total_steps={total}", flush=True)
    print("MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
