# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dump one recorded demo's camera views to an MP4 a human can actually watch.

This exists because the automated inspection cannot see nonsense. The 246-demo pilot passed every
programmatic check -- no frozen frames, three well-shaped views, finite channels, matching labels --
while the wrist camera spent the end of every episode UNDER the bench, because the horizontal tool
pose swung its 110 mm forward offset below the surface. Frames that change and have the right shape
can still show the underside of a table. The only check that catches that is a pair of eyes, which
is why "watch one video" was on the inspection list -- and why skipping it cost the whole pilot.

Renders nothing; reads the HDF5 and pipes raw frames to ffmpeg. Views are tiled side by side so one
file shows everything the policy would see.
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import pathlib
import subprocess

CAMERAS = ("wrist_cam_rgb", "left_cam_rgb", "right_cam_rgb")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--demo", default=None, help="demo key (default: first)")
    ap.add_argument("--out", default=None, help="output mp4 (default: beside the hdf5)")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    path = pathlib.Path(args.hdf5)
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))
        key = args.demo or demos[0]
        obs = f["data"][key]["obs"]
        views = [obs[c][:] for c in CAMERAS if c in obs]
        tile = np.concatenate(views, axis=2)  # (N, H, W*views, 3)

    out = pathlib.Path(args.out or path.with_name(f"{path.stem}_{key}.mp4"))
    n, h, w, _ = tile.shape
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    proc.stdin.write(np.ascontiguousarray(tile, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"{out}  ({key}, {n} frames, {len(views)} views)")


if __name__ == "__main__":
    main()
