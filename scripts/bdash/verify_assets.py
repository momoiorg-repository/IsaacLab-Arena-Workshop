# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the generated B-DASH chuck-loading meshes against the config that produced them.

Runs on the STLs, so it needs no Simulation App — only trimesh. Checks the things that
silently break a scene later:

  * overall extents match the config (a wrong section order or chamfer shows up here)
  * meshes are watertight, which SDF collision requires
  * the chuck bore actually passes through, and is sized so W-C's flange seats while W-B's
    step still clears
  * jaws close to the intended Ø25 / Ø32 radii

Usage::

    /isaac-sim/python.sh scripts/bdash/verify_assets.py
    /isaac-sim/python.sh scripts/bdash/verify_assets.py --assets assets/bdash --config configs/bdash/assets.yaml
"""

import argparse
import numpy as np
import os
import sys
import trimesh
import yaml

TOL = 5e-4  # 0.5 mm — looser than the chamfer, tighter than any feature we care about


def _fail(msg: str, failures: list[str]) -> None:
    print(f"    FAIL  {msg}")
    failures.append(msg)


def _check_extent(name: str, got: float, want: float, axis: str, failures: list[str]) -> None:
    ok = abs(got - want) < TOL
    status = "ok  " if ok else "FAIL"
    print(f"    {status}  {axis} extent {got:.4f} (expected {want:.4f})")
    if not ok:
        failures.append(f"{name}: {axis} extent {got:.4f} != {want:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify B-DASH generated assets.")
    ap.add_argument("--assets", default="assets/bdash")
    ap.add_argument("--config", default="configs/bdash/assets.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    failures: list[str] = []

    # ---------------- workpieces ----------------
    for wid, spec in cfg["workpieces"].items():
        name = f"workpiece_{wid.replace('-', '').lower()}"
        path = os.path.join(args.assets, f"{name}.stl")
        if not os.path.isfile(path):
            _fail(f"{name}: missing {path}", failures)
            continue
        mesh = trimesh.load(path)
        ext = mesh.bounds[1] - mesh.bounds[0]
        total_len = sum(length for _, length in spec["sections"])
        max_dia = max(dia for dia, _ in spec["sections"])
        print(f"{name}  ({wid})  tris={len(mesh.faces)}  vol={mesh.volume * 1e6:.2f} cm^3")
        _check_extent(name, ext[2], total_len, "Z", failures)
        _check_extent(name, max(ext[0], ext[1]), max_dia, "max-dia", failures)
        if not mesh.is_watertight:
            _fail(f"{name}: not watertight", failures)
        if mesh.bounds[0][2] > TOL:
            _fail(f"{name}: leading end is at z={mesh.bounds[0][2]:.4f}, expected 0", failures)
        # the leading section must be the one the jaws grip
        lead_dia = spec["sections"][0][0]
        if abs(lead_dia - 0.025) > TOL:
            _fail(f"{name}: leading diameter {lead_dia} is not the Ø25 the jaws close on", failures)

    # ---------------- chuck body ----------------
    c = cfg["chuck"]
    path = os.path.join(args.assets, "chuck_body.stl")
    if os.path.isfile(path):
        mesh = trimesh.load(path)
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(f"chuck_body  tris={len(mesh.faces)}  vol={mesh.volume * 1e6:.2f} cm^3")
        _check_extent("chuck_body", ext[2], c["body_height"], "Z", failures)
        _check_extent("chuck_body", max(ext[0], ext[1]), c["body_diameter"], "dia", failures)
        if not mesh.is_watertight:
            _fail("chuck_body: not watertight (SDF collision needs it)", failures)
        # bore must pass all the way through: sample the axis
        axis_pts = np.c_[np.zeros(9), np.zeros(9), np.linspace(0.002, c["body_height"] - 0.002, 9)]
        if mesh.contains(axis_pts).any():
            _fail("chuck_body: bore does not pass through (solid material on the axis)", failures)
        else:
            print("    ok    bore is open along the full axis")
    else:
        _fail("chuck_body: missing", failures)

    # seating / clearance relationships that the whole task depends on
    bore_d = c["bore_diameter"]
    flange_d = max(dia for dia, _ in cfg["workpieces"]["W-C"]["sections"])
    step_d = max(dia for dia, _ in cfg["workpieces"]["W-B"]["sections"])
    print("chuck fit relationships")
    if flange_d > bore_d:
        print(f"    ok    W-C flange Ø{flange_d} seats on the Ø{bore_d} bore face")
    else:
        _fail(f"W-C flange Ø{flange_d} falls into the Ø{bore_d} bore — it cannot seat", failures)
    if step_d < bore_d:
        print(f"    ok    W-B step Ø{step_d} clears the Ø{bore_d} bore")
    else:
        _fail(f"W-B step Ø{step_d} cannot enter the Ø{bore_d} bore", failures)

    # ---------------- jaw ----------------
    j = cfg["chuck_jaw"]
    path = os.path.join(args.assets, "chuck_jaw.stl")
    if os.path.isfile(path):
        mesh = trimesh.load(path)
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(f"chuck_jaw  tris={len(mesh.faces)}  vol={mesh.volume * 1e6:.2f} cm^3")
        _check_extent("chuck_jaw", ext[2], j["height"], "Z", failures)
        _check_extent("chuck_jaw", ext[1], j["width"], "Y", failures)
        _check_extent("chuck_jaw", ext[0], j["radial_length"], "X-radial", failures)
        if not mesh.is_watertight:
            _fail("chuck_jaw: not watertight", failures)
        # The stepped face: below the step the jaw reaches x=0 (grips Ø25); above it the face is
        # set back by radius_high - radius_low (grips Ø32). Checked on raw vertices rather than
        # mesh.slice_plane, which pulls in shapely and is not available in the Isaac Sim image.
        setback = j["radius_high"] - j["radius_low"]
        step_h = j["step_height"]
        verts = np.asarray(mesh.vertices)
        below = verts[verts[:, 2] < step_h - TOL]
        above = verts[verts[:, 2] > step_h + TOL]
        low_x = float(below[:, 0].min()) if len(below) else None
        high_x = float(above[:, 0].min()) if len(above) else None
        if low_x is None or high_x is None:
            _fail("chuck_jaw: no vertices on one side of the step", failures)
        elif abs(low_x) < TOL and abs(high_x - setback) < TOL:
            print(f"    ok    stepped face: low reaches x=0 (Ø25), high set back {setback:.4f} (Ø32)")
        else:
            _fail(f"chuck_jaw: step wrong (low_x={low_x:.4f}, high_x={high_x:.4f}, want 0 and {setback:.4f})", failures)
    else:
        _fail("chuck_jaw: missing", failures)

    # ---------------- simple fixtures ----------------
    for name, dims in (
        ("tray", cfg["tray"]["outer"]),
        ("touchoff_block", cfg["touchoff_block"]["size"]),
        ("vblock", cfg["vblock"]["size"]),
    ):
        path = os.path.join(args.assets, f"{name}.stl")
        if not os.path.isfile(path):
            _fail(f"{name}: missing", failures)
            continue
        mesh = trimesh.load(path)
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(f"{name}  tris={len(mesh.faces)}  vol={mesh.volume * 1e6:.2f} cm^3")
        for axis, got, want in zip("XYZ", ext, dims):
            _check_extent(name, got, want, axis, failures)
        if not mesh.is_watertight:
            _fail(f"{name}: not watertight", failures)

    # the V-groove must be able to hold the Ø25 shaft without bottoming out
    v = cfg["vblock"]
    half = np.deg2rad(v["groove_angle_deg"]) / 2.0
    shaft_r = 0.0125
    seat_drop = shaft_r / np.sin(half)  # centre height above the apex for a cylinder in a V
    if seat_drop - shaft_r < v["groove_depth"]:
        print(f"    ok    Ø25 shaft rests in the V (centre {seat_drop * 1000:.1f} mm above apex)")
    else:
        _fail(f"vblock: groove too shallow for Ø25 (needs > {(seat_drop - shaft_r) * 1000:.1f} mm)", failures)

    print()
    if failures:
        print(f"FAILED — {len(failures)} problem(s):")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
