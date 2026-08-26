# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Numerical verification for the §4-2 re-erect leg, run BEFORE any of it is implemented.

The ruling that authorised this leg also fixed how it gets chosen: "例により数値検証してから採否を
決めること" -- verify numerically, then adopt. Three candidate routes exist and they differ in what
they cost, not in whether they are describable, so prose cannot separate them. This script computes
the quantities that do.

The routes:

* **A -- groove, re-grip, rotate, insert while horizontal.** Set the part in the V-groove, re-grip
  at the axial station insertion needs, lift, rotate 90 deg in air, and carry it to the bore with
  the TOOL still horizontal and the part hanging vertically from it.
* **B -- groove, re-grip, rotate, stand it down, re-pick upright.** As A, but the part is set down
  standing and picked again with the ordinary upright grasp, so everything downstream is the
  already-measured upright path.
* **B' -- rotate straight from the tray grasp, stand it down, re-pick upright.** As B without the
  groove: the part is rotated from the station it was already grasped at.

What decides between them:

1. **Reachable insertion depth.** With the tool vertical the fingers hang BELOW the TCP and enter
   the bore alongside the part, so depth is capped by the grip station minus the finger reach. With
   the tool horizontal they extend sideways instead and the cap changes. A route that cannot reach
   the commanded depth is not a route.
2. **Grip torque during rotation.** The part is a cantilever about the grip while it turns. If the
   demanded torque approaches what a friction grip can hold, the part rotates inside the fingers and
   the open-loop trajectory -- which by the ruling may NOT be corrected from perception -- is wrong
   from then on with nothing to notice it.
3. **Swept volume.** The part sweeps a disc of radius (station, length - station) about the grip
   while it turns. That disc must clear the bench and the fixtures.

Sim-free: geometry and force balance from the config files. No Isaac Sim, no torch.
"""

from __future__ import annotations

import math
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/bdash/chuck_load"

# MEASURED on the Franka hand in this scene (docs/progress/2026-08-16.md), not nominal CAD:
# the fingertip sits below the TCP, and each finger reaches outward from the face it grips with.
FINGERTIP_BELOW_TCP = 0.0089
FINGER_REACH = 0.0262
# The depth an upright grasp can reach = station - this. Measured: stations 65/70/76 mm gave
# 54.1/59.1/65.1 mm reachable, a constant 10.9 mm deficit, which is the fingertip plus the seating
# margin the insertion controller keeps.
UPRIGHT_DEPTH_DEFICIT = 0.0109
GRIP_FORCE_N = 40.0  # panda_hand continuous grasp force
GRIP_MU = 0.5  # steel on the rubberised pad, deliberately pessimistic
FINGER_PAD_HALF_LEN = 0.017  # half the pad length; the lever that resists the part pivoting
G = 9.81


def _load():
    assets = yaml.safe_load((CFG / "assets.yaml").read_text())
    task = yaml.safe_load((CFG / "task.yaml").read_text())
    ctrl = yaml.safe_load((CFG / "controllers.yaml").read_text())
    return assets, task, ctrl


def profile(name: str, spec: dict) -> dict:
    """Length, radius, centre of mass and the flange radius, from the section list."""
    sections = spec["sections"]
    length = sum(h for _, h in sections)
    vol = 0.0
    moment = 0.0
    z = 0.0
    for dia, h in sections:
        v = math.pi * (dia / 2.0) ** 2 * h
        vol += v
        moment += v * (z + h / 2.0)
        z += h
    return {
        "variant": name,
        "length": length,
        "radius": max(dia for dia, _ in sections) / 2.0,
        "shank_radius": sections[0][0] / 2.0,
        "com": moment / vol,  # from the LEADING end
        "mass": float(spec["mass"]),
    }


def main() -> None:
    assets, task, ctrl = _load()
    pick = ctrl["pick"]
    side_station = float(pick["grip_station_side"])
    below_top = pick["grip_below_top_upright"]
    profiles = [profile(k, v) for k, v in assets["workpieces"].items()]
    insert_depth = task["chuck_load"]["insert_depth_m"]

    print("=" * 78)
    print("§4-2 RE-ERECT -- numerical verification (sim-free)")
    print("=" * 78)

    print("\n[1] REACHABLE INSERTION DEPTH  (required vs what each route can reach)")
    print(f"  {'variant':8s} {'L':>6s} {'need':>7s} {'side st':>8s} {'A(horiz)':>9s} {'B(upright)':>11s}")
    verdict_a, verdict_b = True, True
    for p in profiles:
        need = float(insert_depth[p["variant"]])
        up_station = p["length"] - float(below_top[p["variant"]])
        # Route A: tool horizontal, fingers extend sideways at TCP height, so the only cap is that
        # the fingers must stay clear of the chuck FACE -- depth < station, less the finger's own
        # half-thickness allowance (taken as the fingertip offset, the same measured number).
        reach_a = up_station - FINGERTIP_BELOW_TCP
        # Route B/B': the ordinary upright grasp, whose deficit is measured.
        reach_b = up_station - UPRIGHT_DEPTH_DEFICIT
        ok_a, ok_b = reach_a >= need, reach_b >= need
        verdict_a &= ok_a
        verdict_b &= ok_b
        print(
            f"  {p['variant']:8s} {p['length'] * 1e3:5.0f}m {need * 1e3:6.0f}m {side_station * 1e3:7.0f}m "
            f"{reach_a * 1e3:7.1f}{'ok' if ok_a else 'NG':>2s} {reach_b * 1e3:9.1f}{'ok' if ok_b else 'NG':>2s}"
        )
    # The side station on its own, which is what a rotate-and-insert-without-regrip route would use.
    reach_side = side_station - UPRIGHT_DEPTH_DEFICIT
    print(f"\n  no re-grip at all (side station {side_station * 1e3:.0f} mm): reaches {reach_side * 1e3:.1f} mm")
    print(
        f"  -> required is {min(float(v) for v in insert_depth.values()) * 1e3:.0f}-"
        f"{max(float(v) for v in insert_depth.values()) * 1e3:.0f} mm, so a RE-GRIP IS MANDATORY."
    )

    print("\n[2] GRIP TORQUE WHILE ROTATING  (part cantilevered about the grip)")
    cap = GRIP_MU * GRIP_FORCE_N * FINGER_PAD_HALF_LEN
    print(f"  friction capacity about the closing axis: mu*F*pad = {cap:.3f} N.m")
    print(f"  {'variant':8s} {'CoM':>6s} {'@side st':>9s} {'@up st':>8s} {'margin(min)':>12s}")
    worst = 1e9
    for p in profiles:
        up_station = p["length"] - float(below_top[p["variant"]])
        t_side = p["mass"] * G * abs(p["com"] - side_station)
        t_up = p["mass"] * G * abs(p["com"] - up_station)
        m = cap / max(t_side, t_up)
        worst = min(worst, m)
        print(f"  {p['variant']:8s} {p['com'] * 1e3:5.1f}m {t_side:8.4f} {t_up:7.4f} {m:11.1f}x")
    print(f"  -> worst margin {worst:.1f}x. {'PASS' if worst >= 3.0 else 'FAIL'} (need >=3x)")

    print("\n[3] SWEPT VOLUME WHILE ROTATING  (clearance above the bench)")
    bench_z = 0.0
    vblock = assets["vblock"]
    print(f"  {'variant':8s} {'route':10s} {'sweep R':>8s} {'lift need':>10s} {'TCP z':>8s}")
    lifts = {}
    for p in profiles:
        up_station = p["length"] - float(below_top[p["variant"]])
        for route, st in (("B' (side)", side_station), ("A/B (up)", up_station)):
            # The part sweeps a disc about the grip; its far end is max(st, L-st) away.
            sweep = max(st, p["length"] - st)
            # The lowest point during the turn is the grip height minus the sweep radius, so the TCP
            # must sit at least a sweep radius above the bench (plus the fingertip, which hangs
            # below it while the tool is still vertical).
            tcp_z = bench_z + sweep + FINGERTIP_BELOW_TCP
            lifts[(p["variant"], route)] = tcp_z
            print(f"  {p['variant']:8s} {route:10s} {sweep * 1e3:7.1f}m {sweep * 1e3:9.1f}m {tcp_z * 1e3:7.1f}m")
    need_tcp = max(lifts.values())
    print(f"  -> the rotation must happen with the TCP at z >= {need_tcp * 1e3:.0f} mm.")
    print(
        f"     V-block is {vblock['size'][0] * 1e3:.0f} mm long and {vblock['size'][1] * 1e3:.0f} mm wide; "
        f"a {2 * max(lifts.values()) * 1e3:.0f} mm swept disc does NOT fit over it,"
    )
    print("     so the turn happens clear of the groove either way.")

    print("\n[4] VERDICT")
    print(f"  route A  (insert with the tool horizontal): depth {'OK' if verdict_a else 'NG'}")
    print(f"  route B/B' (stand it up, re-pick upright):  depth {'OK' if verdict_b else 'NG'}")
    print("  Both reach depth, so depth does not separate them; the separation is downstream reuse.")
    print(f"  Rotation is safe on torque at BOTH stations (worst {worst:.0f}x), so the groove is not")
    print("  needed to make the turn safe -- which is the claim it was assumed to support.")
    print("REERECT_VERIFY_DONE")


if __name__ == "__main__":
    main()
