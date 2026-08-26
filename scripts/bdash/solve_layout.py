# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Solve the chuck-loading scene layout from REQUIREMENTS, and print every margin it clears by.

Written because remote geometry instructions for this scene have twice been wrong in ways only
arithmetic caught (the two-step soft jaw, the jaw step height), so the standing rule is that the
planning side states requirements and the implementation side fixes the numbers. Nothing here is a
coordinate anyone chose: the poses come out of a search over a requirement set, and every constraint
prints its own slack so a marginal pass is visible as marginal.

Requirements encoded (all distances metres, angles degrees, about the robot base at the origin):

1. REACH     every task point -- the whole tray grasp region, the touch-off datum, the V-block
             groove, the chuck bore -- lies in a horizontal annulus [r_min, r_max].
2. SEPARATION the tray and the chuck are at least `min_separation_deg` apart in bearing from the
             base. This is the actual fix for "everything overlaps in the overhead view": a Franka's
             usable reach tops out around 0.6 m, so the two work areas cannot be pushed apart
             RADIALLY -- only angularly. Separated in bearing, the arm working one area leaves the
             other area's camera view.
3. FOOTPRINT no two fixtures overlap in plan, with a margin; everything sits on the table.
4. CAMERA    a look-down camera beside the tray at a pitch in [cam_pitch_min, cam_pitch_max],
             with the robot BASE COLUMN clear of every sight line to the grasp region.

             This constraint is NECESSARY BUT NOT SUFFICIENT, and the difference cost a run. The
             base is not what occludes the target -- the forearm is, and this solver has no arm
             model at all. The camera it puts exactly orthogonal to the approach scored 79.0% on
             audit_camera_visibility.py against a 95% bar, because the perpendicular station sits
             inside the forearm's swing; a station part-way round toward looking back ALONG the
             approach measured 99.8%. So: take the azimuth this solver produces as a starting
             point, then MEASURE it. The audit, not this file, decides camera placement.

Table extents come from ``probe_scene_extents.py`` -- the table is a stock Nucleus prop whose
footprint is a property of the USD, not of ``table_pose``, and the tray is a rigid body that simply
falls off an edge that is not there.

    python3 scripts/bdash/solve_layout.py --table_x MIN MAX --table_y MIN MAX
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cfg(name):
    with open(os.path.join(ROOT, "configs/bdash/chuck_load", name)) as handle:
        return yaml.safe_load(handle)


def _corners(cx, cy, hx, hy, yaw):
    """The four corners of a rectangle centred (cx, cy), half-extents (hx, hy), rotated by ``yaw``."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + sx * hx * c - sy * hy * s, cy + sx * hx * s + sy * hy * c) for sx in (-1, 1) for sy in (-1, 1)]


def _rect_radii(cx, cy, hx, hy, yaw=0.0):
    """(min, max) distance from the origin to a rotated rectangle's AREA.

    The min is taken in the rectangle's own frame, where the nearest point is the origin clamped to
    the box -- doing it on the axis-aligned bounding box instead would understate the near reach for
    any rotated tray, which is precisely the case being solved for.
    """
    pts = _corners(cx, cy, hx, hy, yaw)
    far = max(math.hypot(x, y) for x, y in pts)
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx, ly = -cx * c + -cy * -s, -cx * s + -cy * c  # origin in the rectangle's frame
    near = math.hypot(max(0.0, abs(lx) - hx), max(0.0, abs(ly) - hy))
    return near, far


def _aabb_of(cx, cy, hx, hy, yaw=0.0):
    """Axis-aligned bounds of a rotated rectangle: (xmin, xmax, ymin, ymax)."""
    pts = _corners(cx, cy, hx, hy, yaw)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def _rect_overlap(a, b):
    """Plan-view gap between two footprints, via their axis-aligned bounds (negative = overlap).

    Deliberately CONSERVATIVE for a rotated rectangle -- an AABB is never smaller than the shape --
    so a pass here is a real pass.
    """
    ax0, ax1, ay0, ay1 = _aabb_of(*a)
    bx0, bx1, by0, by1 = _aabb_of(*b)
    return max(max(ax0 - bx1, bx0 - ax1), max(ay0 - by1, by0 - ay1))


def _bearing(x, y):
    return math.degrees(math.atan2(y, x))


def _angle_gap(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _base_blocks_view(cam, point, base_radius):
    """Does the robot's base column cross the segment camera->point in plan view?"""
    (cx, cy), (px, py) = cam, point
    dx, dy = px - cx, py - cy
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return True
    t = max(0.0, min(1.0, -(cx * dx + cy * dy) / denom))
    return math.hypot(cx + t * dx, cy + t * dy) < base_radius


def solve(args) -> dict:
    task, assets = _cfg("task.yaml"), _cfg("assets.yaml")
    tray, spawn = assets["tray"], task["tray_spawn"]
    wall = float(tray["wall"])
    # The GRASP region, not the tray outline: the TCP never goes closer to a wall than this, so it
    # is the region reach has to cover. The outline still has to fit on the table.
    grasp_hx = (float(tray["outer"][0]) - 2.0 * wall) / 2.0 - float(spawn["grasp_wall_margin"])
    grasp_hy = (float(tray["outer"][1]) - 2.0 * wall) / 2.0 - float(spawn["grasp_wall_margin"])
    tray_hx, tray_hy = float(tray["outer"][0]) / 2.0, float(tray["outer"][1]) / 2.0
    # The jaws do NOT stick out past the body: measured world AABBs are body 0.160 x 0.160 with
    # every jaw inside it (probe_scene_extents.py), because the jaws are recessed into the face.
    # Padding for them anyway cost ~18 mm of chuck radius against a table edge that is the binding
    # constraint here, so the measurement is used rather than the conservative guess.
    chuck_r = float(assets["chuck"]["body_diameter"]) / 2.0
    touch_h = [v / 2.0 for v in assets["touchoff_block"]["size"][:2]]
    vblock_h = [v / 2.0 for v in assets["vblock"]["size"][:2]]

    def on_table(cx, cy, hx, hy, yaw=0.0, margin=0.0):
        x0, x1, y0, y1 = _aabb_of(cx, cy, hx, hy, yaw)
        return (
            args.table_x[0] + margin <= x0
            and x1 <= args.table_x[1] - margin
            and args.table_y[0] + margin <= y0
            and y1 <= args.table_y[1] - margin
        )

    best = None
    # Bearing/radius/yaw grids. Coarse enough to finish in seconds, fine enough that the winner's
    # margins are not an artefact of the step: 2 deg and 10 mm are well under every tolerance below.
    tray_bearings = [b * 2.0 for b in range(int(args.tray_bearing[0] / 2), int(args.tray_bearing[1] / 2) + 1)]
    radii = [args.r_min + 0.01 * i for i in range(int((args.r_max - args.r_min) / 0.01) + 1)]
    # TRAY YAW is a free variable, and it has to be: the tray is 300 x 200 mm, so with its long axis
    # RADIAL the grasp region spans 260 mm radially against a 250 mm annulus -- infeasible at every
    # bearing and radius. Turning the long axis tangential drops the radial span to 160 mm. Measured
    # against the table's COLLISION footprint (x [-0.246, 1.034], y [-0.455, 0.455], from
    # probe_scene_extents.py), the bearings where a world-axis-aligned tray would be tangential are
    # all off the table, so yaw is not an optimisation here -- it is what makes the problem solvable.
    yaws = [y * 5.0 for y in range(0, 36)]  # 0..175 deg; a rectangle is symmetric past that
    chuck_bearings = [_bearing(*args.pin_chuck)] if args.pin_chuck else [b * 2.0 for b in range(-90, 91)]
    chuck_radii = [math.hypot(*args.pin_chuck)] if args.pin_chuck else radii

    for tb, td, yaw_deg in itertools.product(tray_bearings, radii, yaws):
        tx, ty = td * math.cos(math.radians(tb)), td * math.sin(math.radians(tb))
        yaw = math.radians(yaw_deg)
        near, far = _rect_radii(tx, ty, grasp_hx, grasp_hy, yaw)
        if near < args.r_min or far > args.r_max:
            continue
        if not on_table(tx, ty, tray_hx, tray_hy, yaw, args.table_margin):
            continue
        for cb, cd in itertools.product(chuck_bearings, chuck_radii):
            if _angle_gap(tb, cb) < args.min_separation_deg:
                continue
            cx, cy = cd * math.cos(math.radians(cb)), cd * math.sin(math.radians(cb))
            if not args.r_min <= cd <= args.r_max:
                continue
            if not on_table(cx, cy, chuck_r, chuck_r, 0.0, args.table_margin):
                continue
            gap_tray_chuck = _rect_overlap((tx, ty, tray_hx, tray_hy, yaw), (cx, cy, chuck_r, chuck_r, 0.0))
            if gap_tray_chuck < args.fixture_gap:
                continue

            # Score: the tightest normalised margin, so the winner is the most robust layout rather
            # than the one that happens to sit first in the grid.
            # Margins are CAPPED at 1.0. Uncapped, separation dominated everything -- 112 deg beat
            # 92 deg on score alone and swung the chuck round behind the robot's shoulder, buying
            # nothing the requirement asked for while making the transport leg and the demo view
            # worse. A requirement met with a full unit of slack is met; more is not better.
            margins = {
                "reach_near": min(1.0, (near - args.r_min) / 0.05),
                "reach_far": min(1.0, (args.r_max - far) / 0.05),
                "separation": min(1.0, (_angle_gap(tb, cb) - args.min_separation_deg) / 15.0),
                "tray_chuck_gap": min(1.0, (gap_tray_chuck - args.fixture_gap) / 0.05),
                "chuck_reach": min(1.0, min(cd - args.r_min, args.r_max - cd) / 0.05),
                # Keep BOTH work areas in the front arc. Scored, not a hard constraint, so the
                # solver still returns something if the surface makes it impossible -- but it goes
                # negative behind +-90 deg, which is enough to lose to any front-arc solution.
                # Without this the separation requirement was met the lazy way, by swinging the
                # chuck round behind the robot's shoulder while the tray sat straight ahead.
                "chuck_front": min(1.0, (args.max_bearing - abs(cb)) / 30.0),
            }
            # Tie-break toward the smallest arm swing, so an over-rotated chuck never wins a tie.
            score = min(margins.values()) - 1e-4 * _angle_gap(tb, cb)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "tray": (round(tx, 4), round(ty, 4)),
                    "tray_yaw_deg": yaw_deg,
                    "chuck": (round(cx, 4), round(cy, 4)),
                    "tray_bearing": tb,
                    "chuck_bearing": cb,
                    "tray_radius": round(td, 4),
                    "chuck_radius": round(cd, 4),
                    "grasp_near": round(near, 4),
                    "grasp_far": round(far, 4),
                    "separation_deg": round(_angle_gap(tb, cb), 1),
                    "tray_chuck_gap": round(gap_tray_chuck, 4),
                    "margins": {k: round(v, 3) for k, v in margins.items()},
                }

    if best is None:
        print("LAYOUT_INFEASIBLE -- no placement satisfies reach + separation + footprint")
        return {}

    # ---- fixtures: place them in the bearing wedge BETWEEN tray and chuck, inside the annulus.
    tx, ty = best["tray"]
    cx, cy = best["chuck"]
    tray_yaw = math.radians(best["tray_yaw_deg"])
    occupied = [(tx, ty, tray_hx, tray_hy, tray_yaw), (cx, cy, chuck_r, chuck_r, 0.0)]
    fixtures = {}
    for key, half in (("touchoff", touch_h), ("vblock", vblock_h)):
        placed = None
        for d in [args.r_min + 0.01 * i for i in range(int((args.r_max - args.r_min) / 0.01) + 1)]:
            for b in [x * 2.0 for x in range(-90, 91)]:
                fx, fy = d * math.cos(math.radians(b)), d * math.sin(math.radians(b))
                near, far = _rect_radii(fx, fy, half[0], half[1])
                if near < args.r_min or far > args.r_max:
                    continue
                if not on_table(fx, fy, half[0], half[1], 0.0, args.table_margin):
                    continue
                gap = min(_rect_overlap((fx, fy, half[0], half[1], 0.0), o) for o in occupied)
                if gap < args.fixture_gap:
                    continue
                # Sit in the WEDGE between the tray and the chuck, near its middle, rather than
                # wherever the gap is largest -- maximising gap alone threw the touch-off block to
                # bearing +116 deg, behind the robot's shoulder, which is reachable but absurd and
                # would put it outside every camera's frame.
                mid = (best["tray_bearing"] + best["chuck_bearing"]) / 2.0
                cost = _angle_gap(b, mid)
                if placed is None or cost < placed["cost"]:
                    placed = {
                        "pos": (round(fx, 4), round(fy, 4)),
                        "gap": round(gap, 4),
                        "bearing": b,
                        "cost": cost,
                    }
        if placed is None:
            print(f"LAYOUT_INFEASIBLE -- nowhere to put {key}")
            return {}
        fixtures[key] = placed
        occupied.append((placed["pos"][0], placed["pos"][1], half[0], half[1], 0.0))

    # ---- camera: beside the tray on the axis ORTHOGONAL to the arm's approach.
    # The arm reaches the tray radially from the base, so the approach direction is the tray's own
    # bearing; the orthogonal offset is what keeps the arm out of the sight line.
    approach = math.radians(best["tray_bearing"])
    side = (-math.sin(approach), math.cos(approach))
    cam = None
    for sign in (+1.0, -1.0):
        for pitch_deg in [
            args.cam_pitch_min + 0.25 * i for i in range(int((args.cam_pitch_max - args.cam_pitch_min) / 0.25) + 1)
        ]:
            for horiz in [0.40 + 0.01 * i for i in range(41)]:
                px = tx + sign * side[0] * horiz
                py = ty + sign * side[1] * horiz
                pz = horiz * math.tan(math.radians(pitch_deg))
                if not args.cam_height[0] <= pz <= args.cam_height[1]:
                    continue
                if any(
                    _base_blocks_view((px, py), c, args.base_radius)
                    for c in _corners(tx, ty, grasp_hx, grasp_hy, tray_yaw)
                ):
                    continue
                # Prefer the side AWAY from the chuck (so the VLA frame is the tray's business only)
                # and, within that, the middle of the allowed pitch band at the requested standoff.
                # Scoring on raw distance-from-chuck alone just pushes the camera as far back as the
                # grid allows, which trades resolution on a 25 mm shaft for nothing.
                on_far_side = math.hypot(px - cx, py - cy) > math.hypot(tx - cx, ty - cy)
                cost = (0.0 if on_far_side else 1.0) + abs(pitch_deg - args.cam_pitch_target) / 10.0
                cost += abs(horiz - args.cam_standoff) / 0.10
                cand = {
                    "pos": (round(px, 4), round(py, 4), round(pz, 4)),
                    "look_at": (round(tx, 4), round(ty, 4), 0.0),
                    "pitch_deg": round(pitch_deg, 2),
                    "horiz": round(horiz, 4),
                    "chuck_dist": round(math.hypot(px - cx, py - cy), 4),
                    "cost": round(cost, 4),
                }
                if cam is None or cost < cam["cost"]:
                    cam = cand
    if cam is None:
        print("LAYOUT_INFEASIBLE -- no camera pose clears the base column at the required pitch")
        return {}

    best["fixtures"] = fixtures
    best["camera"] = cam
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r_min", type=float, default=0.35, help="inner reach limit (requirement 1)")
    parser.add_argument("--r_max", type=float, default=0.60, help="outer reach limit (requirement 1)")
    parser.add_argument("--min_separation_deg", type=float, default=90.0, help="requirement 2")
    parser.add_argument("--fixture_gap", type=float, default=0.02, help="plan-view gap between footprints")
    parser.add_argument("--tray_bearing", type=float, nargs=2, default=(0.0, 120.0))
    parser.add_argument("--pin_chuck", type=float, nargs=2, default=None, help="keep the chuck at this xy")
    parser.add_argument("--table_x", type=float, nargs=2, required=True, help="measured table AABB in x")
    parser.add_argument("--table_y", type=float, nargs=2, required=True, help="measured table AABB in y")
    parser.add_argument(
        "--table_margin", type=float, default=0.02, help="keep footprints this far inside the table edge"
    )
    parser.add_argument("--max_bearing", type=float, default=90.0, help="prefer |chuck bearing| under this")
    parser.add_argument("--base_radius", type=float, default=0.09, help="Franka link0 column radius")
    parser.add_argument("--cam_pitch_min", type=float, default=50.0)
    parser.add_argument("--cam_pitch_target", type=float, default=52.0, help="preferred look-down pitch")
    parser.add_argument(
        "--cam_standoff", type=float, default=0.65, help="preferred horizontal standoff from the tray centre"
    )
    parser.add_argument("--cam_pitch_max", type=float, default=55.0)
    parser.add_argument("--cam_height", type=float, nargs=2, default=(0.55, 0.95))
    args = parser.parse_args()

    best = solve(args)
    if not best:
        return
    print(
        f"tray      {best['tray']} yaw {best['tray_yaw_deg']:5.1f} deg  bearing {best['tray_bearing']:+6.1f} deg "
        f" radius {best['tray_radius']:.3f}"
    )
    print(f"chuck     {best['chuck']}   bearing {best['chuck_bearing']:+6.1f} deg  radius {best['chuck_radius']:.3f}")
    for key, value in best["fixtures"].items():
        print(f"{key:9} {value['pos']}   bearing {value['bearing']:+6.1f} deg  gap {value['gap'] * 1000:5.1f} mm")
    cam = best["camera"]
    print(f"camera    {cam['pos']} -> {cam['look_at']}  pitch {cam['pitch_deg']:.1f} deg  horiz {cam['horiz']:.3f}")
    print()
    print(f"  reach     grasp region {best['grasp_near']:.3f} .. {best['grasp_far']:.3f} m")
    print(f"  separation {best['separation_deg']:.1f} deg")
    print(f"  tray<->chuck gap {best['tray_chuck_gap'] * 1000:.1f} mm")
    print("  margins " + ", ".join(f"{k}={v:+.2f}" for k, v in best["margins"].items()))
    print("LAYOUT_JSON " + json.dumps(best))
    print("LAYOUT_SOLVED")


if __name__ == "__main__":
    main()
