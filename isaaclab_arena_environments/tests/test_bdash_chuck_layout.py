# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for the B-DASH chuck-loading tray layout sampler.

Draws hundreds of seeded layouts with the real six-piece geometry and asserts the invariants the
recording run depends on: nothing interpenetrates, nothing pokes through a tray wall, the
upright/side-lying split is exact, every part is placed at its true resting height, and the
quaternion written to the sim agrees with the axis used to reason about the part. Run with:
    /isaac-sim/python.sh -m pytest isaaclab_arena_environments/tests/test_bdash_chuck_layout.py
or standalone:
    /isaac-sim/python.sh isaaclab_arena_environments/tests/test_bdash_chuck_layout.py
"""

import importlib.util
import math
import os

# Load by file path: the module defers torch/isaaclab into the event functions, so the sampler and
# its geometry are importable without a simulator.
_spec = importlib.util.spec_from_file_location(
    "bdash_chuck_randomization", os.path.join(os.path.dirname(__file__), "..", "mdp", "bdash_chuck_randomization.py")
)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

import random  # noqa: E402

# --- the real geometry, mirroring configs/bdash/chuck_load/assets.yaml ----------------------
SECTIONS = {
    "W-A": [(0.0125, 0.090)],
    "W-B": [(0.0125, 0.060), (0.016, 0.040)],
    "W-C": [(0.0125, 0.072), (0.0225, 0.008)],
}


def _profile(variant):
    sections = SECTIONS[variant]
    return {
        "variant": variant,
        "sections": sections,
        "length": sum(length for _, length in sections),
        "max_radius": max(r for r, _ in sections),
        "lead_radius": sections[0][0],
        "datum_offset": sections[0][1],
    }


PIECES = [_profile(v) for v in ("W-A", "W-A", "W-B", "W-B", "W-C", "W-C")]
CAVITY = (0.290, 0.190)  # tray outer 0.300 x 0.200 less a 5 mm wall on each side
FLOOR_TOP = 0.005
LAYOUT_KW = dict(
    cavity_xy=CAVITY,
    floor_top_z=FLOOR_TOP,
    side_lying_frac=0.4,
    upright_tilt_deg=0.0,
    min_clearance=0.004,
    wall_inset_body=0.003,
    grasp_wall_margin=0.015,
    grasp_station=0.030,
    place_epsilon=0.0002,
    jaw_half_span=0.050,
    jaw_half_width=0.012,
    max_tries=400,
)


def _rotate(quat, v):
    """Rotate v by quaternion (w, x, y, z) -- the same convention the predicates use."""
    w, x, y, z = quat
    tx, ty, tz = 2.0 * (y * v[2] - z * v[1]), 2.0 * (z * v[0] - x * v[2]), 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def test_resting_pose_matches_hand_derivation():
    """A stepped cylinder does not lie flat: it pivots onto its leading end."""
    flat = R.resting_pose(SECTIONS["W-A"])
    assert abs(flat["tilt_rad"]) < 1e-9
    assert abs(flat["origin_height"] - 0.0125) < 1e-9

    for variant, expect_deg, expect_com_mm in (("W-B", 3.34, 56.1), ("W-C", 7.91, 46.6)):
        rest = R.resting_pose(SECTIONS[variant])
        assert abs(math.degrees(rest["tilt_rad"]) - expect_deg) < 0.02, (variant, rest)
        assert abs(rest["com_s"] * 1000.0 - expect_com_mm) < 0.1, (variant, rest)
        # the solved line must put the leading face and the step exactly on the floor
        tilt, h0 = rest["tilt_rad"], rest["origin_height"]
        lead_r, lead_len = SECTIONS[variant][0]
        assert abs(h0 - lead_r * math.cos(tilt)) < 1e-9, "leading end should touch the floor"
        step_axis_z = h0 + lead_len * math.sin(tilt)
        step_r = SECTIONS[variant][1][0]
        assert abs(step_axis_z - step_r * math.cos(tilt)) < 1e-9, "the step should touch the floor"


def test_quat_and_axis_agree():
    """The orientation written to the sim must map local +Z onto the axis used for planning."""
    for tilt_deg in (-20.0, 0.0, 3.34, 7.91, 45.0, 87.0, 90.0):
        for heading_deg in (-179.0, -90.0, 0.0, 37.0, 90.0, 179.0):
            tilt, heading = math.radians(tilt_deg), math.radians(heading_deg)
            axis = R.axis_from_tilt(tilt, heading)
            spun = _rotate(R.quat_from_tilt(tilt, heading), (0.0, 0.0, 1.0))
            assert max(abs(a - b) for a, b in zip(axis, spun)) < 1e-9, (tilt_deg, heading_deg, axis, spun)
    # the flat side-lying case must reproduce the documented "90 deg about +Y" pose
    flat = R.quat_from_tilt(0.0, 0.0)
    assert max(abs(a - b) for a, b in zip(flat, (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0))) < 1e-12


def test_layout_invariants_over_many_draws(draws=500):
    half_x, half_y = CAVITY[0] / 2.0, CAVITY[1] / 2.0
    worst = 0
    side_counts = set()
    for seed in range(draws):
        placements, tries = sample_or_fail(seed)
        worst = max(worst, tries)
        side_counts.add(sum(p["side_lying"] for p in placements))

        segments = []
        for piece, place in zip(PIECES, placements):
            axis, (px, py, pz) = place["axis"], place["pos"]
            # the quaternion and the axis must still agree once inside a real layout
            spun = _rotate(place["quat"], (0.0, 0.0, 1.0))
            assert max(abs(a - b) for a, b in zip(axis, spun)) < 1e-9

            radius = piece["max_radius"]
            far = (px + piece["length"] * axis[0], py + piece["length"] * axis[1])
            for point in ((px, py), far):
                assert abs(point[0]) <= half_x - LAYOUT_KW["wall_inset_body"] - radius + 1e-9, "x wall breach"
                assert abs(point[1]) <= half_y - LAYOUT_KW["wall_inset_body"] - radius + 1e-9, "y wall breach"

            grasp = (px + LAYOUT_KW["grasp_station"] * axis[0], py + LAYOUT_KW["grasp_station"] * axis[1])
            assert abs(grasp[0]) <= half_x - LAYOUT_KW["grasp_wall_margin"] + 1e-9, "grasp too near an x wall"
            assert abs(grasp[1]) <= half_y - LAYOUT_KW["grasp_wall_margin"] + 1e-9, "grasp too near a y wall"

            rest = R.resting_pose(piece["sections"])
            if place["side_lying"]:
                assert abs(axis[2] - math.sin(rest["tilt_rad"])) < 1e-9, "side-lying tilt should be the solved one"
                expect_z = FLOOR_TOP + rest["origin_height"]
            else:
                assert axis[2] > math.cos(math.radians(LAYOUT_KW["upright_tilt_deg"])) - 1e-9, "upright too tilted"
                expect_z = FLOOR_TOP + piece["lead_radius"] * math.sqrt(max(0.0, 1.0 - axis[2] ** 2))
            assert abs(pz - expect_z - LAYOUT_KW["place_epsilon"]) < 1e-9, "not placed at its resting height"

            segments.append(((px, py), far, radius))

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                (a0, a1, ra), (b0, b1, rb) = segments[i], segments[j]
                gap = R._segment_distance_2d(a0, a1, b0, b1) - ra - rb
                assert gap >= LAYOUT_KW["min_clearance"] - 1e-9, f"overlap seed={seed} pair=({i},{j}) gap={gap}"

        # The open jaws must reach every grasp station without striking a neighbour. This is the
        # constraint that body clearance cannot express, and violating it made the teacher grip the
        # wrong part.
        for i, (piece, place) in enumerate(zip(PIECES, placements)):
            grasp = (
                place["pos"][0] + LAYOUT_KW["grasp_station"] * place["axis"][0],
                place["pos"][1] + LAYOUT_KW["grasp_station"] * place["axis"][1],
            )
            c0, c1 = R.jaw_corridor(grasp, place["axis"], place["side_lying"], LAYOUT_KW["jaw_half_span"])
            for j, (b0, b1, rb) in enumerate(segments):
                if i == j:
                    continue
                gap = R._segment_distance_2d(c0, c1, b0, b1) - rb
                assert (
                    gap >= LAYOUT_KW["jaw_half_width"] - 1e-9
                ), f"jaw corridor of {i} hits piece {j} (seed={seed}, gap={gap * 1000:.1f} mm)"

    # exactly round(0.4 * 6) = 2 lying down in EVERY episode, not just on average
    assert side_counts == {2}, side_counts
    return worst


def sample_or_fail(seed):
    return R.sample_tray_layout(random.Random(seed), pieces=PIECES, **LAYOUT_KW)


def test_infeasible_layout_raises_instead_of_overlapping():
    """A loud failure beats the silent interpenetration Isaac Lab's sampler would hand back."""
    try:
        R.sample_tray_layout(
            random.Random(0),
            pieces=PIECES,
            **{**LAYOUT_KW, "cavity_xy": (0.10, 0.08), "max_tries": 50},
        )
    except RuntimeError as err:
        assert "could not lay out" in str(err)
        return
    raise AssertionError("a 100 x 80 mm cavity cannot hold six parts, but the sampler returned a layout")


if __name__ == "__main__":
    test_resting_pose_matches_hand_derivation()
    test_quat_and_axis_agree()
    worst_tries = test_layout_invariants_over_many_draws()
    test_infeasible_layout_raises_instead_of_overlapping()
    print(f"BDASH_CHUCK_LAYOUT_OK max_tries_used={worst_tries}")
