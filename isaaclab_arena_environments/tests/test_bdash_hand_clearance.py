# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for the §0-10 hand-clearance geometry and the §0-11 upright yaw rule.

These encode the mechanism that was measured to predict the observed DESCEND stalls to +-2 mm:
the hand body's underside sits `clear_z` above the TCP and spans `+-half_span` along the closing
direction, so a straight-line descent lands on any neighbour taller than that underside.

    /isaac-sim/python.sh isaaclab_arena_environments/tests/test_bdash_hand_clearance.py
"""

import importlib.util
import math
import os

_spec = importlib.util.spec_from_file_location(
    "bdash_chuck_randomization", os.path.join(os.path.dirname(__file__), "..", "mdp", "bdash_chuck_randomization.py")
)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

HAND = dict(half_span=0.100, half_width=0.032, clear_z=0.0374)
UPRIGHT = (0.0, 0.0, 1.0)
FLAT_X = (1.0, 0.0, 0.0)


def test_short_neighbour_is_passed_over():
    """A neighbour below the hand's underside cannot block, however close it is."""
    grasp_z = 0.050
    below = grasp_z + HAND["clear_z"] - 0.001
    ok, worst = R.hand_clearance((0.0, 0.0), grasp_z, (0.0, -1.0), [((0.0, -0.02), below, 0.0125)], **HAND)
    assert ok and worst == 0.0, (ok, worst)


def test_tall_neighbour_inside_the_span_blocks():
    """Directly under the hand's long axis and taller than its underside -> blocked."""
    grasp_z = 0.020
    top = grasp_z + HAND["clear_z"] + 0.030
    ok, worst = R.hand_clearance((0.0, 0.0), grasp_z, (0.0, -1.0), [((0.0, -0.05), top, 0.0125)], **HAND)
    assert not ok
    assert abs(worst - 0.030) < 1e-9, worst


def test_span_is_anisotropic():
    """The hand is long along the closing direction and short across it -- 60 mm proves both."""
    grasp_z, top = 0.020, 0.020 + HAND["clear_z"] + 0.02
    along = R.hand_clearance((0.0, 0.0), grasp_z, (0.0, -1.0), [((0.0, -0.060), top, 0.0)], **HAND)[0]
    across = R.hand_clearance((0.0, 0.0), grasp_z, (0.0, -1.0), [((0.060, 0.0), top, 0.0)], **HAND)[0]
    assert not along, "60 mm along the closing direction is inside the +-100 mm span"
    assert across, "60 mm across it is outside the +-32 mm half-width"


def test_worst_overlap_orders_fallback_candidates():
    """When nothing is clear the least-obstructed candidate must rank first."""
    grasp_z = 0.020
    base = grasp_z + HAND["clear_z"]
    near = [((0.0, -0.05), base + 0.04, 0.0125)]
    far = [((0.0, -0.05), base + 0.01, 0.0125)]
    _, worst_near = R.hand_clearance((0.0, 0.0), grasp_z, (0.0, -1.0), near, **HAND)
    _, worst_far = R.hand_clearance((0.0, 0.0), grasp_z, (0.0, -1.0), far, **HAND)
    assert worst_near > worst_far > 0.0


def test_piece_top_z_matches_pose():
    """Top height must come from the live axis, not a per-variant constant (spec §0-3)."""
    prof = {"length": 0.100, "max_radius": 0.016}
    assert abs(R.piece_top_z(prof, 0.005, UPRIGHT, False) - 0.105) < 1e-12
    # lying flat: the axis contributes no height, so the top is the axis height plus the radius
    assert abs(R.piece_top_z(prof, 0.0125, FLAT_X, True) - (0.0125 + 0.016)) < 1e-12
    # tilted 7.91 deg (the solved W-C rest): the raised end lifts the top
    tilt = math.radians(7.91)
    axis = (math.cos(tilt), 0.0, math.sin(tilt))
    expect = 0.0124 + 0.100 * math.sin(tilt) + 0.016
    assert abs(R.piece_top_z(prof, 0.0124, axis, True) - expect) < 1e-12


def test_upright_yaw_is_perpendicular_to_the_tallest_neighbour():
    """§0-11: the closing direction (hand long axis) must be perpendicular to the tallest bearing."""
    grasp, hand_bottom = (0.0, 0.0), 0.05
    for bearing_deg in range(-180, 180, 7):
        b = math.radians(bearing_deg)
        nb = [((0.3 * math.cos(b), 0.3 * math.sin(b)), hand_bottom + 0.03, 0.0125)]
        psi = R.upright_yaw_rule(grasp, hand_bottom, nb)
        c = R.closing_direction(UPRIGHT, False, psi)
        d = (math.cos(b), math.sin(b))
        assert abs(c[0] * d[0] + c[1] * d[1]) < 1e-9, (bearing_deg, psi, c)
        assert -math.pi / 2 - 1e-9 <= psi <= math.pi / 2 + 1e-9, psi


def test_upright_yaw_picks_the_tallest_not_the_nearest():
    grasp, hb = (0.0, 0.0), 0.05
    near_short = ((0.05, 0.0), hb + 0.005, 0.0125)
    far_tall = ((0.0, 0.30), hb + 0.050, 0.0125)
    psi = R.upright_yaw_rule(grasp, hb, [near_short, far_tall])
    c = R.closing_direction(UPRIGHT, False, psi)
    assert abs(c[1]) < 1e-9, "closing direction should be perpendicular to the +y bearing of the TALL one"


def test_upright_yaw_falls_back_to_zero():
    """Nothing tall enough to interfere -> psi = 0, i.e. exactly the frozen down-quat."""
    assert R.upright_yaw_rule((0.0, 0.0), 0.05, []) == 0.0
    assert R.upright_yaw_rule((0.0, 0.0), 0.05, [((0.05, 0.0), 0.04, 0.0125)]) == 0.0


def test_side_lying_closing_direction_is_across_the_axis():
    for deg in range(0, 180, 11):
        a = math.radians(deg)
        axis = (math.cos(a), math.sin(a), 0.0)
        c = R.closing_direction(axis, True)
        assert abs(c[0] * axis[0] + c[1] * axis[1]) < 1e-12, deg


def test_python_and_torch_yaw_folds_agree():
    """The sampler (pure python) and the teacher (torch) must pick the same closing LINE.

    §0-10's guarantee is computed with the python rule; §0-11's grasp is executed with the torch
    mirror in bdash_chuck_policy._upright_yaw. If the two folds disagree, the sampler certifies a
    target the teacher then approaches at a different yaw, and the guarantee means nothing.

    The invariant is the closing LINE, not the raw angle: psi and psi+pi give opposite vectors but
    an identical hand pose (the hand is symmetric about its centre), so agreement is |dot| == 1.
    """
    import torch

    for deg in range(-180, 181):
        raw = math.radians(deg)

        psi_py = raw
        while psi_py > math.pi / 2:
            psi_py -= math.pi
        while psi_py < -math.pi / 2:
            psi_py += math.pi

        t = torch.tensor([raw])
        psi_t = float(torch.remainder(t + math.pi / 2.0, math.pi) - math.pi / 2.0)

        c_py = R.closing_direction(UPRIGHT, False, psi_py)
        c_t = R.closing_direction(UPRIGHT, False, psi_t)
        dot = c_py[0] * c_t[0] + c_py[1] * c_t[1]
        assert abs(abs(dot) - 1.0) < 1e-9, f"deg={deg} py={psi_py} torch={psi_t} dot={dot}"


if __name__ == "__main__":
    test_short_neighbour_is_passed_over()
    test_tall_neighbour_inside_the_span_blocks()
    test_span_is_anisotropic()
    test_worst_overlap_orders_fallback_candidates()
    test_piece_top_z_matches_pose()
    test_upright_yaw_is_perpendicular_to_the_tallest_neighbour()
    test_upright_yaw_picks_the_tallest_not_the_nearest()
    test_upright_yaw_falls_back_to_zero()
    test_side_lying_closing_direction_is_across_the_axis()
    test_python_and_torch_yaw_folds_agree()
    print("BDASH_HAND_CLEARANCE_OK")
