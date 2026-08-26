# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""spec §4-1: the touch-off arithmetic recovers the protrusion it was given.

The controller's whole value is that ``p_hat`` stops being an assumption, so the one thing that has
to be pinned is the inversion: put a part at a KNOWN offset from its commanded station, synthesise
the TCP the probe would record on contact, and check the solved ``p_hat`` comes back to that offset.
A sign error here would be invisible in simulation -- the arm would probe, produce a number, and
quietly command the wrong insertion depth.

Sim-free: the geometry is plain arithmetic, and torch is the only import the controller needs for it.
"""

from __future__ import annotations

import pathlib
import sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from isaaclab_arena.controllers.touch_off_controller import DONE, TouchOffController  # noqa: E402

BLOCK_POSE = (0.3867, -0.0509, 0.0)
BLOCK_SIZE = (0.060, 0.060, 0.040)
CFG = {"faces": ["top", "x", "y"], "probe_speed": 0.005, "contact_force": 3.0, "retract": 0.010}


def _controller(n=1):
    ctrl = TouchOffController({"workpiece_finger_sensors": []}, BLOCK_POSE, BLOCK_SIZE, {}, CFG, {})
    ctrl.phase = torch.full((n,), DONE, dtype=torch.long)
    ctrl.face_idx = torch.zeros(n, dtype=torch.long)
    ctrl.contact_tcp = torch.zeros(n, 3, 3)
    ctrl.recorded = torch.zeros(n, 3, dtype=torch.bool)
    ctrl.p_hat = torch.zeros(n)
    ctrl.e_lateral = torch.zeros(n, 2)
    ctrl.station_hat = torch.zeros(n)
    ctrl.measured = torch.zeros(n, dtype=torch.bool)
    return ctrl


def test_top_face_recovers_the_true_protrusion():
    """A part gripped `slip` further out than commanded must read `slip` more protrusion."""
    datum_offset = torch.tensor([0.060])  # W-B: the Ø32 shoulder, 60 mm from the leading end
    top_z = BLOCK_POSE[2] + BLOCK_SIZE[2]
    for slip in (-0.008, -0.002, 0.0, 0.003, 0.011):
        ctrl = _controller()
        # TRUE station = commanded + slip. At contact the tip sits exactly on the block top, so the
        # TCP is one true-station above it -- that is the only thing the probe can observe.
        commanded = torch.tensor([0.065])
        true_station = commanded + slip
        ctrl.contact_tcp[:, 0, 2] = top_z + true_station
        ctrl.recorded[:, 0] = True
        ctrl._solve(commanded, torch.tensor([0.0125]), datum_offset)
        expected = datum_offset - true_station
        assert torch.allclose(ctrl.p_hat, expected, atol=1e-9), f"slip {slip}: {ctrl.p_hat} != {expected}"
        # and the sign is the one the loader needs: gripping further out (positive slip) leaves LESS
        # of the datum feature protruding, not more.
        assert (ctrl.p_hat <= datum_offset - commanded + 1e-12) == (slip >= 0)
        # ...and the station the loader actually consumes is the TRUE one, not the commanded one.
        assert torch.allclose(ctrl.station_hat, true_station, atol=1e-9)
        assert bool(ctrl.measured.all())


def test_side_faces_recover_lateral_offset():
    ctrl = _controller()
    radius = torch.tensor([0.0125])
    nominal_x = BLOCK_POSE[0] + BLOCK_SIZE[0] / 2.0 + radius
    nominal_y = BLOCK_POSE[1] + BLOCK_SIZE[1] / 2.0 + radius
    ctrl.contact_tcp[:, 1, 0] = nominal_x + 0.004  # 4 mm out in x
    ctrl.contact_tcp[:, 2, 1] = nominal_y - 0.006  # 6 mm in, in y
    ctrl.recorded[:, 1] = True
    ctrl.recorded[:, 2] = True
    ctrl._solve(torch.tensor([0.065]), radius, torch.tensor([0.060]))
    assert torch.allclose(ctrl.e_lateral[:, 0], torch.tensor([0.004]), atol=1e-9)
    assert torch.allclose(ctrl.e_lateral[:, 1], torch.tensor([-0.006]), atol=1e-9)


def test_a_missed_face_is_not_recorded_as_a_measurement():
    """Giving up after `max_travel` must leave p_hat untouched, not write a garbage contact."""
    ctrl = _controller()
    ctrl.p_hat = torch.tensor([0.0])
    ctrl.station_hat = torch.tensor([0.0])
    ctrl.contact_tcp[:, 0, 2] = 99.0  # a nonsense TCP, as a give-up would leave
    ctrl.recorded[:, 0] = False  # ...but not marked as measured
    ctrl._solve(torch.tensor([0.065]), torch.tensor([0.0125]), torch.tensor([0.060]))
    assert torch.allclose(ctrl.p_hat, torch.tensor([0.0])), "an unrecorded face must not move p_hat"
    assert not bool(ctrl.measured.any()), "a give-up must not be reported as a measurement"
    assert torch.allclose(ctrl.station_hat, torch.tensor([0.0])), "nor move the station"


def test_probe_targets_stand_off_the_named_face():
    ctrl = _controller()
    station = torch.tensor([0.065])
    tcp = torch.zeros(1, 3)
    bx, by, bz = BLOCK_POSE
    sx, sy, sz = BLOCK_SIZE
    a_top, d_top = ctrl._probe_targets(tcp, "top", station)
    assert torch.allclose(a_top[:, :2], torch.tensor([[bx, by]]), atol=1e-9)
    assert float(a_top[0, 2]) > bz + sz + float(station), "top stand-off must clear the block AND the part"
    assert torch.allclose(d_top[0], torch.tensor([0.0, 0.0, -1.0]))
    a_x, d_x = ctrl._probe_targets(tcp, "x", station)
    assert float(a_x[0, 0]) > bx + sx / 2.0, "x stand-off must be outside the +x face"
    assert torch.allclose(d_x[0], torch.tensor([-1.0, 0.0, 0.0]))
    a_y, d_y = ctrl._probe_targets(tcp, "y", station)
    assert float(a_y[0, 1]) > by + sy / 2.0, "y stand-off must be outside the +y face"
    assert torch.allclose(d_y[0], torch.tensor([0.0, -1.0, 0.0]))


def test_unknown_face_is_rejected_loudly():
    ctrl = _controller()
    try:
        ctrl._probe_targets(torch.zeros(1, 3), "diagonal", torch.tensor([0.065]))
    except ValueError as exc:
        assert "diagonal" in str(exc)
    else:
        raise AssertionError("an unknown face must raise, not silently probe nothing")


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("BDASH_TOUCH_OFF_OK")
