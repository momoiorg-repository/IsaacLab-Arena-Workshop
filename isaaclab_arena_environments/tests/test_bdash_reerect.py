# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""spec §4-2: the re-erect turn must land the LEADING end down, for either axis sign.

This is the one place in the leg where being wrong is silent. ``grasp_quat_from_axis``
canonicalises the part's axis into the +x half-plane before it builds the wrist quaternion, so the
wrist's local +x is the part's axis *up to a sign* -- and that sign is exactly what decides which
end finishes pointing at the bench. Turn the wrong way and the part is stood on its TRAILING end,
which for W-B is a Ø32 flange and for W-C a Ø45 one: faces the Ø40 bore cannot accept. The arm would
then run the whole insertion against a part that was upside down, and every predicate downstream
reports honestly on a part in the wrong orientation rather than flagging the turn.

Nothing about that is visible in a trajectory plot, and it is a single sign in one expression, so it
gets a test that enumerates both cases rather than a comment asking the reader to be careful.

Sim-free: torch and quaternion math only. No Isaac Sim, no scene.
"""

from __future__ import annotations

import math
import torch

from isaaclab.utils.math import quat_apply

from isaaclab_arena.controllers.ee_control import grasp_quat_from_axis
from isaaclab_arena.controllers.reerect_controller import ReerectController

CFG = {"turn_steps": 60, "arrive_tol": 0.015, "travel_step": 0.040, "lower_step": 0.010}


def _controller(band=(60.0, 150.0)) -> ReerectController:
    return ReerectController({}, (0.5, 0.0, 0.0), {"max_rot_step": 0.10}, {**CFG, "turn_azimuth_band_deg": list(band)})


def _axes(n: int = 24) -> torch.Tensor:
    """Horizontal unit axes all the way round the circle, both halves of the branch cut."""
    ang = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    return torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], dim=-1)


def test_turn_puts_the_leading_end_down():
    """For every horizontal axis, the planned end pose carries the leading end to world -z."""
    axis = _axes()
    q0 = grasp_quat_from_axis(axis)
    ctrl = _controller()
    station = torch.full((axis.shape[0],), 0.030)
    ctrl.plan(q0, axis, station, torch.ones(axis.shape[0], dtype=torch.bool))

    local_x = torch.zeros_like(axis)
    local_x[:, 0] = 1.0
    x_start = quat_apply(q0, local_x)
    x_end = quat_apply(ctrl.q_end, local_x)
    # The leading end is -axis from the grasp; express it in the wrist frame at the start, then read
    # where that same body direction points after the turn.
    lead_sign = torch.where((x_start * axis).sum(dim=-1) >= 0.0, -1.0, 1.0)
    lead_end = lead_sign.unsqueeze(-1) * x_end
    assert torch.all(lead_end[:, 2] < -0.999), f"leading end not down: min z = {lead_end[:, 2].max():.4f}"


def test_turn_is_a_quarter_turn_about_the_finger_axis():
    """The turn alone rotates exactly 90 deg, and about the CLOSING axis.

    Checked on ``turned_quat`` -- the turn before the aim -- so this pins the turn itself. The aim
    spins about the part's own axis on top, which necessarily moves the closing axis; see
    :func:`test_free_yaw_keeps_the_grip_across_the_cylinder` for what must survive that.
    """
    axis = _axes()
    q0 = grasp_quat_from_axis(axis)
    ctrl = _controller()
    turned = ctrl.turned_quat(q0, axis)

    local_y = torch.zeros_like(axis)
    local_y[:, 1] = 1.0
    y_start = quat_apply(q0, local_y)
    y_end = quat_apply(turned, local_y)
    # A rotation ABOUT the finger axis leaves the finger axis alone: the fingers must still close
    # across the part, or the turn has also changed the grip.
    assert torch.allclose(y_start, y_end, atol=1e-4), "the turn moved the closing axis"

    local_x = torch.zeros_like(axis)
    local_x[:, 0] = 1.0
    x_start, x_end = quat_apply(q0, local_x), quat_apply(turned, local_x)
    cos = (x_start * x_end).sum(dim=-1)
    assert torch.allclose(cos, torch.zeros_like(cos), atol=1e-4), f"not 90 deg: cos = {cos}"


def test_free_yaw_keeps_the_grip_across_the_cylinder():
    """The free yaw may move the fingers around the part, but never off it.

    Spinning the wrist about the part's own axis is free because the part is a cylinder about that
    axis -- and it is not cosmetic: at yaw 0 the arm ends jammed on a joint limit with 20-47 deg of
    orientation error, at 180 deg it converges exactly with 0.65-0.99 rad of travel to spare. What
    must survive the spin is the GRIP: the fingers close along the wrist's local y, so that axis has
    to stay perpendicular to the part's final (vertical) axis, or the fingers are no longer closing
    across the cylinder but along it, which is not a grip at all.

    Any yaw is checked, not just the configured one, because the value is a config knob and a future
    change to it must not be able to quietly invalidate the grip.
    """
    axis = _axes()
    q0 = grasp_quat_from_axis(axis)
    local_x = torch.zeros_like(axis)
    local_x[:, 0] = 1.0
    local_y = torch.zeros_like(axis)
    local_y[:, 1] = 1.0
    for band in ((60.0, 150.0), (0.0, 0.0), (90.0, 90.0), (-120.0, -60.0)):
        ctrl = _controller(band=band)
        ctrl.plan(q0, axis, torch.full((axis.shape[0],), 0.030), torch.ones(axis.shape[0], dtype=torch.bool))
        part_axis = quat_apply(ctrl.q_end, local_x)  # the part's axis after the turn
        closing = quat_apply(ctrl.q_end, local_y)
        dot = (part_axis * closing).sum(dim=-1).abs()
        assert torch.all(dot < 1e-4), f"band {band}: fingers no longer close across the part"
        assert torch.all(part_axis[:, 2].abs() > 0.999), f"band {band}: the part is not vertical"


def test_slerp_is_monotone_and_ends_where_it_was_planned():
    """The turn is a function of the phase counter alone; t=0 is the latch and t=1 is the plan."""
    axis = _axes(8)
    q0 = grasp_quat_from_axis(axis)
    ctrl = _controller()
    ctrl.plan(q0, axis, torch.full((axis.shape[0],), 0.030), torch.ones(axis.shape[0], dtype=torch.bool))

    zero = ctrl._slerp(torch.zeros(axis.shape[0]))
    one = ctrl._slerp(torch.ones(axis.shape[0]))
    # q and -q are the same rotation, so compare through |dot| rather than componentwise.
    assert torch.all((zero * q0).sum(dim=-1).abs() > 0.9999), "t=0 is not the latched pose"
    assert torch.all((one * ctrl.q_end).sum(dim=-1).abs() > 0.9999), "t=1 is not the planned pose"

    local_x = torch.zeros_like(axis)
    local_x[:, 0] = 1.0
    prev = None
    for k in range(11):
        t = torch.full((axis.shape[0],), k / 10.0)
        z = quat_apply(ctrl._slerp(t), local_x)[:, 2] * torch.where(
            (quat_apply(q0, local_x) * axis).sum(dim=-1) >= 0.0, -1.0, 1.0
        )
        if prev is not None:
            assert torch.all(z <= prev + 1e-5), f"the leading end rose again at t={k / 10:.1f}"
        prev = z


def test_turn_progress_saturates_after_the_turn():
    """Past TURN the wrist must HOLD the turned pose, not restart the turn.

    `timer` restarts at every phase transition, so a turn parameterised on it directly falls back to
    0 the moment TURN ends -- which commands the pre-turn pose again and rotates the part flat while
    the arm is descending to set it down. That is not a subtle degradation: measured, it flung the
    part in 4 of 4 episodes with IK residuals of 96-349 mm. The progress must be monotone across the
    whole leg, which is what this pins.
    """
    from isaaclab_arena.controllers.reerect_controller import DONE, LOWER, RELEASE, RETRACT, TRANSIT, TURN

    ctrl = _controller()
    for phase_code, timer, want in (
        (TRANSIT, 0, 0.0),
        (TRANSIT, 40, 0.0),  # a long transit must not start the turn early
        (TURN, 0, 0.0),
        (TURN, 30, 0.5),
        (TURN, 60, 1.0),
        (LOWER, 0, 1.0),  # <- the defect: this used to command 0.0
        (LOWER, 9, 1.0),
        (RELEASE, 0, 1.0),
        (RETRACT, 3, 1.0),
        (DONE, 0, 1.0),
    ):
        t = ctrl.turn_t(torch.tensor([phase_code]), torch.tensor([timer]))
        msg = f"phase={phase_code} timer={timer}: t={float(t):.3f}, want {want}"
        assert abs(float(t) - want) < 1e-6, msg


def test_turn_holds_its_angular_rate_and_never_slews_180_deg():
    """However far the turn has to go, it goes at the same speed -- and never exactly half a circle.

    Two failures in one test, because they share a cause. Aiming the hand into the reachable band
    adds a spin on top of the 90 deg turn, and how much depends on the grasp angle: 90 deg at
    psi = +90 rising toward 180 deg at psi = -90.

    * **Rate.** With the duration fixed at 60 steps, the biggest slews ran at nearly double the
      angular speed of the case that had been tested -- and the part is held by friction with no
      correction available, the turn being open loop by §9. Measured: a 171 deg slew threw the part
      0.35 m within 11 steps and the episode aborted on contact force with an empty gripper.
    * **180 deg.** A slerp between antipodal orientations has no shortest arc -- ``sin(omega)``
      vanishes and the interpolation axis is arbitrary, so the wrist takes an undefined path. Aiming
      at the NEAREST point of the band rather than a fixed azimuth keeps the slew under half a turn;
      this pins that it actually does.
    """
    axis = _axes(36)
    q0 = grasp_quat_from_axis(axis)
    ctrl = _controller()
    ctrl.plan(q0, axis, torch.full((axis.shape[0],), 0.030), torch.ones(axis.shape[0], dtype=torch.bool))

    dot = (q0 * ctrl.q_end).sum(dim=-1).abs().clamp(max=1.0)
    slew = 2.0 * torch.acos(dot)
    assert torch.all(slew < math.radians(175.0)), f"slew reaches {math.degrees(slew.max()):.1f} deg"

    rate = slew / ctrl.turn_len.to(slew.dtype)
    assert torch.all(rate <= ctrl.turn_rate + 1e-6), f"turn runs at {math.degrees(rate.max()):.2f} deg/step"
    # ...and not absurdly slower than asked, or the leg outruns its phase budget instead.
    assert torch.all(rate > ctrl.turn_rate * 0.5), f"turn crawls at {math.degrees(rate.min()):.2f} deg/step"


def test_aim_lands_inside_the_measured_band():
    """The hand must finish pointing somewhere the arm can actually hold.

    The band is measured, not chosen: 60-150 deg converge with joint travel to spare, everything
    outside jams a joint with 20-47 deg of orientation error left. An aim that lands outside it does
    not fail loudly -- the arm pins itself on the limit and the physics solver crawls, which reads as
    a hang rather than as a fault.
    """
    axis = _axes(36)
    q0 = grasp_quat_from_axis(axis)
    ctrl = _controller()
    ctrl.plan(q0, axis, torch.full((axis.shape[0],), 0.030), torch.ones(axis.shape[0], dtype=torch.bool))

    local_z = torch.zeros_like(axis)
    local_z[:, 2] = 1.0
    approach = quat_apply(ctrl.q_end, local_z)
    beta = torch.atan2(approach[:, 1], approach[:, 0])
    lo, hi = ctrl.azimuth_band
    assert torch.all((beta >= lo - 1e-4) & (beta <= hi + 1e-4)), (
        f"aim lands at {sorted(round(math.degrees(float(b)), 1) for b in beta)} deg, band is "
        f"{math.degrees(lo):.0f}-{math.degrees(hi):.0f}"
    )


def test_plan_only_touches_the_envs_it_was_given():
    """A vectorised run must not have one env's turn planned into another's slot."""
    axis = _axes(4)
    q0 = grasp_quat_from_axis(axis)
    ctrl = _controller()
    mask = torch.tensor([True, False, True, False])
    ctrl.plan(q0, axis, torch.full((4,), 0.030), mask)
    identity = torch.zeros(4, 4)
    identity[:, 0] = 1.0
    assert torch.allclose(ctrl.q_end[~mask], identity[~mask]), "an unmasked env was planned"
    assert not torch.allclose(ctrl.q_end[mask], identity[mask]), "a masked env was not planned"


if __name__ == "__main__":
    test_turn_puts_the_leading_end_down()
    test_turn_is_a_quarter_turn_about_the_finger_axis()
    test_free_yaw_keeps_the_grip_across_the_cylinder()
    test_turn_holds_its_angular_rate_and_never_slews_180_deg()
    test_aim_lands_inside_the_measured_band()
    test_slerp_is_monotone_and_ends_where_it_was_planned()
    test_turn_progress_saturates_after_the_turn()
    test_plan_only_touches_the_envs_it_was_given()
    print("BDASH_REERECT_OK")
