# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the side-lying grasp orientation and the ScriptedPick seam it plugs into.

No Isaac Sim: ``isaaclab.utils.math`` is pure torch, so this runs in seconds. Two things are being
protected here -- that the derived grasp quaternion really does point the approach axis down and the
fingers across the part, and that adding the seam left the frozen v4/v6 peg behaviour untouched.

    /isaac-sim/python.sh -m pytest isaaclab_arena/tests/test_grasp_quat_from_axis.py
    /isaac-sim/python.sh isaaclab_arena/tests/test_grasp_quat_from_axis.py
"""

import math
import torch

from isaaclab.utils.math import quat_apply

from isaaclab_arena.controllers.ee_control import grasp_quat_from_axis

Q_DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
ROOT_HALF = math.sqrt(0.5)


def _axes(quat):
    """EE frame basis vectors in world coords, for a (N,4) wxyz quaternion."""
    n = quat.shape[0]
    eye = torch.eye(3).unsqueeze(0).expand(n, 3, 3)
    return tuple(quat_apply(quat, eye[:, i, :]) for i in range(3))


def test_known_headings():
    """Hand-derived values, so a sign slip in the quaternion product cannot pass silently."""
    along_x = grasp_quat_from_axis(torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(along_x, Q_DOWN, atol=1e-7), along_x

    # a = +y  ->  psi = pi/2  ->  q = Rz(pi/2) (x) q0 = (0, sqrt(1/2), sqrt(1/2), 0)
    along_y = grasp_quat_from_axis(torch.tensor([[0.0, 1.0, 0.0]]))
    expect = torch.tensor([[0.0, ROOT_HALF, ROOT_HALF, 0.0]])
    assert torch.allclose(along_y, expect, atol=1e-7), along_y

    # a vertical axis has no horizontal component and must fall back to the plain down-quat
    vertical = grasp_quat_from_axis(torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(vertical, Q_DOWN, atol=1e-7), vertical


def test_frame_is_correct_for_every_heading():
    headings = torch.linspace(-math.pi, math.pi, 1001)
    axis = torch.stack([headings.cos(), headings.sin(), torch.zeros_like(headings)], dim=-1)
    quat = grasp_quat_from_axis(axis)
    x_e, y_e, z_e = _axes(quat)

    down = torch.tensor([0.0, 0.0, -1.0]).expand_as(z_e)
    assert torch.allclose(z_e, down, atol=1e-6), "approach axis must stay world -z"
    # the Franka closes along the EE frame's local +-y, so that must be across the part
    assert (y_e * axis).sum(-1).abs().max() < 1e-6, "fingers must close perpendicular to the axis"
    assert torch.allclose(torch.linalg.norm(x_e, dim=-1), torch.ones(len(headings)), atol=1e-6)


def test_sign_canonicalisation_bounds_the_wrist():
    """+a and -a are the same grasp, and must give the SAME command, never a 180 deg flip."""
    headings = torch.linspace(-math.pi, math.pi, 1001)
    axis = torch.stack([headings.cos(), headings.sin(), torch.zeros_like(headings)], dim=-1)
    forward, reverse = grasp_quat_from_axis(axis), grasp_quat_from_axis(-axis)
    assert torch.allclose(forward, reverse, atol=1e-6), "opposite axis signs must agree"

    # |psi| <= pi/2 by construction. Recover psi from the quaternion: q = Rz(psi) (x) q0, and for
    # q0 = (0,1,0,0) that means q = (0, cos(psi/2), sin(psi/2), 0) -- so psi = 2*atan2(q_y, q_x).
    psi = 2.0 * torch.atan2(forward[:, 2], forward[:, 1])
    assert psi.abs().max() <= math.pi / 2 + 1e-6, psi.abs().max()
    # panda_joint7 is limited to +-2.8973 rad and sits near 0.785 at the nominal down-quat, so a
    # quarter turn either way stays well inside its travel.
    assert 0.785 + psi.max().item() < 2.8973 and 0.785 + psi.min().item() > -2.8973


def test_tilted_axes_still_grip_across_the_part():
    """The stepped variants rest tilted (W-B 3.3 deg, W-C 7.9 deg), not flat."""
    for tilt_deg in (3.34, 7.91, 15.0):
        tilt = math.radians(tilt_deg)
        headings = torch.linspace(-math.pi, math.pi, 361)
        axis = torch.stack(
            [
                math.cos(tilt) * headings.cos(),
                math.cos(tilt) * headings.sin(),
                torch.full_like(headings, math.sin(tilt)),
            ],
            dim=-1,
        )
        _, y_e, z_e = _axes(grasp_quat_from_axis(axis))
        assert torch.allclose(z_e, torch.tensor([0.0, 0.0, -1.0]).expand_as(z_e), atol=1e-6)
        # y_e is horizontal, so it can only be perpendicular to the axis's horizontal part; the
        # residual is the axis's own tilt, and it must stay far below the grasp's ~30 deg tolerance
        misalign_deg = math.degrees(math.asin(min(1.0, (y_e * axis).sum(-1).abs().max().item())))
        assert misalign_deg < 1e-4, (tilt_deg, misalign_deg)


def test_scripted_pick_seam_is_a_no_op_by_default():
    """The peg path must behave exactly as it did before the seam was added."""
    from isaaclab_arena.controllers.scripted_pick import ScriptedPick

    cfg = {"approach_height": 0.12, "lift_height": 0.22, "grip_offset": 0.06}
    pick = ScriptedPick(names={"peg": "peg", "socket": "socket"}, ee_cfg={}, pick_cfg=cfg)
    assert pick.grasp_quat_override is None, "the new seam must default to off"

    pick.q_hold = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    assert pick._grasp_quat(None) is pick.q_hold, "with no override the v6 held quat is used verbatim"

    # _waypoints was extracted from step(); reproduce the pre-refactor expressions exactly
    class _FakeSocket:
        class data:
            root_pos_w = torch.tensor([[0.5, 0.1, 0.0], [0.4, -0.2, 0.0]])

    class _FakeEnv:
        class unwrapped:
            scene = {"socket": _FakeSocket}

    gp = torch.tensor([[0.30, 0.10, 0.05], [0.45, -0.05, 0.07]])
    above, lift, transport = pick._waypoints(_FakeEnv, gp)

    want_above = gp.clone()
    want_above[:, 2] += cfg["approach_height"]
    want_lift = gp.clone()
    want_lift[:, 2] = cfg["lift_height"]
    want_transport = torch.cat([_FakeSocket.data.root_pos_w[:, :2], want_lift[:, 2:3]], dim=-1)

    assert torch.equal(above, want_above) and torch.equal(lift, want_lift)
    assert torch.equal(transport, want_transport)
    assert torch.equal(gp, torch.tensor([[0.30, 0.10, 0.05], [0.45, -0.05, 0.07]])), "gp was mutated"


def _fake_env(n=2, ee_pos=(0.30, 0.10, 0.30)):
    """Minimal env exposing just what ScriptedPick.step() touches."""

    class _D:
        target_pos_w = torch.tensor([[list(ee_pos)]]).repeat(n, 1, 1)
        target_quat_w = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]).repeat(n, 1, 1)
        root_pos_w = torch.tensor([[0.5, 0.1, 0.0]]).repeat(n, 1)
        root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
        joint_pos = torch.cat([torch.zeros(n, 7), torch.full((n, 2), 0.0005)], dim=1)

    class _E:
        data = _D()

    class _U:
        num_envs = n
        device = torch.device("cpu")
        scene = {"ee_frame": _E(), "socket": _E(), "peg": _E(), "robot": _E()}

    class _Env:
        unwrapped = _U()

    return _Env()


def test_stall_watchdog_never_runs_on_the_peg_path():
    """The frozen v4/v6 peg path must not gain the watchdog.

    ScriptedPick.step() calls _stall_watchdog ONLY under `grasp_quat_override is not None`. If that
    guard ever regresses, the peg task's transition table changes and its published results move.
    """
    from isaaclab_arena.controllers.scripted_pick import ScriptedPick

    cfg = {
        "approach_height": 0.12,
        "lift_height": 0.22,
        "grip_offset": 0.06,
        "grasp_tol": 0.006,
        "settle_tol": 0.02,
        "close_steps": 15,
        "min_grip_width": 0.018,
        "stall_after_steps": 1,  # would fire immediately if it ran at all
        "max_retries": 3,
    }
    ee_cfg = {"pos_gain": 8.0, "rot_gain": 4.0, "max_pos_step": 0.05, "max_rot_step": 0.30, "action_scale": 0.5}
    pick = ScriptedPick(names={"peg": "peg", "socket": "socket", "robot": "robot"}, ee_cfg=ee_cfg, pick_cfg=cfg)

    called = []
    pick._stall_watchdog = lambda *a, **k: called.append(1)

    env = _fake_env()
    for _ in range(40):  # far beyond stall_after_steps=1
        pick.step(env)
    assert not called, "the stall watchdog ran on the peg path (grasp_quat_override is None)"
    assert int(pick.attempts.max()) == 0, "peg path must never accrue retries"


def test_stall_watchdog_retreats_and_is_bounded():
    """On the chuck path a non-converging phase must retreat to APPROACH, at most max_retries times."""
    from isaaclab_arena.controllers.scripted_pick import APPROACH, ScriptedPick

    cfg = {
        "approach_height": 0.12,
        "lift_height": 0.22,
        "grasp_tol": 0.006,
        "settle_tol": 0.02,
        "close_steps": 15,
        "rot_tol": 0.06,
        "min_grip_width": 0.018,
        "stall_eps": 0.001,
        "stall_after_steps": 5,
        "max_retries": 3,
    }
    ee_cfg = {"pos_gain": 8.0, "rot_gain": 4.0, "max_pos_step": 0.05, "max_rot_step": 0.30, "action_scale": 0.5}
    pick = ScriptedPick(names={"peg": "peg", "socket": "socket", "robot": "robot"}, ee_cfg=ee_cfg, pick_cfg=cfg)

    env = _fake_env()
    n = env.unwrapped.num_envs
    # a grasp target the fake arm can never reach (the fake EE pose never moves) -> permanent stall
    pick.grasp_override = torch.tensor([[0.30, 0.10, 0.00]]).repeat(n, 1)
    pick.grasp_quat_override = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(n, 1)

    for _ in range(200):
        pick.step(env)

    assert int(pick.attempts.max()) == cfg["max_retries"], f"expected {cfg['max_retries']} retries, got {pick.attempts}"
    assert bool(pick.gave_up.all()), "gave_up must be raised once the retry budget is spent"
    assert int(pick.phase.max()) == APPROACH, "a permanently stalled pick should sit at the retreat pose"


def _descending_env(n=1, ee_z=0.17):
    """A fake env whose EE height can be driven step by step, to simulate a real descent.

    ``_fake_env`` holds the EE still, which is what the permanent-stall test needs. This one has to
    MOVE, because the property under test is about a descent that converges.
    """

    class _D:
        def __init__(self):
            self.target_pos_w = torch.tensor([[[0.30, 0.10, ee_z]]]).repeat(n, 1, 1)
            self.target_quat_w = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]).repeat(n, 1, 1)
            self.root_pos_w = torch.tensor([[0.5, 0.1, 0.0]]).repeat(n, 1)
            self.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
            self.joint_pos = torch.cat([torch.zeros(n, 7), torch.full((n, 2), 0.02)], dim=1)

    class _E:
        def __init__(self):
            self.data = _D()

    ee = _E()

    class _U:
        num_envs = n
        device = torch.device("cpu")
        scene = {"ee_frame": ee, "socket": ee, "peg": ee, "robot": ee}

    class _Env:
        unwrapped = _U()

    return _Env(), ee


def test_slow_but_converging_descent_is_never_retreated():
    """A descent that keeps making progress must not be retried, however long it takes.

    This is the property ``_stall_watchdog`` documents ("a slow-but-converging approach is never
    interrupted") and it was BROKEN: the re-arm mask omitted ``to1``, so entering DESCEND kept the
    APPROACH phase's ``best_dist`` (<= settle_tol) while the distance jumped back up to a full
    ``approach_height``. ``progressed`` was then false from the first step of every descent, turning
    the progress watchdog into a fixed ``stall_after_steps`` deadline. Measured cost: retries fired
    in 52/320 episodes and every one of the 36 upright failures had retried at least once, while
    268/268 episodes that never retried succeeded.

    The descent here takes 60 steps against a 20-step budget, and closes 2 mm each step -- always
    more than ``stall_eps``, so every step is real progress and nothing may fire.
    """
    from isaaclab_arena.controllers.scripted_pick import DESCEND, ScriptedPick

    cfg = {
        "approach_height": 0.12,
        "lift_height": 0.22,
        "grasp_tol": 0.006,
        "settle_tol": 0.02,
        "close_steps": 15,
        "rot_tol": 0.06,
        "min_grip_width": 0.018,
        "stall_eps": 0.001,
        "stall_after_steps": 20,  # a third of the descent, so the old deadline would fire ~3 times
        "max_retries": 3,
    }
    ee_cfg = {"pos_gain": 8.0, "rot_gain": 4.0, "max_pos_step": 0.05, "max_rot_step": 0.30, "action_scale": 0.5}
    pick = ScriptedPick(names={"peg": "peg", "socket": "socket", "robot": "robot"}, ee_cfg=ee_cfg, pick_cfg=cfg)

    env, ee = _descending_env(ee_z=0.17)  # exactly the hover point: 0.05 grasp + 0.12 approach
    pick.grasp_override = torch.tensor([[0.30, 0.10, 0.05]])
    pick.grasp_quat_override = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    pick.step(env)  # APPROACH is satisfied on arrival -> DESCEND
    assert int(pick.phase.max()) == DESCEND, "the fake EE starts on the hover point"

    for _ in range(60):  # 0.17 -> 0.05 at 2 mm a step
        ee.data.target_pos_w[:, 0, 2] -= 0.002
        pick.step(env)

    assert int(pick.attempts.max()) == 0, f"a converging descent was retreated: attempts={pick.attempts}"
    assert not bool(pick.gave_up.any())


def test_empty_close_triggers_a_retry():
    """CLOSE finishing with the jaws shut on nothing must route into the same retry path."""
    from isaaclab_arena.controllers.scripted_pick import CLOSE, ScriptedPick

    cfg = {
        "approach_height": 0.12,
        "lift_height": 0.22,
        "grasp_tol": 0.006,
        "settle_tol": 0.02,
        "close_steps": 2,
        "rot_tol": 0.06,
        "min_grip_width": 0.018,  # fake robot reports 0.001 total -> empty
        "stall_eps": 0.001,
        "stall_after_steps": 10_000,  # disable the progress path so only empty-close can fire
        "max_retries": 3,
    }
    ee_cfg = {"pos_gain": 8.0, "rot_gain": 4.0, "max_pos_step": 0.05, "max_rot_step": 0.30, "action_scale": 0.5}
    pick = ScriptedPick(names={"peg": "peg", "socket": "socket", "robot": "robot"}, ee_cfg=ee_cfg, pick_cfg=cfg)

    env = _fake_env()
    n = env.unwrapped.num_envs
    pick.grasp_override = torch.tensor([[0.30, 0.10, 0.30]]).repeat(n, 1)  # reachable: EE already there
    pick.grasp_quat_override = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(n, 1)

    pick._ensure(env)
    pick.phase[:] = CLOSE
    pick.timer[:] = cfg["close_steps"]  # next step completes CLOSE
    pick.step(env)

    assert int(pick.attempts.max()) >= 1, "an empty close must consume a retry"


if __name__ == "__main__":
    test_known_headings()
    test_frame_is_correct_for_every_heading()
    test_sign_canonicalisation_bounds_the_wrist()
    test_tilted_axes_still_grip_across_the_part()
    test_scripted_pick_seam_is_a_no_op_by_default()
    test_stall_watchdog_never_runs_on_the_peg_path()
    test_stall_watchdog_retreats_and_is_bounded()
    test_slow_but_converging_descent_is_never_retreated()
    test_empty_close_triggers_a_retry()
    print("BDASH_GRASP_QUAT_OK")
