# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH chuck-loading phase predicates, all decided from simulator state.

Pure functions over the env's scene tensors (torch only -- no Isaac Sim import -- so they unit-test
without launching the simulator, exactly like :mod:`bdash_peg_predicates`). Each returns a bool
tensor of shape ``(num_envs,)``; the two measurement helpers return floats instead.

**Multiple candidate workpieces.** Unlike the peg task there are K workpieces in the tray (6 with
``--variants all``), so every predicate has to know which one the episode is about. That is latched
at reset onto ``env.bdash_target_idx`` -- a ``(N,)`` long tensor -- by
:func:`~isaaclab_arena_environments.mdp.bdash_chuck_randomization.select_target_workpiece`. An env
attribute is used rather than a termination-term param because ``TerminationTermCfg.params`` are
frozen when the task is built, and ``num_envs`` is not known then; the same ``env`` object reaches
the termination manager, the recorder terms, the event functions and ``record_vla_demos.py``, so one
attribute serves all four call sites and is trivially faked in the sim-free test.

When the attribute is absent the predicates fall back to reducing over *all* workpieces with
``any``. That keeps the module usable in a scene with no selector, and keeps ``grasped`` honest when
the policy picks up a part that was not the one asked for -- but note that under the fallback
``grasped`` and ``lifted`` may be satisfied by *different* workpieces.

Geometry conventions (meters), all overridable via params and externalized in
``configs/bdash/chuck_load/task.yaml``:

  * each workpiece USD is built along local **+Z** with its origin at the **leading end** face
    centre (the end that enters the chuck), so its tip offset is the origin and its axis, pointing
    leading -> trailing, is the local +Z axis rotated into the world
  * the chuck bore axis is world **+Z** and its face is ``chuck_face_height`` above the chuck root
  * the tray rim is ``tray_rim_height`` above the tray root
"""

from __future__ import annotations

import torch

# workpiece local axis, leading end -> trailing end
_LOCAL_AXIS = (0.0, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# small torch helpers (no isaaclab dependency)
# --------------------------------------------------------------------------------------
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (w, x, y, z). Batched over leading dims."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _stack(env, names, field: str) -> torch.Tensor:
    """Stack a per-entity data field over K entities -> (N, K, D)."""
    return torch.stack([getattr(env.scene[n].data, field) for n in names], dim=1)


def _workpiece_axes(env, names) -> torch.Tensor:
    """Unit axis of each workpiece in world coords, leading -> trailing end. (N, K, 3)."""
    quat = _stack(env, names, "root_quat_w")
    local = torch.as_tensor(_LOCAL_AXIS, device=quat.device, dtype=quat.dtype).expand(quat.shape[:-1] + (3,))
    return _quat_rotate(quat, local)


def _workpiece_tips(env, names, tip_offset) -> torch.Tensor:
    """World position of each workpiece's leading end. (N, K, 3)."""
    pos = _stack(env, names, "root_pos_w")
    quat = _stack(env, names, "root_quat_w")
    off = torch.as_tensor(tip_offset, device=pos.device, dtype=pos.dtype).expand_as(pos)
    return pos + _quat_rotate(quat, off)


def _filtered_norms(env, sensors) -> torch.Tensor:
    """Per-workpiece sum of |filtered contact force| (workpiece<->fingers). (N, K).

    One ContactSensor per workpiece: a filtered sensor is one-body-to-many-filters, so a single
    regex sensor spanning K workpieces *and* filtering against the fingers is not expressible.
    """
    return torch.stack(
        [torch.norm(env.scene[s].data.force_matrix_w, dim=-1).flatten(start_dim=1).sum(dim=1) for s in sensors],
        dim=1,
    )


def _net_norms(env, sensors) -> torch.Tensor:
    """Per-workpiece norm of the net contact force. (N, K)."""
    return torch.stack(
        [torch.norm(env.scene[s].data.net_forces_w, dim=-1).flatten(start_dim=1).sum(dim=1) for s in sensors],
        dim=1,
    )


def gripper_width(env, robot_name: str = "robot") -> torch.Tensor:
    """Total gripper opening = |finger_joint_1| + |finger_joint_2| (last two robot joints)."""
    jp = env.scene[robot_name].data.joint_pos
    return jp[:, -1].abs() + jp[:, -2].abs()


def target_index(env, attr: str = "bdash_target_idx") -> torch.Tensor | None:
    """The latched per-env target workpiece index, or None when no selector has run."""
    idx = getattr(env, attr, None)
    return idx if isinstance(idx, torch.Tensor) else None


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Select entry ``idx[n]`` from ``x[n]`` along dim 1. (N, K, ...) -> (N, ...)."""
    return x[torch.arange(x.shape[0], device=x.device), idx]


def _reduce(mask: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
    """Gather the target's value, or fall back to ``any`` over all workpieces. (N, K) -> (N,)."""
    return mask.any(dim=1) if idx is None else _gather(mask, idx)


# --------------------------------------------------------------------------------------
# pilot predicates -- these gate the VLA demo recording
# --------------------------------------------------------------------------------------
def grasped(
    env,
    *,
    robot_name: str = "robot",
    finger_sensors=(),
    width_min: float = 0.020,
    width_max: float = 0.048,
    grasp_force: float = 1.0,
    target_attr: str = "bdash_target_idx",
) -> torch.Tensor:
    """gripper width in the closed range /\\ workpiece-finger contact force > threshold.

    The width window spans every graspable section: the Ø25 shaft closes to about 0.025, the Ø32
    step to 0.032, and the Ø45 flange -- the widest thing that could be gripped -- to 0.045.
    """
    width = gripper_width(env, robot_name)
    in_close = (width > width_min) & (width < width_max)
    contact = _filtered_norms(env, finger_sensors) > grasp_force
    return in_close & _reduce(contact, target_index(env, target_attr))


def grasped_index(
    env,
    *,
    robot_name: str = "robot",
    finger_sensors=(),
    width_min: float = 0.020,
    width_max: float = 0.048,
    grasp_force: float = 1.0,
) -> torch.Tensor:
    """Which workpiece is actually held, as a (N,) long tensor; -1 where nothing is.

    Diagnostic only -- it is what tells a failed episode "the gripper closed on the wrong part"
    apart from "the gripper closed on nothing".
    """
    width = gripper_width(env, robot_name)
    in_close = (width > width_min) & (width < width_max)
    force = _filtered_norms(env, finger_sensors)
    holding = force > grasp_force
    best = torch.argmax(torch.where(holding, force, torch.zeros_like(force)), dim=1)
    any_held = holding.any(dim=1) & in_close
    return torch.where(any_held, best, torch.full_like(best, -1))


def lifted(
    env,
    *,
    workpiece_names=(),
    tray_name: str = "bdash_tray",
    robot_name: str = "robot",
    finger_sensors=(),
    rim_height: float = 0.030,
    height: float = 0.060,
    speed_max: float = 0.08,
    tip_offset=(0.0, 0.0, 0.0),
    width_min: float = 0.020,
    width_max: float = 0.048,
    grasp_force: float = 1.0,
    target_attr: str = "bdash_target_idx",
) -> torch.Tensor:
    """SUCCESS for the pick stage: grasped /\\ clear of the tray rim by ``height`` /\\ moving slowly.

    This is where the VLA's episode is cut. The speed gate matters because the workpiece swings
    after it leaves the tray, and a demo cut mid-swing teaches the policy to release early.

    Height is measured on the workpiece root, i.e. the leading end face centre: for an upright part
    that is its bottom face and for a side-lying part it is the axis, so the two pose classes differ
    by at most a shaft radius against a 60 mm threshold.
    """
    held = grasped(
        env,
        robot_name=robot_name,
        finger_sensors=finger_sensors,
        width_min=width_min,
        width_max=width_max,
        grasp_force=grasp_force,
        target_attr=target_attr,
    )
    tips = _workpiece_tips(env, workpiece_names, tip_offset)
    rim_z = env.scene[tray_name].data.root_pos_w[:, 2] + rim_height
    clear = (tips[..., 2] - rim_z.unsqueeze(1)) > height
    speed = torch.norm(_stack(env, workpiece_names, "root_lin_vel_w"), dim=-1)
    return held & _reduce(clear & (speed < speed_max), target_index(env, target_attr))


def dropped(
    env,
    *,
    workpiece_names=(),
    min_z: float = -0.05,
    target_attr: str = "bdash_target_idx",
) -> torch.Tensor:
    """The workpiece fell off the table.

    ``mdp.root_height_below_minimum`` cannot express this: it takes a single ``asset_cfg`` and so
    cannot follow a target that is chosen per episode.
    """
    below = _stack(env, workpiece_names, "root_pos_w")[..., 2] < min_z
    return _reduce(below, target_index(env, target_attr))


# --------------------------------------------------------------------------------------
# chuck-loading predicates -- signatures land now, wired into terminations after GATE 1
# --------------------------------------------------------------------------------------
def seated(
    env,
    *,
    workpiece_names=(),
    chuck_name: str = "bdash_chuck_body",
    face_height: float = 0.080,
    seat_angle_rad=(),
    min_insert_depth: float = 0.030,
    max_insert_depth: float = 0.080,
    bore_clearance: float = 0.0075,
    tip_offset=(0.0, 0.0, 0.0),
    target_attr: str = "bdash_target_idx",
) -> torch.Tensor:
    """SUCCESS for the load stage: axis aligned to the bore /\\ deep enough /\\ inside the bore.

    The seat angle is measured **signed** against world +Z, not as an absolute value, so an
    upside-down workpiece is never seated however deep it is. That matters: W-B's shoulder and
    W-C's flange are the protrusion datum and the seating face respectively, and both only work one
    way up. (``bdash_handoff_logger`` takes an absolute value, which is right for a symmetric peg
    and wrong here.)

    ``seat_angle_rad`` is per workpiece, ordered like ``workpiece_names``, because the flanged
    variant seats to a tighter angle than the plain one.
    """
    axes = _workpiece_axes(env, workpiece_names)
    tips = _workpiece_tips(env, workpiece_names, tip_offset)
    chuck = env.scene[chuck_name].data.root_pos_w
    face_z = chuck[:, 2] + face_height

    theta = torch.acos(torch.clamp(axes[..., 2], -1.0, 1.0))
    tol = torch.as_tensor(seat_angle_rad, device=theta.device, dtype=theta.dtype).view(1, -1)
    depth = face_z.unsqueeze(1) - tips[..., 2]
    lateral = torch.norm(tips[..., :2] - chuck[:, None, :2], dim=-1)

    # UPPER bound as well as lower. The bore is a through hole, so a part that was released before
    # anything held it does not stop at the bottom -- it falls out, and `depth` then keeps growing.
    # With only the lower bound, a part lying on the table under the chuck reads as seated: measured,
    # W-B came out at 83 mm of "insertion" against a bore only 80 mm deep and still passed.
    ok = (theta < tol) & (depth > min_insert_depth) & (depth < max_insert_depth) & (lateral < bore_clearance)
    return _reduce(ok, target_index(env, target_attr))


def _in_bore_index(env, workpiece_names, chuck_name, tip_offset) -> torch.Tensor:
    """Fallback target for the measurement helpers: whichever workpiece is nearest the bore axis."""
    tips = _workpiece_tips(env, workpiece_names, tip_offset)
    chuck_xy = env.scene[chuck_name].data.root_pos_w[:, None, :2]
    return torch.argmin(torch.norm(tips[..., :2] - chuck_xy, dim=-1), dim=1)


def protrusion_m(
    env,
    *,
    workpiece_names=(),
    chuck_name: str = "bdash_chuck_body",
    face_height: float = 0.080,
    datum_offsets=(),
    tip_offset=(0.0, 0.0, 0.0),
    target_attr: str = "bdash_target_idx",
) -> torch.Tensor:
    """How far the variant's datum feature stands off the chuck face, in meters. (N,) float.

    The datum point is ``tip + datum_offset * axis``, so this stays correct while the part is still
    tilting into the bore; once seated it reduces to ``tip_z + datum_offset - face_z``. This is the
    quantity the touch-off probe measures on the real machine, which is why it is a measurement and
    not a bool.
    """
    axes = _workpiece_axes(env, workpiece_names)
    tips = _workpiece_tips(env, workpiece_names, tip_offset)
    face_z = env.scene[chuck_name].data.root_pos_w[:, 2] + face_height

    offsets = torch.as_tensor(datum_offsets, device=tips.device, dtype=tips.dtype).view(1, -1, 1)
    datum_z = (tips + offsets * axes)[..., 2]

    idx = target_index(env, target_attr)
    if idx is None:
        idx = _in_bore_index(env, workpiece_names, chuck_name, tip_offset)
    return _gather(datum_z, idx) - face_z


def protrusion_ok(
    env,
    *,
    workpiece_names=(),
    chuck_name: str = "bdash_chuck_body",
    face_height: float = 0.080,
    datum_offsets=(),
    nominals=(),
    tolerances=(),
    tip_offset=(0.0, 0.0, 0.0),
    target_attr: str = "bdash_target_idx",
) -> torch.Tensor:
    """Measured protrusion within the target variant's acceptance window."""
    measured = protrusion_m(
        env,
        workpiece_names=workpiece_names,
        chuck_name=chuck_name,
        face_height=face_height,
        datum_offsets=datum_offsets,
        tip_offset=tip_offset,
        target_attr=target_attr,
    )
    idx = target_index(env, target_attr)
    if idx is None:
        idx = _in_bore_index(env, workpiece_names, chuck_name, tip_offset)
    nominal = torch.as_tensor(nominals, device=measured.device, dtype=measured.dtype)[idx]
    tol = torch.as_tensor(tolerances, device=measured.device, dtype=measured.dtype)[idx]
    return (measured - nominal).abs() < tol


def force_violation(
    env,
    *,
    contact_sensors=(),
    force_max: float = 60.0,
    target_attr: str = "bdash_target_idx",
) -> torch.Tensor:
    """Net contact-force norm on the workpiece exceeds the jam limit."""
    return _reduce(_net_norms(env, contact_sensors) > force_max, target_index(env, target_attr))


def load_failed(
    env,
    *,
    contact_sensors=(),
    force_max: float = 60.0,
    target_attr: str = "bdash_target_idx",
    clamp_attr: str = "bdash_jaws_commanded",
) -> torch.Tensor:
    """Force-exceed component of a failed load. The timeout component is the episode length.

    ARMED ONLY UNTIL THE CLAMP IS COMMANDED. Jam detection exists to catch a part binding against
    the bore on the way in; once the jaws are told to close, a large contact force on the part is
    the CLAMP -- the very thing the cycle is for. Without this gate the abort fired on 5 of 6
    otherwise-good loads mid-stroke, once in the same step as `chuck_closed` itself: the machine was
    aborting because it succeeded.
    """
    violated = force_violation(env, contact_sensors=contact_sensors, force_max=force_max, target_attr=target_attr)
    clamping = getattr(env.unwrapped, clamp_attr, None)
    if clamping is None:
        return violated
    return violated & ~clamping


def loaded(
    env,
    *,
    chuck_name: str = "bdash_chuck_body",
    face_height: float = 0.080,
    stations=(),
    target_depths=(),
    force_seated=(),
    fixture_sensors=(),
    seat_force: float = 8.0,
    depth_tol: float = 0.002,
    over_chuck_radius: float = 0.040,
    robot_name: str = "robot",
    min_grip_width: float = 0.018,
    target_attr: str = "bdash_target_idx",
) -> torch.Tensor:
    """TERMINAL for the load stage: the part is loaded to its commanded depth (spec §4-3).

    **Deliberately not a QC test and deliberately not privileged.** ``seated``/``protrusion_ok``
    read the workpiece's ground-truth pose, which is the ``/privileged/*`` namespace: those decide
    whether the part is a good part, which is a judgement made *after* the irreversible commit
    (§2.3), not a reason to stop moving. A termination that consumed them would let the teacher
    stop on information the real cell does not have at that instant (§0-5).

    Everything here is a signal the machine really has:

    * ``depth`` is proprioceptive. The chuck is a fixture at a surveyed pose, the TCP comes from the
      ``ee_frame`` sensor, and the grasp station is the commanded grip -- so
      ``depth = face_z - (tcp_z - station)`` needs no workpiece pose. Equivalent to servoing the
      measured protrusion ``p_hat = datum_offset - depth`` onto its nominal, which is the form
      ``touch_off`` will supply once it replaces the computed ``p_hat`` with a probed one.
    * ``force`` is the wrist F/T model already used by the insertion controller.

    Which of the two applies is per variant, in ``force_seated``: a variant whose datum feature is a
    FLANGE seats by hitting the face, and its depth target is whatever the flange geometry gives --
    stopping on depth would command a number the flange has already refused.
    """
    from isaaclab_arena.controllers.ee_control import read_ee_pose

    tcp, _ = read_ee_pose(env)
    chuck = env.unwrapped.scene[chuck_name].data.root_pos_w
    face_z = chuck[:, 2] + face_height
    idx = target_index(env, target_attr)

    # GATING, and it is not optional. `depth` below is a pure height test, so on its own it is
    # satisfied the moment the TCP drops low enough -- including while the arm is still down in the
    # TRAY picking the part up. Measured with the gate missing: the termination fired in 93.8% of
    # episodes while `seated` (the part is actually in the bore) held in only 25.0%.
    #
    # Both gates stay non-privileged. The chuck is a surveyed fixture so its axis is known, the TCP
    # is the `ee_frame` sensor, and gripper width is on §0-5's list of signals the real cell has.
    over_chuck = torch.norm(tcp[:, :2] - chuck[:, :2], dim=-1) < over_chuck_radius
    holding = gripper_width(env, robot_name) > min_grip_width

    def _per_target(values):
        row = torch.as_tensor(values, device=tcp.device, dtype=tcp.dtype).view(1, -1)
        return row.expand(tcp.shape[0], -1) if idx is None else _gather(row.expand(tcp.shape[0], -1), idx)

    station = _per_target(stations)
    depth = face_z - (tcp[:, 2] - station)
    by_depth = depth > (_per_target(target_depths) - depth_tol)

    if fixture_sensors:
        # sum over contact points AND over the fixture filters: a flange landing on the face and a
        # part wedged on a jaw are both "the chuck is pushing back".
        fz = torch.stack(
            [
                env.unwrapped.scene[s].data.force_matrix_w.flatten(start_dim=1, end_dim=-2).sum(dim=1)[:, 2]
                for s in fixture_sensors
            ],
            dim=1,
        )
        fz = fz.sum(dim=1) if idx is None else _gather(fz, idx)
        by_force = fz.abs() > seat_force
    else:
        by_force = torch.zeros_like(by_depth)

    use_force = _per_target(force_seated) > 0.5
    return torch.where(use_force, by_force, by_depth) & over_chuck & holding


def chuck_closed(env, *, attr: str = "bdash_jaws_closed") -> torch.Tensor:
    """TERMINAL for the load stage once the chuck actuates: the jaws have completed their stroke.

    spec §2.3 puts the irreversible commit at "閉爪＋サイクルスタート発火", and QC judges what
    happened only AFTER that. So the episode ends on machine state -- the chuck reports its jaws
    are home -- and nothing about whether the part is any good enters the termination. Whether it
    was a good load is `protrusion_ok` / seat angle, reported alongside and never wired in here.

    Non-privileged by construction: a jaw drive knows its own position.
    """
    flag = getattr(env.unwrapped, attr, None)
    if flag is None:
        return torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    return flag
