# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""End-effector pose → Franka relative-IK action.

The Franka arm action term is a ``DifferentialInverseKinematicsAction`` in **relative pose** mode
(``command_type='pose'``, ``use_relative_mode=True``, ``scale=0.5``): each step the 6-vector
``[dx, dy, dz, ax, ay, az]`` is scaled by 0.5 and applied to the *current* EE pose as
``apply_delta_pose`` — world-frame position add and **left-multiplied** axis-angle rotation
(``q_des = Δq ⊗ q_cur``). The gripper is a binary DOF: ``< 0`` closes, ``>= 0`` opens.

:func:`ee_pose_action` turns a desired *world* EE pose (read from the ``ee_frame`` sensor, so it is
frame-consistent with whatever the IK controls) into that 7-vector with a clamped P-controller, so
the scripted controllers can just emit waypoints. Everything is vectorized over envs.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul


def read_ee_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Current EE (end_effector frame) world pose, shape (N,3),(N,4). Uses the ``ee_frame`` sensor."""
    data = env.unwrapped.scene["ee_frame"].data
    return data.target_pos_w[:, 0, :], data.target_quat_w[:, 0, :]


def tip_estimate(env, grip_offset: float) -> torch.Tensor:
    """Proprioceptive peg-tip world position (N,3) — the §2.1-compliant stand-in for the peg pose.

    The scripted pick drives the EE (``ee_frame``) to ``peg_tip + grip_offset·peg_axis`` and closes,
    so once grasped the EE frame sits exactly ``grip_offset`` above the tip along the held-down
    approach axis (EE local +z, pointed world-down by :func:`level_to_down`). Hence
    ``tip ≈ ee_pos + R_ee · (grip_offset · ẑ_local)``. This uses only the ``ee_frame`` sensor (robot
    proprioception) and the nominal grasp constant — **no peg ground-truth pose**, which brief §2.1
    forbids as an insertion-controller input. The residual (tip_est ≠ true tip, from peg slip/tilt in
    the gripper) is the realistic unobserved disturbance the F/T loop must absorb."""
    pos, quat = read_ee_pose(env)
    off = torch.zeros_like(pos)
    off[:, 2] = grip_offset
    return pos + quat_apply(quat, off)


def level_to_down(q: torch.Tensor) -> torch.Tensor:
    """Minimally rotate ``q`` (N,4 wxyz) so the EE approach axis (local +z) points world **-z**.

    The Franka ready pose grips ~5° off vertical; carrying that tilt into the peg makes insertion
    jam. Leveling the held orientation to exactly straight-down keeps the grasped peg vertical while
    preserving yaw (the square grip cube is yaw-agnostic anyway)."""
    zloc = torch.tensor([0.0, 0.0, 1.0], device=q.device, dtype=q.dtype).expand(q.shape[0], 3)
    z_w = quat_apply(q, zloc)  # current approach axis in world
    tgt = torch.tensor([0.0, 0.0, -1.0], device=q.device, dtype=q.dtype).expand(q.shape[0], 3)
    v = torch.cross(z_w, tgt, dim=-1)  # rotation axis (unnormalized)
    s = torch.norm(v, dim=-1)
    c = (z_w * tgt).sum(dim=-1)
    angle = torch.atan2(s, c)
    axis = v / s.clamp(min=1e-9).unsqueeze(-1)
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, q)


def grasp_quat_from_axis(axis_w: torch.Tensor) -> torch.Tensor:
    """EE orientation (N,4 wxyz) that grips a cylinder lying along ``axis_w`` (N,3).

    The approach axis stays world **-z** (unchanged from the frozen v4/v6 grasp) and the wrist is
    rotated so the fingers close **across** the cylinder, which is the only way a parallel-jaw
    gripper can pick a part lying on its side.

    Derivation. The v6 down-quat ``q0 = (0,1,0,0)`` is a 180 deg rotation about world x, whose EE
    frame axes are ``x_e = +x``, ``y_e = -y``, ``z_e = -z``; the Franka closes its fingers along the
    EE frame's local **±y** (``panda_finger_joint1`` has axis ``0 1 0`` on ``panda_hand``).
    Pre-multiplying by a world-z rotation of ``psi`` -- pre-multiplication because
    :func:`ee_pose_action` applies ``q_des = dq (x) q_cur`` -- gives
    ``y_e = (sin psi, -cos psi, 0)``, which is orthogonal to ``a = (cos psi, sin psi, 0)``. Hence
    ``q = Rz(psi) (x) q0`` with ``psi = atan2(a_y, a_x)``.

    The axis sign is canonicalised into the +x half-plane first. ``+a`` and ``-a`` are the same
    grasp, so leaving the sign free would let ``atan2`` command 180 deg wrist flips that the IK
    cannot follow (rotation is clamped to ``max_rot_step`` per step, and panda_joint7 is limited to
    +-2.8973 rad). Canonicalising makes ``|psi| <= pi/2`` by construction, so the wrist never has to
    travel more than a quarter turn from its nominal.

    The map is deliberately **stateless** -- it is a pure function of the observed axis, with no
    "nearest to the current wrist" hysteresis. Hysteresis would make the recorded action depend on
    history rather than on anything visible in the frame, which is a subtler version of the very
    coupling that made the v3 dataset unlearnable.

    One residual is irreducible: a cylinder's direction lives on RP^1 (a circle) while the wrist
    angle lives on the line, so any lift is discontinuous somewhere. Here the cut sits at
    ``a_x = 0``, and two near-identical scenes either side of it want opposite wrist signs. That is
    topology, not a bug; it is measured rather than engineered away.

    A vertical (or near-vertical) axis has no horizontal component and falls through to
    ``psi = 0``, i.e. exactly the v6 down-quat -- the correct answer for an upright, yaw-symmetric
    part.
    """
    ax, ay = axis_w[:, 0], axis_w[:, 1]
    flip = (ax < 0) | ((ax == 0) & (ay < 0))
    psi = torch.atan2(torch.where(flip, -ay, ay), torch.where(flip, -ax, ax))

    z_axis = torch.zeros_like(axis_w)
    z_axis[:, 2] = 1.0
    q_down = torch.zeros(axis_w.shape[0], 4, device=axis_w.device, dtype=axis_w.dtype)
    q_down[:, 1] = 1.0  # (w,x,y,z) = (0,1,0,0): approach axis points exactly world -z
    return quat_mul(quat_from_angle_axis(psi, z_axis), q_down)


def _clamp_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    n = torch.norm(v, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / n.clamp(min=1e-9), max=1.0)
    return v * scale


def ee_pose_action(
    env,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    gripper_open: torch.Tensor,
    cfg: dict,
    max_pos_step: float | None = None,
) -> torch.Tensor:
    """Build the (N,7) relative-IK + gripper action that drives the EE toward ``target_pos/quat``.

    Args:
        target_pos: desired EE world position (N,3).
        target_quat: desired EE world orientation wxyz (N,4).
        gripper_open: bool (N,) — True opens, False closes.
        cfg: the ``ee_control`` sub-dict of controllers.yaml.
        max_pos_step: optional per-call override of the position clamp (e.g. slower during insertion).
    """
    cur_pos, cur_quat = read_ee_pose(env)
    step_cap = cfg["max_pos_step"] if max_pos_step is None else max_pos_step

    # position: clamped P-controller in world frame
    dp = _clamp_norm(cfg["pos_gain"] * (target_pos - cur_pos), step_cap)

    # orientation: Δq = q_tgt ⊗ q_cur⁻¹  ->  world-frame axis-angle (matches apply_delta_pose)
    dq = quat_mul(target_quat, quat_conjugate(cur_quat))
    da = _clamp_norm(cfg["rot_gain"] * axis_angle_from_quat(dq), cfg["max_rot_step"])

    raw_arm = torch.cat([dp, da], dim=-1) / cfg["action_scale"]
    grip = torch.where(gripper_open, 1.0, -1.0).unsqueeze(-1)
    return torch.cat([raw_arm, grip], dim=-1)


def pos_reached(env, target_pos: torch.Tensor, tol: float) -> torch.Tensor:
    """Per-env bool: EE within ``tol`` (m) of ``target_pos``."""
    cur_pos, _ = read_ee_pose(env)
    return torch.norm(target_pos - cur_pos, dim=-1) < tol
