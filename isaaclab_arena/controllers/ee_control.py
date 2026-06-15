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

from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_angle_axis, quat_mul, quat_apply


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
    z_w = quat_apply(q, zloc)                       # current approach axis in world
    tgt = torch.tensor([0.0, 0.0, -1.0], device=q.device, dtype=q.dtype).expand(q.shape[0], 3)
    v = torch.cross(z_w, tgt, dim=-1)               # rotation axis (unnormalized)
    s = torch.norm(v, dim=-1)
    c = (z_w * tgt).sum(dim=-1)
    angle = torch.atan2(s, c)
    axis = v / s.clamp(min=1e-9).unsqueeze(-1)
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, q)


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
