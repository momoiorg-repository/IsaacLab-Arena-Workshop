# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""V-DASH phase predicates (brief §3.4), all decided from simulator state.

These are pure functions over the env's scene tensors (torch only — no Isaac Sim import — so they
unit-test without launching the simulator). Each returns a bool tensor of shape (num_envs,).

Scene entities expected (names configurable):
  * ``peg``                 RigidObject (held asset)
  * ``socket``              RigidObject/Articulation (kinematic fixed asset)
  * ``robot``               Franka articulation (last two joints = gripper fingers)
  * ``peg_finger_contact``  ContactSensor on the peg, filtered against the two fingers
  * ``peg_contact``         ContactSensor on the peg (net contact force)

Geometry conventions (meters), all overridable via params and externalized in
``configs/vdash/task.yaml``:
  * peg tip in the peg's local frame = ``tip_offset`` (default origin; peg root ≈ tip, see M1)
  * socket mouth is ``mouth_height`` above the socket root; bore axis is the socket's vertical axis
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------------------
# small torch helpers (no isaaclab dependency)
# --------------------------------------------------------------------------------------
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (w, x, y, z). Batched over leading dims."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _peg_tip_w(env, peg_name: str, tip_offset) -> torch.Tensor:
    peg = env.scene[peg_name]
    p = peg.data.root_pos_w
    q = peg.data.root_quat_w
    off = torch.as_tensor(tip_offset, device=p.device, dtype=p.dtype).expand_as(p)
    return p + _quat_rotate(q, off)


def _hole_frame(env, socket_name: str, mouth_height: float):
    """Return (bore_axis_xy [N,2], mouth_z [N]) in world coords."""
    s = env.scene[socket_name]
    sp = s.data.root_pos_w
    return sp[:, :2], sp[:, 2] + mouth_height


def _filtered_force_norm(env, sensor_name: str) -> torch.Tensor:
    """Sum of |filtered contact force| per env (e.g. peg↔fingers). force_matrix_w: (N,B,M,3)."""
    fm = env.scene[sensor_name].data.force_matrix_w
    return torch.norm(fm, dim=-1).flatten(start_dim=1).sum(dim=1)


def _net_force_norm(env, sensor_name: str) -> torch.Tensor:
    """Norm of the net contact force on the sensor body per env. net_forces_w: (N,B,3)."""
    nf = env.scene[sensor_name].data.net_forces_w
    return torch.norm(nf, dim=-1).flatten(start_dim=1).sum(dim=1)


def gripper_width(env, robot_name: str = "robot") -> torch.Tensor:
    """Total gripper opening = |finger_joint_1| + |finger_joint_2| (last two robot joints)."""
    jp = env.scene[robot_name].data.joint_pos
    return jp[:, -1].abs() + jp[:, -2].abs()


# --------------------------------------------------------------------------------------
# §3.4 predicates
# --------------------------------------------------------------------------------------
def grasped(
    env,
    *,
    robot_name: str = "robot",
    peg_finger_sensor: str = "peg_finger_contact",
    width_min: float = 0.005,
    width_max: float = 0.025,
    grasp_force: float = 1.0,
) -> torch.Tensor:
    """gripper width in the closed range ∧ peg–finger contact force > threshold."""
    w = gripper_width(env, robot_name)
    in_close = (w > width_min) & (w < width_max)
    f = _filtered_force_norm(env, peg_finger_sensor)
    return in_close & (f > grasp_force)


def handoff(
    env,
    *,
    peg_name: str = "peg",
    socket_name: str = "socket",
    robot_name: str = "robot",
    peg_finger_sensor: str = "peg_finger_contact",
    tip_offset=(0.0, 0.0, 0.0),
    mouth_height: float = 0.030,
    r_h: float = 0.015,
    h_low: float = 0.020,
    h_high: float = 0.060,
    speed_max: float = 0.05,
    width_min: float = 0.005,
    width_max: float = 0.025,
    grasp_force: float = 1.0,
) -> torch.Tensor:
    """grasped ∧ peg tip inside the handoff cylinder (radius r_h, height h_low..h_high above the
    mouth) ∧ peg speed < speed_max."""
    g = grasped(
        env, robot_name=robot_name, peg_finger_sensor=peg_finger_sensor,
        width_min=width_min, width_max=width_max, grasp_force=grasp_force,
    )
    tip = _peg_tip_w(env, peg_name, tip_offset)
    axis_xy, mouth_z = _hole_frame(env, socket_name, mouth_height)
    lateral = torch.norm(tip[:, :2] - axis_xy, dim=-1)
    height = tip[:, 2] - mouth_z
    in_cyl = (lateral < r_h) & (height > h_low) & (height < h_high)
    speed = torch.norm(env.scene[peg_name].data.root_lin_vel_w, dim=-1)
    return g & in_cyl & (speed < speed_max)


def inserted(
    env,
    *,
    peg_name: str = "peg",
    socket_name: str = "socket",
    clearance: float = 0.002,
    tip_offset=(0.0, 0.0, 0.0),
    mouth_height: float = 0.030,
    min_depth: float = 0.015,
) -> torch.Tensor:
    """SUCCESS: insertion depth (mouth_z − tip_z) > min_depth ∧ lateral offset < clearance c."""
    tip = _peg_tip_w(env, peg_name, tip_offset)
    axis_xy, mouth_z = _hole_frame(env, socket_name, mouth_height)
    depth = mouth_z - tip[:, 2]
    lateral = torch.norm(tip[:, :2] - axis_xy, dim=-1)
    return (depth > min_depth) & (lateral < clearance)


def force_violation(env, *, peg_sensor: str = "peg_contact", force_max: float = 50.0) -> torch.Tensor:
    """Contact-force norm on the peg exceeds the limit (the logger counts occurrences)."""
    return _net_force_norm(env, peg_sensor) > force_max


def insertion_failed(
    env,
    *,
    peg_sensor: str = "peg_contact",
    force_max: float = 50.0,
) -> torch.Tensor:
    """Force-exceed component of insertion failure. The timeout component is tracked by the
    insertion controller / episode length (it needs the insertion-phase entry time)."""
    return force_violation(env, peg_sensor=peg_sensor, force_max=force_max)
