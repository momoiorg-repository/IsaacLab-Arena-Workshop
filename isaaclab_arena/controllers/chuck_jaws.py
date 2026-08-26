# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The chuck's own actuator: drives the three jaws radially in and out (spec §4-3's 閉爪).

This is cell equipment, not the robot. On the real machine ``chuck_load`` fires the chuck after the
part is presented; here the same command drives the jaws directly, which is why it lives beside the
arm controllers rather than inside the policy.

**Kinematic, not an articulation.** ``assets.yaml`` builds the jaws as three separate kinematic
rigid bodies placed at 120 deg (``bdash_chuck_load_environment._place_jaws``); the mesh has its
gripping face at local x=0 and extends toward +x, so a jaw at radius ``r`` puts its face exactly
``r`` from the bore axis. Closing is therefore a radial pose write, which is what a kinematic body
is for -- it is driven, it does not respond. spec §2.1 asks for prismatic synchronised drive; this
is that drive without the URDF, and it reproduces what matters here (the part is captured and stops
being the gripper's problem). What it cannot reproduce is grip FORCE, so nothing downstream may
read a clamping force.

Why the part has to be captured at all: the bore is a THROUGH hole. Until the jaws hold it, opening
the gripper drops the part straight through. That is also why W-C needs the jaws -- its Ø45 flange
seats on the chuck face at 72 mm of insertion, but the fingers cannot follow it below the face
(they span radius 12.5-38.7 mm against a bore radius of 20 mm), so the last stretch has to happen
with the part released and guided by the bore.
"""

from __future__ import annotations

import math
import torch


class ChuckJaws:
    """Radial jaw drive. ``command(env, close_mask)`` steps every env toward its commanded radius."""

    def __init__(self, jaw_names, chuck_xy, jaw_z: float, radius_closed: float, cfg: dict):
        self.jaw_names = tuple(jaw_names)
        self.chuck_xy = chuck_xy
        self.jaw_z = float(jaw_z)
        self.radius_closed = float(radius_closed)
        self.radius_open = self.radius_closed + float(cfg.get("open_radius_extra", 0.010))
        self.rate = float(cfg.get("close_rate", 0.0006))
        self.close_steps = int(cfg.get("close_steps", 20))
        self.radius: torch.Tensor | None = None  # (N,) current commanded radius
        self._clamped: torch.Tensor | None = None
        self._clamped_pose: torch.Tensor | None = None
        self._angles = [k * 2.0 * math.pi / max(len(self.jaw_names), 1) for k in range(len(self.jaw_names))]

    def reset(self, env_ids=None) -> None:
        if self.radius is None:
            return
        idx = slice(None) if env_ids is None else env_ids
        self.radius[idx] = self.radius_open
        if self._clamped is not None:
            self._clamped[idx] = False

    @property
    def closed(self) -> torch.Tensor | None:
        """(N,) bool -- the jaws have completed their close stroke. Machine state, not privileged."""
        if self.radius is None:
            return None
        return self.radius <= self.radius_closed + 1e-6

    def command(self, env, close_mask: torch.Tensor) -> None:
        """Move each env's jaws one step toward closed (where ``close_mask``) or open (elsewhere)."""
        uenv = env.unwrapped
        if self.radius is None:
            self.radius = torch.full((uenv.num_envs,), self.radius_open, device=uenv.device)

        target = torch.where(close_mask, self.radius_closed, self.radius_open)
        step = torch.clamp(target - self.radius, min=-self.rate, max=self.rate)
        self.radius = self.radius + step
        self._write(uenv)

    def _write(self, uenv) -> None:
        origins = uenv.scene.env_origins  # (N,3)
        for name, angle in zip(self.jaw_names, self._angles):
            asset = uenv.scene[name]
            pose = asset.data.root_state_w[:, :7].clone()
            pose[:, 0] = origins[:, 0] + self.chuck_xy[0] + self.radius * math.cos(angle)
            pose[:, 1] = origins[:, 1] + self.chuck_xy[1] + self.radius * math.sin(angle)
            pose[:, 2] = origins[:, 2] + self.jaw_z
            asset.write_root_pose_to_sim(pose)

    def clamp(self, env, workpiece_names, target_idx, mask) -> None:
        """Hold the clamped workpiece rigid in the chuck -- the ``fixed joint`` half of the design.

        ``assets.yaml`` describes the jaws as "KINEMATIC (close + fixed joint)": the close is the
        radial drive above, and this is the joint. Without it the jaws are geometry that touches the
        part and nothing more -- they carry no grip force, so opening the gripper drops the part
        straight through the bore. Measured with the joint missing: W-B ended at 83 mm of insertion
        against an 80 mm bore, i.e. out the bottom and onto the table, while `seated` still passed
        it because that predicate only had a lower bound.

        Implemented as "freeze the pose, zero the velocity" rather than a USD joint. For a part held
        by a fixture that is itself static, the two are the same constraint, and this one needs no
        stage surgery mid-episode. What it does NOT model is slip under load: a real chuck can be
        overcome, this cannot. Nothing downstream may read a clamping force (see the module note on
        the kinematic jaws being a conservative, force-free model).

        The pose is captured ONCE, on the step the clamp closes, so QC measures the part where the
        chuck took it -- which is what spec §2.3 asks for ("閉爪+サイクルスタート発火後にQC").
        """
        uenv = env.unwrapped
        if self._clamped_pose is None or self._clamped_pose.shape[0] != uenv.num_envs:
            self._clamped_pose = torch.zeros(uenv.num_envs, 7, device=uenv.device)
            self._clamped = torch.zeros(uenv.num_envs, dtype=torch.bool, device=uenv.device)
        newly = mask & ~self._clamped
        self._clamped = self._clamped | mask

        for k, name in enumerate(workpiece_names):
            is_target = (target_idx == k) & self._clamped
            if not bool(is_target.any()):
                continue
            asset = uenv.scene[name]
            pose = asset.data.root_state_w[:, :7].clone()
            capture = (target_idx == k) & newly
            self._clamped_pose = torch.where(capture.unsqueeze(-1), pose, self._clamped_pose)
            asset.write_root_pose_to_sim(torch.where(is_target.unsqueeze(-1), self._clamped_pose, pose))
            vel = asset.data.root_state_w[:, 7:].clone()
            asset.write_root_velocity_to_sim(torch.where(is_target.unsqueeze(-1), torch.zeros_like(vel), vel))
