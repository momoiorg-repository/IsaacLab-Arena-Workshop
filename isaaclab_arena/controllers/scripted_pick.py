# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted pick / transport expert (brief §3.3 — stand-in for the VLA front-end).

A *privileged* state machine: it reads the ground-truth peg/socket poses from the scene (the VLA's
perception job is out of scope here) and drives the Franka through
``approach → descend → grasp → lift → transport`` to hover the grasped peg over the socket, where the
:class:`~isaaclab_arena.controllers.insertion_controller.InsertionController` takes over. Vectorized
over envs. Orientation is held at the arm's ready (down-pointing) pose, captured on the first step
after reset.
"""

from __future__ import annotations

import os
import torch

from isaaclab.utils.math import quat_apply

from isaaclab_arena.controllers.ee_control import ee_pose_action, read_ee_pose

# phase codes
APPROACH, DESCEND, CLOSE, LIFT, TRANSPORT, DONE = range(6)


class ScriptedPick:
    def __init__(self, names: dict, ee_cfg: dict, pick_cfg: dict):
        self.names = names
        self.ee_cfg = ee_cfg
        self.cfg = pick_cfg
        self.phase = None  # lazily sized (num_envs,)
        # M8: if set (N,3), use this perception-estimated grasp point instead of the ground-truth peg
        # pose — the rule-based baseline's grasp must not read the peg's true pose (brief §2.1).
        self.grasp_override = None

    # ------------------------------------------------------------------ state
    def _ensure(self, env):
        if self.phase is not None:
            return
        n, dev = env.unwrapped.num_envs, env.unwrapped.device
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.timer = torch.zeros(n, dtype=torch.long, device=dev)
        self.q_hold = torch.zeros(n, 4, device=dev)
        self.q_hold[:, 0] = 1.0
        self.need_capture = torch.ones(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None):
        if self.phase is None:
            return
        idx = slice(None) if env_ids is None else env_ids
        self.phase[idx] = APPROACH
        self.timer[idx] = 0
        self.need_capture[idx] = True

    # --------------------------------------------------------------- geometry
    def _grasp_point(self, env) -> torch.Tensor:
        if self.grasp_override is not None:  # M8: perception estimate (no peg ground truth, §2.1)
            return self.grasp_override
        peg = env.unwrapped.scene[self.names["peg"]].data
        off = torch.zeros_like(peg.root_pos_w)
        off[:, 2] = self.cfg["grip_offset"]
        gp = peg.root_pos_w + quat_apply(peg.root_quat_w, off)
        # TEST-BED (diagnostic, default off): BDASH_GRASP_OFF_MM shifts the grasp target laterally so the
        # gripper closes off-centre on the grip cube -> the peg is held off the gripper axis. This
        # reproduces the VLA's in-gripper grasp error WITHOUT the VLA, to measure the blind-channel
        # response R(g) (see docs/bdash_precision_budget_plan.md). Fixed +x direction for determinism;
        # the realized in-gripper lateral/tilt is read out via BDASH_GRASP_DIAG.
        off_mm = float(os.environ.get("BDASH_GRASP_OFF_MM", "0") or "0")
        if off_mm != 0.0:
            gp = gp.clone()
            gp[:, 0] = gp[:, 0] + off_mm / 1000.0
        return gp

    def _socket_xy(self, env) -> torch.Tensor:
        return env.unwrapped.scene[self.names["socket"]].data.root_pos_w[:, :2]

    # ------------------------------------------------------------------ step
    def step(self, env):
        """Return (action (N,7), finished (N,) bool)."""
        self._ensure(env)
        cap = self.need_capture
        if cap.any():
            # Grasp straight down at a FIXED yaw (no peg-yaw alignment). The cylindrical grip is
            # yaw-symmetric, so wrist yaw is irrelevant to the grasp; holding a constant wrist keeps the
            # recorded demos position-only — no peg-yaw -> wrist coupling for the VLA to have to perceive
            # (that coupling, from the old square-peg yaw-align, made the data unlearnable -> 0% grasp).
            q_down0 = torch.zeros_like(self.q_hold)
            q_down0[:, 1] = 1.0  # (w,x,y,z)=(0,1,0,0): gripper approach axis points exactly world -z
            # TEST-BED (default off): BDASH_GRASP_OFF_DEG tilts the grasp approach to induce an in-gripper
            # cock (for the tilt-fix study); realized in-gripper tilt is read out via BDASH_GRASP_DIAG.
            deg = float(os.environ.get("BDASH_GRASP_OFF_DEG", "0") or "0")
            if deg != 0.0:
                from isaaclab.utils.math import quat_from_angle_axis, quat_mul

                ax = torch.zeros(q_down0.shape[0], 3, device=q_down0.device)
                ax[:, 0] = 1.0  # tilt about world-x
                ang = torch.full((q_down0.shape[0],), deg * 3.141592653589793 / 180.0, device=q_down0.device)
                q_down0 = quat_mul(quat_from_angle_axis(ang, ax), q_down0)
            self.q_hold = torch.where(cap.unsqueeze(-1), q_down0, self.q_hold)
            self.need_capture = self.need_capture & ~cap

        gp = self._grasp_point(env)
        above = gp.clone()
        above[:, 2] += self.cfg["approach_height"]
        lift = gp.clone()
        lift[:, 2] = self.cfg["lift_height"]
        transport = torch.cat([self._socket_xy(env), lift[:, 2:3]], dim=-1)

        phase = self.phase
        # active target per phase
        tgt = above.clone()
        tgt = torch.where((phase == DESCEND).unsqueeze(-1) | (phase == CLOSE).unsqueeze(-1), gp, tgt)
        tgt = torch.where((phase == LIFT).unsqueeze(-1), lift, tgt)
        tgt = torch.where((phase >= TRANSPORT).unsqueeze(-1), transport, tgt)

        gripper_open = phase <= DESCEND

        # tolerances per phase
        tol = torch.full_like(phase, 0, dtype=torch.float32)
        tol = torch.where(phase == APPROACH, self.cfg["settle_tol"], tol)
        tol = torch.where(phase == DESCEND, self.cfg["grasp_tol"], tol)
        tol = torch.where(phase == CLOSE, self.cfg["grasp_tol"], tol)
        tol = torch.where(phase == LIFT, self.cfg["settle_tol"], tol)
        tol = torch.where(phase >= TRANSPORT, self.cfg["settle_tol"], tol)
        cur_pos, _ = read_ee_pose(env)
        reached = torch.norm(tgt - cur_pos, dim=-1) < tol

        # transitions
        self.timer += 1
        to1 = (phase == APPROACH) & reached
        to2 = (phase == DESCEND) & reached
        to3 = (phase == CLOSE) & (self.timer >= self.cfg["close_steps"])
        to4 = (phase == LIFT) & reached
        to5 = (phase == TRANSPORT) & reached
        self.phase = torch.where(to1, torch.full_like(phase, DESCEND), self.phase)
        self.phase = torch.where(to2, torch.full_like(phase, CLOSE), self.phase)
        self.timer = torch.where(to2, torch.zeros_like(self.timer), self.timer)  # restart for CLOSE
        self.phase = torch.where(to3, torch.full_like(phase, LIFT), self.phase)
        self.phase = torch.where(to4, torch.full_like(phase, TRANSPORT), self.phase)
        self.phase = torch.where(to5, torch.full_like(phase, DONE), self.phase)

        action = ee_pose_action(env, tgt, self.q_hold, gripper_open, self.ee_cfg)
        finished = self.phase >= DONE
        return action, finished
