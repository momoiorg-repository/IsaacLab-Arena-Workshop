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

from isaaclab.utils.math import quat_apply, quat_error_magnitude

from isaaclab_arena.controllers.ee_control import ee_pose_action, read_ee_pose


def _clamp_action_norm(v, max_norm):
    n = v.norm(dim=-1, keepdim=True)
    return v * (max_norm / n.clamp(min=1e-9)).clamp(max=1.0)


# phase codes
APPROACH, DESCEND, CLOSE, LIFT, TRANSPORT, DONE = range(6)


class ScriptedPick:
    def __init__(self, names: dict, ee_cfg: dict, pick_cfg: dict):
        self.names = names
        self.ee_cfg = ee_cfg
        self.cfg = pick_cfg
        self.phase = None  # lazily sized (num_envs,)
        # (N,) bool or None -- envs whose arm another controller now drives, so this one's progress
        # watchdog must stand down. None (the peg path and the pick stage) leaves it inert.
        self.watchdog_suppress: torch.Tensor | None = None
        # M8: if set (N,3), use this perception-estimated grasp point instead of the ground-truth peg
        # pose — the rule-based baseline's grasp must not read the peg's true pose (brief §2.1).
        self.grasp_override = None
        # B-DASH chuck load: if set (N,4 wxyz), grasp at this orientation instead of the fixed
        # down-quat, so the wrist can align across a cylinder lying on its side. Leaving it None is
        # the peg path and is bit-for-bit the frozen v6 behaviour — including the reached test
        # below, which only gains its orientation term when this override is present.
        self.grasp_quat_override = None

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
        # --- stall watchdog (chuck path only; see step()) --------------------------------------
        # A PROGRESS watchdog, not a time watchdog, mirroring insertion_controller.py:582-641: the
        # counter only advances while the best distance to the waypoint fails to improve. `timer`
        # cannot serve this role because it counts from episode reset, not from phase entry.
        self.best_dist = torch.full((n,), float("inf"), device=dev)
        self.stall_timer = torch.zeros(n, dtype=torch.long, device=dev)
        self.attempts = torch.zeros(n, dtype=torch.long, device=dev)
        self.gave_up = torch.zeros(n, dtype=torch.bool, device=dev)
        self.empty_close = torch.zeros(n, dtype=torch.bool, device=dev)
        # Snapshot of the FIRST stall, kept because the counters alone say "the watchdog fired" but
        # not what stopped the arm. The TCP height at the stall is the discriminating measurement:
        # `neighbour_top - clear_z` predicted the pre-§0-10 stalls to +-2 mm, so if the residual
        # stalls still land on that line the hand-on-neighbour mechanism survives the guarantee,
        # and if they do not the cause is elsewhere. NaN = never stalled.
        self.stall_tcp_z = torch.full((n,), float("nan"), device=dev)
        self.stall_phase = torch.full((n,), -1, dtype=torch.long, device=dev)

    def reset(self, env_ids=None):
        if self.phase is None:
            return
        idx = slice(None) if env_ids is None else env_ids
        self.phase[idx] = APPROACH
        self.timer[idx] = 0
        self.need_capture[idx] = True
        self.best_dist[idx] = float("inf")
        self.stall_timer[idx] = 0
        self.attempts[idx] = 0
        self.gave_up[idx] = False
        self.empty_close[idx] = False
        self.stall_tcp_z[idx] = float("nan")
        self.stall_phase[idx] = -1

    # --------------------------------------------------------------- geometry
    def _grasp_point(self, env) -> torch.Tensor:
        if self.grasp_override is not None:  # M8: perception estimate (no peg ground truth, §2.1)
            # TEST-BED, default off. The override path is what the CHUCK task uses, so without this
            # the existing BDASH_GRASP_OFF_MM knob below is unreachable there -- and a knob that
            # silently does nothing is worse than no knob. Injecting a KNOWN grasp error is the only
            # way to ask whether measuring the held part (touch_off) beats assuming it: the scripted
            # teacher's own grasps are near-perfect (~0.1 deg in-gripper), so with no error injected
            # the measured and assumed values agree by construction and the comparison is vacuous.
            off_mm = float(os.environ.get("BDASH_GRASP_OFF_MM", "0") or "0")
            if off_mm != 0.0:
                shifted = self.grasp_override.clone()
                shifted[:, 0] = shifted[:, 0] + off_mm / 1000.0
                return shifted
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

    def _grasp_quat(self, env) -> torch.Tensor:
        """Commanded EE orientation. Symmetric with :meth:`_grasp_point`; ``q_hold`` is the v6 path."""
        if self.grasp_quat_override is not None:
            return self.grasp_quat_override
        return self.q_hold

    def _socket_xy(self, env) -> torch.Tensor:
        return env.unwrapped.scene[self.names["socket"]].data.root_pos_w[:, :2]

    def _waypoints(self, env, gp: torch.Tensor):
        """(above, lift, transport) for a grasp point ``gp``. Subclasses retarget the sequence."""
        above = gp.clone()
        above[:, 2] += self.cfg["approach_height"]
        lift = gp.clone()
        lift[:, 2] = self.cfg["lift_height"]
        transport = torch.cat([self._socket_xy(env), lift[:, 2:3]], dim=-1)
        return above, lift, transport

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
        above, lift, transport = self._waypoints(env, gp)
        q_cmd = self._grasp_quat(env)

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
        cur_pos, cur_quat = read_ee_pose(env)
        dist = torch.norm(tgt - cur_pos, dim=-1)
        reached = dist < tol
        if self.grasp_quat_override is not None:
            # Only gate on orientation when the wrist actually has to slew. A side-lying grasp can
            # ask for up to 90 deg of yaw at max_rot_step = 0.30 rad/step, so without this the
            # fingers would still be turning as they descend onto the part. The peg path keeps the
            # original position-only test, so its frozen results cannot move.
            reached = reached & (quat_error_magnitude(q_cmd, cur_quat) < self.cfg.get("rot_tol", 0.06))

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

        if self.grasp_quat_override is not None:
            # Every phase entry re-arms the progress baseline, so each phase gets its own budget.
            #
            # `to1` (APPROACH -> DESCEND) MUST be in here. Leaving it out turned the DESCEND
            # watchdog into a fixed 60-step deadline instead of the progress watchdog this class
            # documents: `to1` fires when the hover waypoint is `reached` at settle_tol, so
            # `best_dist` is <= 0.02 at that instant, while DESCEND's target is a full
            # `approach_height` (0.12 m) away. `progressed` then stays false for the whole descent --
            # the counter runs from its first step -- until `dist` drops back under 0.02 near the
            # very end. Measured with it missing: retries fired in 52/320 episodes, all 36 failures
            # had retried at least once, and 268/268 episodes that never retried succeeded.
            entered = to1 | to2 | to3 | to4 | to5
            self._stall_watchdog(env, phase, dist, to3, entered, cur_pos)

        action = ee_pose_action(env, tgt, q_cmd, gripper_open, self.ee_cfg)
        # LEARNING-DATA MODE (BDASH_PICK_SLOW_STEP, meters/step; default off = production speed).
        # Cap ONLY the approach/descend/close phases: at the production 0.08 m/step the whole
        # grasp window spans ~20-30 frames, which starved the VLA of supervision exactly where
        # precision is needed (measured: lift247 model reaches the part but topples it). LIFT and
        # TRANSPORT keep production speed -- the slow window is per env, so a vectorized recording
        # can have one env creeping onto a part while another transports at full speed.
        slow = float(os.environ.get("BDASH_PICK_SLOW_STEP", "0") or "0")
        if slow > 0.0:
            rows = (phase <= CLOSE).unsqueeze(-1)
            cap = slow / float(self.ee_cfg["action_scale"])
            action = torch.where(
                rows, torch.cat([_clamp_action_norm(action[:, :3], cap), action[:, 3:]], dim=-1), action
            )
        finished = self.phase >= DONE
        return action, finished

    # ------------------------------------------------------------ stall watchdog
    def _stall_watchdog(self, env, phase, dist, to3, entered, cur_pos):
        """Retreat and retry when a phase stops making progress, or when CLOSE grips nothing.

        Chuck path only — it is called under ``grasp_quat_override is not None`` so the peg path's
        transition table is bit-for-bit unchanged and the frozen v4/v6 results cannot move.

        Why this is needed: ``to1``/``to2``/``to4``/``to5`` gate purely on ``reached`` with no time
        bound, so a descent that cannot converge sits until the episode ends. Measured, the arm
        froze with the TCP identical to four decimals for 245 steps while the hand rested on a
        neighbouring part — every such block became a whole lost episode.

        The counter is a PROGRESS watchdog (``insertion_controller.py:582-641``): it only advances
        while the best distance achieved in this attempt fails to improve, so a slow-but-converging
        approach is never interrupted.
        """
        cfg = self.cfg
        stall_eps = float(cfg.get("stall_eps", 0.001))
        stall_after = int(cfg.get("stall_after_steps", 60))
        max_retries = int(cfg.get("max_retries", 3))

        # Empty-close detection at the moment CLOSE completes. `min_grip_width` had been dead config
        # in both tasks (zero Python references); this is where it earns its keep. The width is
        # computed with the same expression as bdash_chuck_predicates.gripper_width -- read here
        # directly rather than importing the predicates, which must stay sim-free.
        min_w = float(cfg.get("min_grip_width", 0.0))
        if min_w > 0.0 and bool(to3.any()):
            jp = env.unwrapped.scene[self.names["robot"]].data.joint_pos
            width = jp[:, -1].abs() + jp[:, -2].abs()
            self.empty_close = self.empty_close | (to3 & (width < min_w))

        # progress tracking, armed per attempt; CLOSE is time-based so it is exempt
        tracking = phase != CLOSE
        if self.watchdog_suppress is not None:
            # Envs that have been handed to another controller. Their arm is no longer following
            # THIS controller's waypoint, so "the distance stopped improving" is not a stall -- it
            # is the handoff. Left running, the watchdog retries a pick that has already succeeded
            # and drops `phase` back to APPROACH while the part is held, which makes `finished()`
            # unreachable and corrupts the retry diagnostic. Measured on the first load-stage smoke
            # test: retries hit max_retries in 3 of 3 episodes, including the two that seated.
            tracking = tracking & ~self.watchdog_suppress
        progressed = tracking & (dist < self.best_dist - stall_eps)
        self.best_dist = torch.where(progressed, dist, self.best_dist)
        self.stall_timer = torch.where(
            progressed,
            torch.zeros_like(self.stall_timer),
            torch.where(tracking, self.stall_timer + 1, self.stall_timer),
        )

        stalled = tracking & (self.stall_timer > stall_after)

        # Latch the FIRST stall only: `stalled` is true for a single step per attempt (the retry
        # below zeroes the timer), and the first one is the least contaminated by the retreat.
        first_stall = stalled & torch.isnan(self.stall_tcp_z)
        self.stall_tcp_z = torch.where(first_stall, cur_pos[:, 2], self.stall_tcp_z)
        self.stall_phase = torch.where(first_stall, phase, self.stall_phase)

        retry = (stalled | self.empty_close) & (self.attempts < max_retries)

        # retreat to the APPROACH waypoint (`above`), the same shape as the insertion controller
        # falling back to SETTLE: the retreat is implicit in the phase's own target.
        self.attempts = torch.where(retry, self.attempts + 1, self.attempts)
        self.phase = torch.where(retry, torch.full_like(self.phase, APPROACH), self.phase)
        self.timer = torch.where(retry, torch.zeros_like(self.timer), self.timer)
        self.stall_timer = torch.where(retry, torch.zeros_like(self.stall_timer), self.stall_timer)
        self.best_dist = torch.where(retry, torch.full_like(self.best_dist, float("inf")), self.best_dist)
        self.empty_close = self.empty_close & ~retry

        self.best_dist = torch.where(entered, torch.full_like(self.best_dist, float("inf")), self.best_dist)
        self.stall_timer = torch.where(entered, torch.zeros_like(self.stall_timer), self.stall_timer)

        self.gave_up = (self.attempts >= max_retries) & (self.phase < LIFT) & (stalled | self.empty_close)
