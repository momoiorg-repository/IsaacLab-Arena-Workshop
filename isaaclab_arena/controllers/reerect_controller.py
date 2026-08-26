# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""spec §4-2: stand a side-lying workpiece up so the vertical chuck can accept it.

**The number this exists to fix.** A side-lying part is gripped ACROSS its axis, 30 mm from its
leading end (``pick.grip_station_side``). Insertion can only reach ``station - 10.9 mm`` before the
fingers arrive at the chuck face, i.e. **19.1 mm** against a commanded depth of 40-72 mm. The part
is held in a way that makes the hole unreachable, so no amount of better servoing loads it: the
grip station itself has to change, and a station cannot be changed without letting go. Measured
before this leg existed: 20 of 20 side-lying episodes ended with the part never entering the bore.

**What the arm does.** Rotate the held part 90 deg in the air, set it down STANDING on the bench,
release, and pick it again -- at which point it is an ordinary upright part and every downstream
leg (touch-off, transport, insertion, creep) is the already-measured upright path, unchanged.

**§9 compliance, and it is the reason the trajectory looks the way it does.** The ruling permits
only "既知の幾何量（治具pose・突き当て位置・指令把持オフセット）のみから実行前に確定し、フェーズ/
時間のみの関数として進む開ループ軌道", with the CLOSE-time latched pose as its reference. So:

* the target orientation is computed ONCE, from the quaternion latched when the fingers closed, and
  from the commanded grip offset -- never from the part's live pose while it is held;
* the turn is a slerp in a phase counter, not a servo: nothing it reads can change where it ends;
* the stand-down point is the SURVEYED station pose from the config, not a place found by looking.

After the release the part is no longer held, so the re-pick tracks it exactly as the first pick
does. That is not a re-entry of the forbidden feedback -- the prohibition is on re-deriving the
wrist target from a HELD object, whose pose is a function of the arm's own output.

**Why the V-block is not used.** The groove was assumed to be what makes the turn safe, by holding
the part while the grip station changes. Verified numerically instead
(``scripts/bdash/verify_reerect.py``): the part is a cantilever of at most 0.043 N.m about the grip
while it turns, against 0.34 N.m of friction capacity -- an **8x margin at the worst variant**, so
the turn is safe at the station the part is already held at and the intermediate set-down the groove
would provide buys nothing. The swept disc is 100-152 mm across and the block is 80x60 mm, so the
turn has to happen clear of the groove regardless. The groove remains in the scene as a fixture.
"""

from __future__ import annotations

import math
import torch

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

from isaaclab_arena.controllers.ee_control import ee_pose_action, read_ee_pose

# phase codes
TRANSIT, TURN, LOWER, RELEASE, RETRACT, DONE = range(6)
PHASE_NAMES = ("transit", "turn", "lower", "release", "retract", "done")


class ReerectController:
    """Turn a held side-lying part upright and set it down. ``step`` returns ``(action, grip, finished)``.

    ``grip`` is the commanded gripper column (1 = closed) and is meaningful only where the caller
    hands this controller the arm; everything is masked by ``active`` so a passive env accumulates
    no state.
    """

    def __init__(self, names: dict, station_pose, ee_cfg: dict, cfg: dict):
        self.names = names
        self.ee_cfg = ee_cfg
        self.cfg = cfg
        self.station_pose = tuple(float(v) for v in station_pose)
        self.turn_steps = int(cfg.get("turn_steps", 60))
        self.release_steps = int(cfg.get("release_steps", 12))
        self.settle_steps = int(cfg.get("settle_steps", 20))
        self.place_gap = float(cfg.get("place_gap", 0.003))
        # Height of the face the part is stood ON, measured from the bench top (z = 0). Not zero:
        # the arm cannot reach the bench with the tool horizontal -- measured, the hand bottoms out
        # 13.9-16.6 mm short at every wrist azimuth, with joint travel to spare -- so the part is set
        # down on the 仮置き台 instead. See assets.yaml reerect_pad.
        self.place_height = float(cfg.get("place_height", 0.040))
        self.turn_height = float(cfg.get("turn_height", 0.200))
        self.arrive_tol = float(cfg.get("arrive_tol", 0.015))
        self.travel_step = float(cfg.get("travel_step", 0.040))
        self.lower_step = float(cfg.get("lower_step", 0.010))
        # Per-phase budget. Without one, a phase whose exit test never fires spins until the
        # episode times out and the record says only "time_out" -- which is how the first version
        # of this leg burned 997 steps with nothing to show for it. With one, the leg gives up and
        # says WHERE, and the episode ends on the real failure instead of the clock.
        self.phase_budget = int(cfg.get("phase_budget", 400))
        # THE FREE YAW, and it is what makes the turn reachable at all. Once the part is vertical it
        # is a cylinder about that axis, so spinning the wrist around it leaves the part's final
        # standing pose bit-identical while putting the ARM in a completely different configuration.
        # Leaving it unused is what made the leg run 42 minutes without finishing an episode: an arm
        # pinned against a joint limit makes the solver crawl.
        #
        # It is spent on a WORLD AZIMUTH, not a fixed relative angle, because the measurement says
        # the arm cares about where the hand ends up pointing in the cell and not about how far it
        # turned to get there. Same probe, 300 steps per pose, joint margin at convergence:
        #
        #   grasp psi   relative yaw 180        relative yaw 90
        #   -90 deg     ok, margin 0.979        FAIL (jammed, 19.9 deg left)
        #   -60 deg     ok, margin 0.909        FAIL
        #   -30 deg     ok, margin 0.652        ok, margin 0.631
        #     0 deg     FAIL (19.6 deg left)    ok, margin 0.993
        #
        # A constant relative yaw therefore cannot work -- 180 is right at psi = -90 and wrong at
        # psi = 0. Re-read as world azimuth (the tool's approach direction after the turn is the
        # grasp axis, so azimuth = psi + yaw) every one of those rows agrees: 60-150 deg reachable,
        # -90 / 0 / 180 jammed. So the config names the azimuth and the yaw is solved for it.
        band = cfg.get("turn_azimuth_band_deg", [60.0, 150.0])
        self.azimuth_band = (math.radians(float(band[0])), math.radians(float(band[1])))
        # Angular RATE, not a step count. The slew is not a constant: aiming the hand into the band
        # adds a spin on top of the 90 deg turn, and how much depends on the grasp angle -- measured
        # 90 deg at psi = +90 rising to 180 deg at psi = -90. Holding the step count fixed therefore
        # doubles the angular speed for exactly the grasps that need the biggest move, and the part
        # is held by friction. Measured with a fixed 60 steps: a 171 deg slew threw the part 0.35 m
        # in 11 steps and the episode aborted on contact force with an empty gripper.
        #
        # 1.5 deg/step is the rate the 90-deg-in-60-steps case ran at, which is the one that did not
        # throw anything.
        self.turn_rate = math.radians(float(cfg.get("turn_rate_deg_per_step", 1.5)))
        self.azimuth_samples = int(cfg.get("azimuth_samples", 31))

        self.phase: torch.Tensor | None = None
        self.timer: torch.Tensor | None = None
        self.q_start: torch.Tensor | None = None
        self.q_end: torch.Tensor | None = None
        self.station: torch.Tensor | None = None
        self.gave_up: torch.Tensor | None = None
        self.turn_len: torch.Tensor | None = None
        self.stuck_phase: torch.Tensor | None = None

    # ------------------------------------------------------------------ state
    def _ensure(self, tcp: torch.Tensor):
        n, dev = tcp.shape[0], tcp.device
        if self.phase is None or self.phase.shape[0] != n:
            self.phase = torch.zeros(n, dtype=torch.long, device=dev)
            self.timer = torch.zeros(n, dtype=torch.long, device=dev)
            self.q_start = torch.zeros(n, 4, device=dev, dtype=tcp.dtype)
            self.q_start[:, 0] = 1.0
            self.q_end = self.q_start.clone()
            self.station = torch.zeros(n, device=dev, dtype=tcp.dtype)
            self.gave_up = torch.zeros(n, dtype=torch.bool, device=dev)
            self.stuck_phase = torch.full((n,), -1, dtype=torch.long, device=dev)
            self.turn_len = torch.full((n,), max(self.turn_steps, 1), dtype=torch.long, device=dev)

    def reset(self, env_ids=None):
        if self.phase is None:
            return
        if env_ids is None:
            self.phase.zero_()
            self.timer.zero_()
            self.gave_up.zero_()
            self.stuck_phase.fill_(-1)
            return
        self.phase[env_ids] = 0
        self.timer[env_ids] = 0
        self.gave_up[env_ids] = False
        self.stuck_phase[env_ids] = -1

    # ------------------------------------------------------------------ geometry
    def plan(self, latched_quat: torch.Tensor, axis: torch.Tensor, station: torch.Tensor, arm: torch.Tensor):
        """Fix the turn's endpoints ONCE, from quantities known before the turn starts.

        ``latched_quat`` is the CLOSE-time wrist pose (the ruling's reference), ``axis`` the
        leading->trailing direction latched with it, ``station`` the COMMANDED grip offset, and
        ``arm`` selects the envs to plan for.

        Which way to turn is not free. ``grasp_quat_from_axis`` canonicalises the axis sign into the
        +x half-plane, so the wrist's local +x is the part's axis up to a sign, and that sign is
        exactly what says whether the leading end is the end that must finish pointing DOWN. Turning
        the wrong way stands the part on its trailing end -- which for W-B and W-C is the FLANGE, a
        face the bore cannot accept at all.

        A rotation of ``theta`` about the world vector ``y_e`` (the finger-closing axis, horizontal
        while the part lies flat) carries ``x_e`` to ``cos(theta) x_e + sin(theta) z``. The leading
        end is ``-x_e`` when the axis was not flipped and ``+x_e`` when it was, so the turn that
        puts it along ``-z`` is ``+90 deg`` in the first case and ``-90 deg`` in the second.
        """
        self._ensure(latched_quat)
        q_end = self._aim(self.turned_quat(latched_quat, axis), latched_quat)
        sel = arm.unsqueeze(-1)
        self.q_start = torch.where(sel, latched_quat, self.q_start)
        self.q_end = torch.where(sel, q_end, self.q_end)
        self.station = torch.where(arm, station, self.station)
        # How long the turn takes is DERIVED from how far it has to go, so the angular rate is the
        # same for every grasp angle. A fixed step count silently speeds up the long turns.
        dot = (latched_quat * q_end).sum(dim=-1).abs().clamp(max=1.0)
        slew = 2.0 * torch.acos(dot)
        steps = torch.ceil(slew / self.turn_rate).clamp(min=1.0).long()
        self.turn_len = torch.where(arm, steps, self.turn_len)

    def plan_refix(self, latched_quat: torch.Tensor, place_station: torch.Tensor, arm: torch.Tensor):
        """Plan a PLACE-ONLY pass: no turn, no aim -- set the upright part down and let go.

        The refix route (spec §5). The part is already upright; what is wrong is WHERE ALONG ITS
        AXIS the fingers hold it, and that cannot be corrected while holding on. Standing it on the
        仮置き台 and re-grasping at the commanded station resets the grip to process-nominal --
        CP1b measured that pass at 100% stood / 100% correct-end / 0.00 deg over 50 episodes.

        ``place_station`` is the axial distance from TCP to the part's bottom face, and the caller
        passes the MEASURED value where touch-off produced one: with a grip error e the commanded
        station is wrong by exactly e, and a place height computed from it would press the part e
        into the pad or drop it e onto it. The measurement is what makes the set-down gentle.

        Same phase machine as the re-erect, with the turn degenerate: q_end = q_start, zero-length
        TURN. Everything else -- transit, lower, release, settle, retract, give-up budgets -- is the
        measured leg, unchanged.
        """
        self._ensure(latched_quat)
        sel = arm.unsqueeze(-1)
        self.q_start = torch.where(sel, latched_quat, self.q_start)
        self.q_end = torch.where(sel, latched_quat, self.q_end)
        self.station = torch.where(arm, place_station, self.station)
        self.turn_len = torch.where(arm, torch.zeros_like(self.turn_len), self.turn_len)

    def turned_quat(self, latched_quat: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """The PURE 90 deg turn about the closing axis, before the free spin is aimed.

        Split out so the turn and the aim can be checked separately: the turn is what stands the
        part up and must not move the grip, the aim is what makes the pose reachable and necessarily
        does move it.
        """
        local_x = torch.zeros_like(axis)
        local_x[:, 0] = 1.0
        local_y = torch.zeros_like(axis)
        local_y[:, 1] = 1.0
        x_e = quat_apply(latched_quat, local_x)
        y_e = quat_apply(latched_quat, local_y)
        # +1 when the wrist's +x already points at the TRAILING end (leading end is -x_e).
        sign = torch.where((x_e * axis).sum(dim=-1) >= 0.0, 1.0, -1.0)
        theta = sign * (torch.pi / 2.0)
        q_turn = quat_from_angle_axis(theta, y_e)
        # Pre-multiplication: `ee_pose_action` applies `q_des = dq (x) q_cur`, so a world-frame
        # delta composes on the left. `y_e` is already a WORLD vector, which is what makes this a
        # rotation about the finger axis rather than about a body axis of the same name.
        return quat_mul(q_turn, latched_quat)

    def _aim(self, q_end: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """Spin about the part's own axis (world z once it is vertical) to the configured azimuth.

        Applied on the left for the same reason as the turn: ``ee_pose_action`` composes world-frame
        deltas by pre-multiplication. The tool's approach axis is the wrist's local z; where the turn
        alone leaves it is READ rather than derived, so the sign of the turn needs no second case.
        """
        local_z = torch.zeros_like(q_end[:, :3])
        local_z[:, 2] = 1.0
        approach = quat_apply(q_end, local_z)
        alpha = torch.atan2(approach[:, 1], approach[:, 0])
        # The point of the reachable band that costs the LEAST TOTAL SLEW, found by evaluating the
        # band rather than by picking its nearest edge.
        #
        # Nearest-edge is the obvious choice and it is wrong. The wrist ends at a composite of two
        # rotations about different axes -- the 90 deg turn about the finger axis, then this spin
        # about world z -- and the composite is not monotone in the spin: minimising the spin does
        # not minimise the slew, and at psi = -90 the nearest edge still lands on a full 180 deg,
        # where a slerp has no shortest arc at all and the wrist takes an undefined path.
        #
        # Any azimuth in the band grips a cylinder equally and all of them are reachable, so there
        # is nothing to trade off against slew: evaluate and take the argmin. Deterministic, and a
        # function of the CLOSE-time pose alone, so the trajectory stays fixed before it starts.
        lo, hi = self.azimuth_band
        betas = torch.linspace(lo, hi, self.azimuth_samples, device=q_end.device, dtype=q_end.dtype)
        world_z = torch.zeros_like(local_z)
        world_z[:, 2] = 1.0
        best = q_end
        best_slew = torch.full_like(alpha, float("inf"))
        for beta in betas:
            delta = torch.remainder(beta - alpha + math.pi, 2.0 * math.pi) - math.pi
            cand = quat_mul(quat_from_angle_axis(delta, world_z), q_end)
            slew = (start * cand).sum(dim=-1).abs().clamp(max=1.0).acos() * 2.0
            take = (slew < best_slew).unsqueeze(-1)
            best = torch.where(take, cand, best)
            best_slew = torch.where(slew < best_slew, slew, best_slew)
        return best

    def turn_t(self, phase: torch.Tensor, timer: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        """Turn progress in [0,1] as a function of PHASE AND TIMER ONLY -- the §9 open-loop clock.

        Pulled out as a pure function because getting it wrong is invisible and was in fact wrong:
        ``timer`` counts WITHIN a phase and restarts at every transition, so using it directly makes
        the turn's parameter drop back to 0 the instant TURN ends. The wrist then commands
        ``q_start`` again and rotates the part flat WHILE the arm is descending to set it down.

        Measured with that defect in place: IK residual ran to 96-349 mm and the part was flung --
        4 of 4 episodes ended with it back beside the tray and still horizontal (95.9, 89.5, 97.4
        and 90.0 deg from vertical). It has to SATURATE past TURN, not restart.
        """
        length = (torch.full_like(timer, max(self.turn_steps, 1)) if self.turn_len is None else self.turn_len).to(dtype)
        t = (timer.to(dtype) / length.clamp(min=1.0)).clamp(max=1.0)
        t = torch.where(phase > TURN, torch.ones_like(t), t)
        return torch.where(phase < TURN, torch.zeros_like(t), t)

    def _slerp(self, t: torch.Tensor) -> torch.Tensor:
        """Shortest-arc interpolation from ``q_start`` to ``q_end`` at ``t`` in [0,1]."""
        q0, q1 = self.q_start, self.q_end
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)  # shortest arc; q and -q are the same rotation
        dot = dot.abs().clamp(max=1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        t = t.unsqueeze(-1)
        # Near-parallel endpoints make sin(omega) vanish; fall back to lerp, which is exact there.
        near = sin_omega < 1e-6
        a = torch.where(near, 1.0 - t, torch.sin((1.0 - t) * omega) / sin_omega.clamp(min=1e-9))
        b = torch.where(near, t, torch.sin(t * omega) / sin_omega.clamp(min=1e-9))
        q = a * q0 + b * q1
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    # ------------------------------------------------------------------ act
    def step(self, env, active: torch.Tensor):
        """Return ``(action, want_open, finished)``. State advances only where ``active``.

        ``want_open`` is the (N,) bool the caller routes through ``_take_gripper``; this controller
        never writes the gripper column itself, because §9's owner rule is what caught the last
        three defects in this file's sibling.
        """
        tcp, _ = read_ee_pose(env)
        self._ensure(tcp)
        dtype = tcp.dtype
        phase = self.phase

        sx, sy, _ = self.station_pose
        # TRANSIT and RETRACT share this point: the part is carried over the set-down spot at turn
        # height BEFORE it is turned, so the swept disc (100-152 mm across, measured per variant in
        # verify_reerect.py) turns over bare bench rather than over the tray it just came out of.
        p_clear = torch.zeros_like(tcp)
        p_clear[:, 0] = sx
        p_clear[:, 1] = sy
        p_clear[:, 2] = self.turn_height

        # LOWER/RELEASE: once the part is vertical its leading end hangs `station` below the TCP, so
        # the TCP stops that far above the face it is being set on, plus a release gap. The bench top
        # is z = 0 by construction (build_workbench: "z = 0 is the top, not the centre"), and
        # `place_height` raises the target to the 仮置き台's top face -- which is not a nicety, see
        # its definition.
        p_place = p_clear.clone()
        p_place[:, 2] = self.place_height + self.station + self.place_gap

        placing = (phase == LOWER) | (phase == RELEASE)
        target = torch.where(placing.unsqueeze(-1), p_place, p_clear)

        quat_cmd = self._slerp(self.turn_t(phase, self.timer, dtype))

        want_open = phase >= RELEASE

        # Two speeds. The transit is up to ~0.5 m and the set-down must be slow or the part is
        # driven into the bench before the arrival test can fire. Built as two actions and selected
        # per env, because `max_pos_step` is a scalar: collapsing a per-env tensor with `.max()`
        # silently gives every env the fastest one, which is how the touch-off leg travelled at
        # probe speed for 850 steps.
        # BY KEYWORD, and the argument is `gripper_open` -- TRUE opens. It was first passed a local
        # named `closed`, which inverted it: the fingers opened the instant the leg took over and the
        # part fell out on the FIRST step of the turn. Every later symptom followed from that, and
        # none of them named it -- the leg went on turning an empty hand through its whole open-loop
        # trajectory, which is indistinguishable from a working turn in every log line except the
        # gripper's own width (measured: 27.8 mm opening to 76.5 mm over nine steps).
        act_fast = ee_pose_action(
            env, target, quat_cmd, gripper_open=want_open, cfg=self.ee_cfg, max_pos_step=self.travel_step
        )
        act_slow = ee_pose_action(
            env, target, quat_cmd, gripper_open=want_open, cfg=self.ee_cfg, max_pos_step=self.lower_step
        )
        action = torch.where((phase == LOWER).unsqueeze(-1), act_slow, act_fast)

        # -------------------------------------------------------------- advance
        at_clear = (p_clear - tcp).norm(dim=-1) < self.arrive_tol
        at_place = (p_place[:, 2] - tcp[:, 2]).abs() < self.arrive_tol
        turn_done = self.timer >= self.turn_len.clamp(min=1)
        # RELEASE holds still for the fingers to open AND for the part to stop moving. Retracting
        # while it is still settling drags it over with the finger that has not cleared yet.
        released = self.timer >= (self.release_steps + self.settle_steps)

        advance = (
            ((phase == TRANSIT) & at_clear)
            | ((phase == TURN) & turn_done)
            | ((phase == LOWER) & at_place)
            | ((phase == RELEASE) & released)
            | ((phase == RETRACT) & at_clear)
        ) & active

        # The timer counts WITHIN a phase, so it restarts on every transition; TURN and RELEASE are
        # the two phases timed rather than triggered by arrival.
        # Give up rather than spin. A phase that outruns its budget records WHICH phase it was --
        # the one fact that separates "the arm could not get there" from "the exit test is wrong" --
        # and the leg reports finished so the episode ends on its real outcome rather than on the
        # clock. The part is left wherever it is; the insertion that follows then fails honestly.
        stuck = active & ~advance & (phase < DONE) & (self.timer >= self.phase_budget)
        self.gave_up = self.gave_up | stuck
        self.stuck_phase = torch.where(stuck & (self.stuck_phase < 0), phase, self.stuck_phase)

        self.timer = torch.where(advance, torch.zeros_like(self.timer), self.timer + active.long())
        self.phase = torch.where(advance & (phase < DONE), phase + 1, phase)
        return action, want_open, (self.phase >= DONE) | self.gave_up

    # ------------------------------------------------------------------ report
    def stats(self) -> dict:
        if self.phase is None:
            return {}
        out = {"reerect_phase": PHASE_NAMES[int(self.phase[0])]}
        if self.gave_up is not None and bool(self.gave_up[0]):
            out["reerect_gave_up"] = True
            out["reerect_stuck_in"] = PHASE_NAMES[int(self.stuck_phase[0])]
        return out
