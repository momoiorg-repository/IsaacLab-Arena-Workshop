# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Rule-based peg insertion controller (brief §3.3): force-limited press + spiral search.

Takes over from the scripted pick once the grasped peg is hovering over the socket. Phases:

  * **SETTLE** — bring the peg tip a few mm above the mouth on the socket axis and dwell until the
    EE has settled (speed below a threshold). Killing the approach momentum here is the cheapest way
    to avoid the rim-impact force spike, and it makes the §3.4 ``handoff`` predicate fire reliably.
  * **PRESS** — drive the tip target down under a **low-force servo** (track a small Fz target read
    from the wrist-F/T model), running an Archimedean **spiral search** in xy while the tip is loaded
    against the mouth face but not yet captured, plus an optional **RCC** lateral/rotational
    compliance that turns offset rim contact into slip-into-bore (Whitney, 1982).
  * **RELEASE** — once the tip is seated below the mouth, open the gripper and retract.

§2.1 compliance: the controller's only inputs are the **socket pose** (known, jig-fixed fixture
assumption), **EE proprioception** (the ``ee_frame`` sensor → a nominal-grasp tip estimate, see
:func:`~isaaclab_arena.controllers.ee_control.tip_estimate`), and a **wrist-F/T sensor model** (the
resultant of the peg contact sensor, :func:`_wrench`). It never reads the peg's ground-truth pose —
that is forbidden for the insertion controller (brief §2.1) and is used only by the predicates / §3.5
logger. Control is *tip-relative*: because the peg is rigidly grasped, moving the EE by Δ moves the
tip by Δ, so the EE target is ``cur_EE + (tip_target − tip_est)``. Vectorized; state updates are
gated by an ``active`` mask so the controller only runs after handoff.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from isaaclab_arena.controllers.ee_control import ee_pose_action, level_to_down, read_ee_pose, tip_estimate

# phase codes
SETTLE, PRESS, RELEASE, DONE = range(4)


def _wrench(env, sensor_name: str) -> torch.Tensor:
    """Wrist-F/T sensor model (brief §2.1): world-frame resultant contact force on the peg, (N,3).

    This synthetic wrench (the peg contact-sensor resultant) is the *only* contact signal the
    insertion controller may use; the peg pose is off-limits. With a downward press, ``Fz > 0`` is the
    upward reaction (how hard we are pressing), and ``Fxy`` is the lateral binding / offset force."""
    nf = env.unwrapped.scene[sensor_name].data.net_forces_w  # (N,B,3)
    return nf.sum(dim=1)


class InsertionController:
    def __init__(self, names: dict, mouth_height: float, ee_cfg: dict, ins_cfg: dict, grip_offset: float):
        self.names = names
        self.mouth_height = mouth_height
        self.ee_cfg = ee_cfg
        self.cfg = ins_cfg
        self.grip_offset = grip_offset
        self.phase = None
        # M7 (E4 convergence-zone harness): hold the peg tip at a controlled lateral offset + tilt and
        # press straight down (no spiral / retry / re-center) to map the funnel capture region.
        self.m7_active = False
        self.m7_offset = None    # (N,2) lateral offset from the socket axis (m)
        self.m7_tilt_vec = None  # (N,2) lean = θ(rad) · (cos az, sin az); peg leans along this dir

    def set_m7(self, offset_xy: torch.Tensor, tilt_vec: torch.Tensor):
        """Enable the M7 straight-press mode with per-env offset (m) and lean (rad·dir)."""
        self.m7_active = True
        self.m7_offset = offset_xy
        self.m7_tilt_vec = tilt_vec

    def _ensure(self, env):
        if self.phase is not None:
            return
        n, dev = env.unwrapped.num_envs, env.unwrapped.device
        self.step_dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 60.0))
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.timer = torch.zeros(n, dtype=torch.long, device=dev)
        self.z_press = torch.full((n,), 10.0, device=dev)
        self.spiral_r = torch.zeros(n, device=dev)
        self.spiral_th = torch.zeros(n, device=dev)
        self.best_depth = torch.full((n,), -1.0, device=dev)   # deepest tip reached this attempt
        self.stuck_timer = torch.zeros(n, dtype=torch.long, device=dev)
        self.attempts = torch.zeros(n, dtype=torch.long, device=dev)
        self.q_hold = torch.zeros(n, 4, device=dev)
        self.q_hold[:, 0] = 1.0
        self.need_capture = torch.ones(n, dtype=torch.bool, device=dev)
        self.prev_tip = torch.zeros(n, 3, device=dev)
        self.have_prev = False
        self.gave_up = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None):
        if self.phase is None:
            return
        idx = slice(None) if env_ids is None else env_ids
        self.phase[idx] = SETTLE
        self.timer[idx] = 0
        self.z_press[idx] = 10.0
        self.spiral_r[idx] = 0.0
        self.spiral_th[idx] = 0.0
        self.best_depth[idx] = -1.0
        self.stuck_timer[idx] = 0
        self.attempts[idx] = 0
        self.need_capture[idx] = True
        self.gave_up[idx] = False
        self.have_prev = False

    def step(self, env, active: torch.Tensor):
        """Return the (N,7) action; only mutate state where ``active`` (post-handoff envs)."""
        self._ensure(env)
        c = self.cfg
        dt = self.step_dt

        tip = tip_estimate(env, self.grip_offset)        # proprioceptive (§2.1), not peg pose
        cur_ee, cur_quat = read_ee_pose(env)

        # capture the straight-down hold orientation once (commanded/proprioceptive, not peg pose)
        cap = self.need_capture & active
        if cap.any():
            q_down = level_to_down(cur_quat)
            self.q_hold = torch.where(cap.unsqueeze(-1), q_down, self.q_hold)
            self.need_capture = self.need_capture & ~cap

        sock = env.unwrapped.scene[self.names["socket"]].data
        axis_xy = sock.root_pos_w[:, :2]                  # socket pose: allowed (known fixture, §2.1)
        mouth_z = sock.root_pos_w[:, 2] + self.mouth_height
        depth = mouth_z - tip[:, 2]
        m7 = self.m7_active
        # In M7 the controller works the peg toward a held offset point (axis + offset); normally it
        # works toward the socket axis. Lateral error and all xy targets are taken w.r.t. this center.
        center_xy = axis_xy + self.m7_offset if m7 else axis_xy
        lateral = torch.norm(tip[:, :2] - center_xy, dim=-1)

        # EE settling speed: finite-difference of the (proprioceptive) tip estimate
        if not self.have_prev:
            self.prev_tip = tip.clone()
            self.have_prev = True
        speed = torch.norm(tip - self.prev_tip, dim=-1) / max(dt, 1e-6)
        self.prev_tip = torch.where(active.unsqueeze(-1), tip, self.prev_tip)

        # wrist-F/T model
        W = _wrench(env, self.names["peg_sensor"])
        if c.get("ft_noise_std", 0.0) > 0.0:
            W = W + torch.randn_like(W) * c["ft_noise_std"]
        fz = W[:, 2].clamp(min=0.0)                        # upward press reaction
        fxy = W[:, :2]
        fxy_mag = torch.norm(fxy, dim=-1)
        f_mag = torch.norm(W, dim=-1)

        phase = self.phase
        captured = depth > c.get("capture_depth", 0.003)

        # ---------------- commanded tip target ----------------
        settle_z = mouth_z + c.get("settle_height", 0.005)
        press_entry_z = mouth_z + c["approach_above_mouth"]
        z_floor = mouth_z - c["press_target_depth"]

        z_tip = torch.where(phase == SETTLE, settle_z, self.z_press)
        z_tip = torch.where(phase == RELEASE, mouth_z + 0.10, z_tip)

        spiral_xy = torch.stack(
            [self.spiral_r * torch.cos(self.spiral_th), self.spiral_r * torch.sin(self.spiral_th)], dim=-1
        )
        # RCC translational compliance: nudge the xy target along the lateral contact force, so an
        # offset tip is actively walked toward the bore centre (on top of the funnel geometry).
        # In M7 the controller works around the *held offset* (center_xy = axis + offset) — i.e. it
        # spirals/servos from the handoff point — so the map = how far the funnel+spiral recover.
        rcc_xy = c.get("rcc_xy_gain", 0.0) * fxy
        press_xy = center_xy + spiral_xy + rcc_xy
        xy_tip = torch.where((phase == PRESS).unsqueeze(-1), press_xy, center_xy)
        tip_target = torch.cat([xy_tip, z_tip.unsqueeze(-1)], dim=-1)

        # tip-relative EE target (peg rigidly grasped: ΔEE == Δtip)
        ee_target = cur_ee + (tip_target - tip)

        # RCC rotational compliance: small rotation of the held orientation about a horizontal axis so
        # the tip leans toward the lateral force — converts offset rim contact into slip-into-bore.
        q_cmd = self.q_hold
        if m7:
            # M7: lean the peg by θ along the offset azimuth (m7_tilt_vec = θ·(cos az, sin az)); the
            # tilt axis is perpendicular to the lean dir so the downward peg axis tips toward +dir.
            theta = torch.norm(self.m7_tilt_vec, dim=-1)
            d = self.m7_tilt_vec / theta.clamp(min=1e-9).unsqueeze(-1)
            axis = torch.zeros_like(tip)
            axis[:, 0] = d[:, 1]
            axis[:, 1] = -d[:, 0]
            axis[:, 2] = torch.where(theta < 1e-6, torch.ones_like(theta), torch.zeros_like(theta))
            q_cmd = quat_mul(quat_from_angle_axis(theta, axis), self.q_hold)
        rot_gain = c.get("rcc_rot_gain", 0.0)
        if rot_gain > 0.0 and not m7:
            fxy3 = torch.cat([fxy, torch.zeros_like(fxy[:, :1])], dim=-1)
            zaxis = torch.zeros_like(fxy3)
            zaxis[:, 2] = 1.0
            rax = torch.cross(fxy3, zaxis, dim=-1)
            rn = torch.norm(rax, dim=-1, keepdim=True)
            rax = rax / rn.clamp(min=1e-9)
            angle = (rot_gain * fxy_mag).clamp(max=c.get("rcc_rot_max", 0.10))
            do_rot = (phase == PRESS) & ~captured & (fxy_mag > 1e-3)
            angle = torch.where(do_rot, angle, torch.zeros_like(angle))
            dq = quat_from_angle_axis(angle, rax)
            q_cmd = quat_mul(dq, self.q_hold)

        gripper_open = phase >= RELEASE
        # move gently near/inside the mouth; the (short) settle descent may stay a touch faster
        near = depth > -0.025
        fine = (phase == PRESS) | ((phase == SETTLE) & near)
        step_cap = c.get("fine_max_pos_step", self.ee_cfg["max_pos_step"])
        cap_val = step_cap if bool(fine.any()) else None
        action = ee_pose_action(env, ee_target, q_cmd, gripper_open, self.ee_cfg, max_pos_step=cap_val)

        # ---------------- state updates (gated by active) ----------------
        a = active
        self.timer = torch.where(a, self.timer + 1, self.timer)

        # ---- SETTLE -> PRESS: tip centred just above the mouth and (optionally) settled ----
        align_xy_tol = c.get("align_xy_tol", 0.004)
        settle_z_tol = c.get("settle_z_tol", 0.004)
        z_ok = (tip[:, 2] - settle_z).abs() < settle_z_tol
        if bool(c.get("use_settle", True)):
            ready = (
                (lateral < align_xy_tol) & z_ok
                & (speed < c.get("settle_speed", 0.03))
                & (self.timer >= int(c.get("settle_min_steps", 8)))
            )
        else:
            ready = (lateral < align_xy_tol) & ((tip[:, 2] - settle_z).abs() < c.get("settle_z_tol_fast", 0.006))
        to_press = a & (phase == SETTLE) & ready
        self.z_press = torch.where(to_press, press_entry_z, self.z_press)
        self.spiral_r = torch.where(to_press, torch.zeros_like(self.spiral_r), self.spiral_r)
        self.spiral_th = torch.where(to_press, torch.zeros_like(self.spiral_th), self.spiral_th)
        self.best_depth = torch.where(to_press, torch.full_like(self.best_depth, -1.0), self.best_depth)
        self.stuck_timer = torch.where(to_press, torch.zeros_like(self.stuck_timer), self.stuck_timer)
        self.timer = torch.where(to_press, torch.zeros_like(self.timer), self.timer)
        self.phase = torch.where(to_press, torch.full_like(phase, PRESS), self.phase)
        phase = self.phase

        # ---- PRESS: force-limited descent (servo or two-stage band) ----
        in_press = a & (phase == PRESS)
        if bool(c.get("use_force_servo", True)):
            # servo the press load Fz to a small target: descent speed ∝ (f_target − Fz), so the tip
            # eases onto the face at a few N and only advances as the bore lets it. Drive home faster
            # once captured (bore walls constrain the peg).
            rate = (c.get("f_gain", 0.002) * (c.get("f_target", 4.0) - fz)).clamp(
                min=-c.get("back_speed", 0.010), max=c.get("contact_speed", 0.008)
            )
            seated = captured & (fz < c.get("f_target_seated", 15.0))
            rate = torch.where(seated, torch.full_like(rate, c.get("seated_speed", 0.020)), rate)
            self.z_press = torch.where(in_press, torch.clamp(self.z_press - rate * dt, min=z_floor), self.z_press)
        else:
            pstep = torch.where(captured, c.get("press_step_seated", 0.003), c["press_step"])
            pband = torch.where(captured, c.get("press_force_band_seated", 15.0), c["press_force_band"])
            descend = in_press & (f_mag < pband)
            self.z_press = torch.where(descend, torch.clamp(self.z_press - pstep, min=z_floor), self.z_press)
            back = in_press & (f_mag > c.get("back_off_force", 10.0)) & ~captured
            self.z_press = torch.where(back, self.z_press + c.get("back_off_step", 0.0015), self.z_press)

        # spiral search only while still searching for the bore (loaded, not captured); around the
        # held center (axis, or axis+offset in M7) so the tip drops from the funnel cone into the bore.
        spin = in_press & ~captured & (fz > c["spiral_engage_force"]) & (depth < c["seated_depth"])
        self.spiral_r = torch.where(
            spin, torch.clamp(self.spiral_r + c["spiral_radius_rate"], max=c["spiral_radius_max"]), self.spiral_r
        )
        self.spiral_th = torch.where(spin, self.spiral_th + c["spiral_omega"], self.spiral_th)

        # progress / stuck tracking: reset the stuck timer whenever the tip reaches a new depth
        progressed = in_press & (depth > self.best_depth + 0.001)
        self.best_depth = torch.where(progressed, depth, self.best_depth)
        self.stuck_timer = torch.where(
            progressed, torch.zeros_like(self.stuck_timer),
            torch.where(in_press, self.stuck_timer + 1, self.stuck_timer),
        )

        # retry-on-jam (F/T based, §2.1): no depth progress for too long, OR persistent high lateral
        # binding before capture (the compliant stand-in for the old peg-tilt trigger).
        retry_after = int(c.get("retry_after_steps", 150))
        max_retries = int(c.get("max_retries", 5))
        binding = in_press & (fxy_mag > c.get("jam_force", 12.0)) & ~captured
        retry = in_press & ((self.stuck_timer > retry_after) | binding) & (self.attempts < max_retries)
        self.attempts = torch.where(retry, self.attempts + 1, self.attempts)
        self.spiral_r = torch.where(retry, torch.zeros_like(self.spiral_r), self.spiral_r)
        self.best_depth = torch.where(retry, torch.full_like(self.best_depth, -1.0), self.best_depth)
        self.stuck_timer = torch.where(retry, torch.zeros_like(self.stuck_timer), self.stuck_timer)
        self.timer = torch.where(retry, torch.zeros_like(self.timer), self.timer)
        self.phase = torch.where(retry, torch.full_like(phase, SETTLE), self.phase)
        phase = self.phase

        # PRESS -> RELEASE once seated
        to_release = a & (phase == PRESS) & (depth > c["seated_depth"])
        self.timer = torch.where(to_release, torch.zeros_like(self.timer), self.timer)
        self.phase = torch.where(to_release, torch.full_like(phase, RELEASE), self.phase)
        phase = self.phase

        # RELEASE -> DONE after a few steps
        to_done = a & (phase == RELEASE) & (self.timer >= c["release_steps"])
        self.phase = torch.where(to_done, torch.full_like(phase, DONE), self.phase)

        # gave-up flag for the §3.5 logger (attempts exhausted, never seated) -> insertion_failed
        self.gave_up = (self.attempts >= max_retries) & (self.phase < RELEASE)

        return action
