# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
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

import os
import torch

from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from isaaclab_arena.controllers.ee_control import ee_pose_action, level_to_down, read_ee_pose, tip_estimate

# phase codes (CALIBRATE is prepended for the tactile probe; renumbers the codes but every comparison
# below is name-based so behavior is unchanged when the probe is off and the start phase stays SETTLE)
CALIBRATE, SETTLE, PRESS, RELEASE, DONE = range(5)


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
        # DIAGNOSTIC ORACLE (default off, VIOLATES §2.1): feed the controller the peg's TRUE tip so its
        # assumed tip == true tip. This is the M11 "cheat" upper bound that proves the blind in-gripper
        # grasp error is the binding constraint (8 mm offset: 0/20 -> 11/20). NOT a proposed solution.
        self._cheat_tip = bool(os.environ.get("VDASH_CHEAT_TIP"))
        # Estimator-noise study (default 0): add a frozen per-episode Gaussian LATERAL error (mm) to the
        # oracle correction. Sweeping it characterizes R_widened(σ) — how accurate a §2.1 grasp-pose
        # estimate must be to recover success — and sets the spec for the tactile probe (Phase 2b).
        self._tip_noise_mm = float(os.environ.get("VDASH_TIP_NOISE_MM", "0") or "0")
        # §2.1-honest tactile CALIBRATE probe (default off): before inserting, actively touch a grid of
        # assumed-tip xy against the known socket fixture and read the wrist-F/T model to localize the
        # bore relative to the (blind) assumed tip -> estimate the in-gripper error e and correct
        # tip_estimate. Inputs are proprioception + socket pose + F/T only (no peg ground-truth).
        self._tactile_cal = bool(os.environ.get("VDASH_TACTILE_CAL"))
        self._cal_half_mm = float(os.environ.get("VDASH_CAL_HALF_MM", "12") or "12")  # grid half-extent (mm)
        self._cal_step_mm = float(os.environ.get("VDASH_CAL_STEP_MM", "6") or "6")  # grid spacing (mm)
        self._cal_safe_clear = float(os.environ.get("VDASH_CAL_SAFE_MM", "12") or "12") / 1000.0  # rise between touches
        self._cal_max_depth = (
            float(os.environ.get("VDASH_CAL_DEPTH_MM", "6") or "6") / 1000.0
        )  # probe floor below mouth
        self._cal_lead = float(os.environ.get("VDASH_CAL_LEAD_MM", "20") or "20") / 1000.0  # fast GOTO lead/step
        self._cal_desc_lead = float(os.environ.get("VDASH_CAL_DESCLEAD_MM", "6") or "6") / 1000.0  # gentle DESCEND lead
        #   (soft arm follows the lead with lag; GOTO travels fast, DESCEND is gentle for clean contact-z)
        self._cal_f_touch = float(
            os.environ.get("VDASH_CAL_FTOUCH_N", "2.0") or "2.0"
        )  # contact threshold (N over baseline)
        self._cal_xy_tol = float(os.environ.get("VDASH_CAL_XYTOL_MM", "2.0") or "2.0") / 1000.0
        self._cal_z_tol = 0.003
        self._cal_pen_margin = (
            float(os.environ.get("VDASH_CAL_PENMARGIN_MM", "2.0") or "2.0") / 1000.0
        )  # ignore funnel-lip noise
        # §2.1 EE de-cant (default off): estimate in-gripper TILT from the per-finger contact-force
        # z-asymmetry (gripper's own sensing, NOT peg pose — the net wrench cancels but the per-finger
        # moment doesn't) and rotate the straight-down hold so the cocked peg becomes vertical.
        self._decant = bool(os.environ.get("VDASH_DECANT"))
        self._decant_gain = float(os.environ.get("VDASH_DECANT_GAIN", "0.7") or "0.7")  # finger dz (N) per deg
        self._decant_sign = float(os.environ.get("VDASH_DECANT_SIGN", "1") or "1")  # +1 straightens (validated)
        # §2.1 observable tip correction from the de-cant cue (default off): estimate the in-gripper cock from
        # dz and shift the assumed tip by ~L*sin(phi) laterally — an observable approximation of the oracle
        # (VDASH_CHEAT_TIP) that recovered 0->11/20. Unlike de-cant (which rotates the hold and tilts the
        # press), this corrects the tip the controller servos to, keeping the press straight.
        self._gate_correct = bool(os.environ.get("VDASH_GATE_CORRECT"))
        self._gate_L_mm = float(os.environ.get("VDASH_GATE_L_MM", "60") or "60")  # grip->tip lever (mm)
        self._init_phase = CALIBRATE if self._tactile_cal else SETTLE
        self.phase = None
        # M7 (E4 convergence-zone harness): hold the peg tip at a controlled lateral offset + tilt and
        # press straight down (no spiral / retry / re-center) to map the funnel capture region.
        self.m7_active = False
        self.m7_offset = None  # (N,2) lateral offset from the socket axis (m)
        self.m7_tilt_vec = None  # (N,2) lean = θ(rad) · (cos az, sin az); peg leans along this dir
        # recover mode: deliver the peg to the offset+tilt at SETTLE (the perturbed handoff), then on
        # PRESS entry RELEASE the hold so the controller recenters on the true axis with q_hold + the
        # configured RCC/spiral compliance. This maps the *recovery* basin (what the controller can pull
        # back from a perturbed handoff) — the faithful E4 convergence zone, vs the held static-tolerance
        # map (recover=False). Default off → the held map and all non-m7 paths are unchanged.
        self.m7_recover = False

    def set_m7(self, offset_xy: torch.Tensor, tilt_vec: torch.Tensor, recover: bool = False):
        """Enable M7 with per-env offset (m) and lean (rad·dir). recover=True → release the hold at
        PRESS entry so the controller recenters on the true axis (recovery basin)."""
        self.m7_active = True
        self.m7_offset = offset_xy
        self.m7_tilt_vec = tilt_vec
        self.m7_recover = recover

    def _ensure(self, env):
        if self.phase is not None:
            return
        n, dev = env.unwrapped.num_envs, env.unwrapped.device
        self.step_dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 60.0))
        self.phase = torch.full((n,), self._init_phase, dtype=torch.long, device=dev)
        self.timer = torch.zeros(n, dtype=torch.long, device=dev)
        self.z_press = torch.full((n,), 10.0, device=dev)
        self.spiral_r = torch.zeros(n, device=dev)
        self.spiral_th = torch.zeros(n, device=dev)
        self.best_depth = torch.full((n,), -1.0, device=dev)  # deepest tip reached this attempt
        self.stuck_timer = torch.zeros(n, dtype=torch.long, device=dev)
        self.hop_count = torch.zeros(n, dtype=torch.long, device=dev)  # hop-and-search hops this attempt
        self.attempts = torch.zeros(n, dtype=torch.long, device=dev)
        self.q_hold = torch.zeros(n, 4, device=dev)
        self.q_hold[:, 0] = 1.0
        self.need_capture = torch.ones(n, dtype=torch.bool, device=dev)
        self.prev_tip = torch.zeros(n, 3, device=dev)
        self.have_prev = False
        self.gave_up = torch.zeros(n, dtype=torch.bool, device=dev)
        self.m7_released = torch.zeros(n, dtype=torch.bool, device=dev)  # recover mode: hold dropped at PRESS
        # §2.1-honest tip correction (m, world): added to the proprioceptive tip_estimate so the
        # controller servos the TRUE tip. Zero by default; filled by the tactile CALIBRATE probe
        # (Phase 2) or overwritten each step by the VDASH_CHEAT_TIP oracle. See precision-budget plan.
        self.tip_correction = torch.zeros(n, 3, device=dev)
        self._frozen_noise = torch.zeros(n, 3, device=dev)  # per-episode estimator error (m), frozen once
        self._noise_set = torch.zeros(n, dtype=torch.bool, device=dev)
        self._gatecorr_set = torch.zeros(n, dtype=torch.bool, device=dev)  # observable tip-corr frozen once/ep
        if self._tactile_cal:
            half = self._cal_half_mm / 1000.0
            step = self._cal_step_mm / 1000.0
            ncells = max(2, int(round(2.0 * half / step)) + 1)
            coords = torch.linspace(-half, half, ncells, device=dev)
            gx, gy = torch.meshgrid(coords, coords, indexing="ij")
            self.cal_grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (K,2) assumed-xy offsets (m)
            kk = self.cal_grid.shape[0]
            self.cal_idx = torch.zeros(n, dtype=torch.long, device=dev)  # current touch index
            self.cal_sub = torch.zeros(n, dtype=torch.long, device=dev)  # 0=GOTO (rise+centre), 1=DESCEND
            self.cal_zc = torch.zeros(n, kk, device=dev)  # assumed-tip z recorded at each contact
            self.cal_baseline = torch.zeros(n, device=dev)  # pre-touch wrench baseline (cocked grip)

    def reset(self, env_ids=None):
        if self.phase is None:
            return
        idx = slice(None) if env_ids is None else env_ids
        self.phase[idx] = self._init_phase
        self.timer[idx] = 0
        self.z_press[idx] = 10.0
        self.spiral_r[idx] = 0.0
        self.spiral_th[idx] = 0.0
        self.best_depth[idx] = -1.0
        self.stuck_timer[idx] = 0
        self.hop_count[idx] = 0
        self.attempts[idx] = 0
        self.need_capture[idx] = True
        self.gave_up[idx] = False
        if hasattr(self, "m7_released"):
            self.m7_released[idx] = False
        if hasattr(self, "tip_correction"):
            self.tip_correction[idx] = 0.0
        if hasattr(self, "_noise_set"):
            self._noise_set[idx] = False
            self._frozen_noise[idx] = 0.0
        if hasattr(self, "_gatecorr_set"):
            self._gatecorr_set[idx] = False
        if hasattr(self, "cal_idx"):
            self.cal_idx[idx] = 0
            self.cal_sub[idx] = 0
            self.cal_zc[idx] = 0.0
            self.cal_baseline[idx] = 0.0
        self.have_prev = False

    def _apply_decant(self, env, q_down):
        """§2.1 EE de-cant: estimate the in-gripper cock from the per-finger contact-force z-asymmetry and
        rotate the straight-down hold so the cocked peg becomes world-vertical. Uses only the gripper's
        own finger contact forces (peg_finger_contact), never the peg pose. Observable axis is
        perpendicular to the finger line (dz ∝ cock about x); cock about the finger line is force-blind
        (would need a 90° wrist-yaw re-measure). Returns the de-canted (N,4) hold quat."""
        try:
            fm = env.unwrapped.scene[self.names["peg_finger_sensor"]].data.force_matrix_w  # (N,B,M,3)
            ff = fm.sum(dim=1)  # (N,M,3) per-finger net force
            if ff.shape[1] < 2:
                return q_down
            dz = ff[:, 0, 2] - ff[:, 1, 2]  # (N,) per-finger z-asymmetry ∝ cock
        except Exception:  # noqa: BLE001
            return q_down
        beta = self._decant_sign * (dz / max(self._decant_gain, 1e-6)) * (3.141592653589793 / 180.0)  # rad
        ax = torch.zeros(q_down.shape[0], 3, device=q_down.device)
        ax[:, 0] = 1.0  # de-cant about x (the finger-observable axis)
        if os.environ.get("VDASH_CAL_DIAG"):
            print(f"[decant] dz={float(dz[0]):.2f}N beta_est={float(beta[0]) * 180 / 3.14159:.1f}deg", flush=True)
        return quat_mul(quat_from_angle_axis(beta, ax), q_down)

    def _calibrate_step(self, env, active, tip, cur_ee, axis_xy, mouth_z, f_mag, speed):
        """Vectorized active-touch CALIBRATE sub-FSM (§2.1-honest interface-widening).

        Probes a fixed grid of assumed-tip xy around the known socket axis, recording the assumed-tip z
        at contact (wrist-F/T over a per-touch baseline) at each grid point. The touch whose TRUE tip
        lands over the bore descends deepest, so a penetration-weighted centroid of the grid offsets
        locates the bore in assumed coordinates -> e_xy = -centroid; e_z comes from the flat-top
        plateau. On completion it writes self.tip_correction and transitions CALIBRATE -> SETTLE. Inputs
        are only the proprioceptive tip/EE, the known socket axis, and the F/T model (no peg pose)."""
        c = self.cfg
        K = self.cal_grid.shape[0]
        in_cal = (self.phase == CALIBRATE) & active
        safe_z = mouth_z + self._cal_safe_clear
        floor_z = mouth_z - self._cal_max_depth
        off = self.cal_grid[self.cal_idx.clamp(max=K - 1)]  # (N,2); clamp guards done envs (idx==K)
        probe_xy = axis_xy + off
        at_xy = torch.norm(tip[:, :2] - probe_xy, dim=-1) < self._cal_xy_tol
        risen = tip[:, 2] >= (safe_z - self._cal_z_tol)
        if os.environ.get("VDASH_CAL_DIAG") and bool(in_cal[0]):
            self._cal_dbg = getattr(self, "_cal_dbg", 0) + 1
            if self._cal_dbg % 25 == 0:
                print(
                    f"[calprog] idx={int(self.cal_idx[0])}/{K} sub={int(self.cal_sub[0])} "
                    f"tipz={float(tip[0, 2]):.4f} safez={float(safe_z[0]):.4f} at_xy={bool(at_xy[0])} "
                    f"risen={bool(risen[0])} fmag={float(f_mag[0]):.2f} base={float(self.cal_baseline[0]):.2f}",
                    flush=True,
                )
        # GOTO (sub 0) -> arm DESCEND once risen + centred; snapshot the pre-touch wrench baseline
        g2d = in_cal & (self.cal_sub == 0) & at_xy & risen
        self.cal_baseline = torch.where(g2d, f_mag, self.cal_baseline)
        self.cal_sub = torch.where(g2d, torch.ones_like(self.cal_sub), self.cal_sub)
        # DESCEND (sub 1) -> record assumed-tip z on contact (over baseline, slow) or at the probe floor
        contact = (f_mag - self.cal_baseline) > self._cal_f_touch
        reached_floor = tip[:, 2] <= floor_z
        rec = in_cal & (self.cal_sub == 1) & ((contact & (speed < c.get("settle_speed", 0.03))) | reached_floor)
        rec_i = rec.nonzero(as_tuple=True)[0]
        if rec_i.numel():
            self.cal_zc[rec_i, self.cal_idx[rec_i]] = tip[rec_i, 2]
        self.cal_idx = torch.where(rec, self.cal_idx + 1, self.cal_idx)
        self.cal_sub = torch.where(rec, torch.zeros_like(self.cal_sub), self.cal_sub)
        # SOLVE once all K touches are recorded
        done = in_cal & (self.cal_idx >= K)
        if bool(done.any()):
            plateau = self.cal_zc.max(dim=1).values  # flat-top contact z = mouth_z - e_z
            pen = (plateau.unsqueeze(1) - self.cal_zc).clamp(min=0.0)  # penetration below the plateau
            w = (pen - self._cal_pen_margin).clamp(min=0.0)  # drop funnel-lip noise
            wsum = w.sum(dim=1)
            centroid = (w.unsqueeze(-1) * self.cal_grid.unsqueeze(0)).sum(dim=1) / wsum.clamp(min=1e-9).unsqueeze(-1)
            corr = torch.zeros_like(self.tip_correction)
            corr[:, :2] = -centroid  # e_xy = -(assumed-coord bore centre)
            corr[:, 2] = mouth_z - plateau  # e_z from the plateau
            apply = done & (wsum > 1e-9)
            self.tip_correction = torch.where(apply.unsqueeze(-1), corr, self.tip_correction)
            if os.environ.get("VDASH_CAL_DIAG") and bool(done[0]):
                print(
                    f"[caldiag] env0 e_xy=({corr[0, 0] * 1000:.1f},{corr[0, 1] * 1000:.1f})mm "
                    f"e_z={corr[0, 2] * 1000:.1f}mm wsum={float(wsum[0]):.4f} valid={(wsum[0] > 1e-9).item()}",
                    flush=True,
                )
            self.phase = torch.where(done, torch.full_like(self.phase, SETTLE), self.phase)
            self.timer = torch.where(done, torch.zeros_like(self.timer), self.timer)
        # probe action: GOTO -> (probe_xy, safe_z); DESCEND -> (probe_xy, just below the floor)
        z_target = torch.where(self.cal_sub == 0, safe_z, floor_z - 0.005)
        cal_ee_target = cur_ee + (torch.cat([probe_xy, z_target.unsqueeze(-1)], dim=-1) - tip)
        grip_closed = torch.zeros(tip.shape[0], dtype=torch.bool, device=tip.device)
        # two-speed: fast lead for GOTO travel, gentle lead for the DESCEND touch (clean contact-z, no ram)
        act_fast = ee_pose_action(
            env, cal_ee_target, self.q_hold, grip_closed, self.ee_cfg, max_pos_step=self._cal_lead
        )
        act_slow = ee_pose_action(
            env, cal_ee_target, self.q_hold, grip_closed, self.ee_cfg, max_pos_step=self._cal_desc_lead
        )
        return torch.where((self.cal_sub == 1).unsqueeze(-1), act_slow, act_fast)

    def step(self, env, active: torch.Tensor):
        """Return the (N,7) action; only mutate state where ``active`` (post-handoff envs)."""
        self._ensure(env)
        c = self.cfg
        dt = self.step_dt

        tip = tip_estimate(env, self.grip_offset)  # proprioceptive (§2.1), not peg pose
        # DIAGNOSTIC ORACLE (VIOLATES §2.1, default off): overwrite the correction so assumed == true tip.
        if self._cheat_tip:
            true_tip = env.unwrapped.scene[self.names["peg"]].data.root_pos_w
            if self._tip_noise_mm > 0.0:
                fresh = active & ~self._noise_set
                if bool(fresh.any()):
                    nz = torch.randn(self._frozen_noise.shape, device=self._frozen_noise.device)
                    nz[:, 2] = 0.0  # model the estimator error on the lateral (xy) channel only
                    nz = nz * (self._tip_noise_mm / 1000.0)
                    self._frozen_noise = torch.where(fresh.unsqueeze(-1), nz, self._frozen_noise)
                    self._noise_set = self._noise_set | fresh
            self.tip_correction = (true_tip - tip) + self._frozen_noise
        # §2.1 observable tip correction from the de-cant cue (VDASH_GATE_CORRECT): a cock phi about the
        # finger-perp axis displaces the blind tip by ~L*sin(phi) laterally; estimate phi from the per-finger
        # force-z asymmetry (dz) and shift the assumed tip there (like the oracle, but observable). Frozen once.
        if self._gate_correct and not self._cheat_tip:
            fresh = active & ~self._gatecorr_set
            if bool(fresh.any()):
                from isaaclab.utils.math import quat_apply

                try:
                    fm = env.unwrapped.scene[self.names["peg_finger_sensor"]].data.force_matrix_w
                    ff = fm.sum(dim=1)
                    dz = ff[:, 0, 2] - ff[:, 1, 2] if ff.shape[1] >= 2 else torch.zeros(tip.shape[0], device=tip.device)
                except Exception:  # noqa: BLE001
                    dz = torch.zeros(tip.shape[0], device=tip.device)
                beta = self._decant_sign * (dz / max(self._decant_gain, 1e-6)) * (3.141592653589793 / 180.0)
                _, eq = read_ee_pose(env)
                yloc = torch.zeros(tip.shape[0], 3, device=tip.device)
                yloc[:, 1] = 1.0
                y_world = quat_apply(eq, yloc)  # EE y-axis in world (cock about x displaces the tip along y)
                corr = (-(self._gate_L_mm / 1000.0) * torch.sin(beta)).unsqueeze(-1) * y_world
                corr[:, 2] = 0.0  # lateral (xy) correction only
                self.tip_correction = torch.where(fresh.unsqueeze(-1), corr, self.tip_correction)
                self._gatecorr_set = self._gatecorr_set | fresh
        # Add the §2.1-honest correction (filled by the tactile CALIBRATE probe, Phase 2) / oracle above.
        # Zero by default -> identical to the committed headline path.
        tip = tip + self.tip_correction
        cur_ee, cur_quat = read_ee_pose(env)

        # capture the straight-down hold orientation once (commanded/proprioceptive, not peg pose)
        cap = self.need_capture & active
        if cap.any():
            q_down = level_to_down(cur_quat)
            if self._decant:
                q_down = self._apply_decant(env, q_down)
            self.q_hold = torch.where(cap.unsqueeze(-1), q_down, self.q_hold)
            self.need_capture = self.need_capture & ~cap

        sock = env.unwrapped.scene[self.names["socket"]].data
        axis_xy = sock.root_pos_w[:, :2]  # socket pose: allowed (known fixture, §2.1)
        mouth_z = sock.root_pos_w[:, 2] + self.mouth_height
        depth = mouth_z - tip[:, 2]
        m7 = self.m7_active
        # In M7 the controller works the peg toward a held offset point (axis + offset); normally it
        # works toward the socket axis. Lateral error and all xy targets are taken w.r.t. this center.
        # recover mode: once released (at PRESS entry) the effective offset/lean drop to zero so the
        # controller recenters on the true axis (axis_xy) and presses upright (recovery basin).
        if m7 and self.m7_recover:
            held = (~self.m7_released).unsqueeze(-1).float()
            m7_offset_eff = self.m7_offset * held
            m7_tilt_eff = self.m7_tilt_vec * held
        else:
            m7_offset_eff = self.m7_offset
            m7_tilt_eff = self.m7_tilt_vec
        center_xy = axis_xy + m7_offset_eff if m7 else axis_xy
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
        fz = W[:, 2].clamp(min=0.0)  # upward press reaction
        fxy = W[:, :2]
        fxy_mag = torch.norm(fxy, dim=-1)
        f_mag = torch.norm(W, dim=-1)

        # tactile CALIBRATE (§2.1-honest): run the probe sub-FSM for CALIBRATE-phase envs; it writes
        # self.tip_correction and flips CALIBRATE -> SETTLE on completion. Default off (no CALIBRATE
        # envs -> the committed headline path is untouched).
        if self._tactile_cal:
            cal_action = self._calibrate_step(env, active, tip, cur_ee, axis_xy, mouth_z, f_mag, speed)

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
            # M7: lean the peg by θ along the offset azimuth (m7_tilt_eff = θ·(cos az, sin az)); the
            # tilt axis is perpendicular to the lean dir so the downward peg axis tips toward +dir. In
            # recover mode m7_tilt_eff drops to 0 at release → theta=0 → q_cmd stays q_hold (upright).
            theta = torch.norm(m7_tilt_eff, dim=-1)
            d = m7_tilt_eff / theta.clamp(min=1e-9).unsqueeze(-1)
            axis = torch.zeros_like(tip)
            axis[:, 0] = d[:, 1]
            axis[:, 1] = -d[:, 0]
            axis[:, 2] = torch.where(theta < 1e-6, torch.ones_like(theta), torch.zeros_like(theta))
            q_cmd = quat_mul(quat_from_angle_axis(theta, axis), self.q_hold)
        rot_gain = c.get("rcc_rot_gain", 0.0)
        # RCC rotation runs on the normal path (not m7) and on the m7 *recover* path (after release,
        # to straighten a perturbed handoff). It stays OFF during the held-m7 map so that map is purely
        # geometric. The do_rot mask below further gates it to PRESS ∧ ~captured.
        if rot_gain > 0.0 and (not m7 or self.m7_recover):
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
        if self._tactile_cal:  # override CALIBRATE-phase envs with the probe motion
            in_cal = (self.phase == CALIBRATE) & active
            action = torch.where(in_cal.unsqueeze(-1), cal_action, action)

        # ---------------- state updates (gated by active) ----------------
        a = active
        self.timer = torch.where(a, self.timer + 1, self.timer)

        # ---- SETTLE -> PRESS: tip centred just above the mouth and (optionally) settled ----
        align_xy_tol = c.get("align_xy_tol", 0.004)
        settle_z_tol = c.get("settle_z_tol", 0.004)
        z_ok = (tip[:, 2] - settle_z).abs() < settle_z_tol
        if bool(c.get("use_settle", True)):
            ready = (
                (lateral < align_xy_tol)
                & z_ok
                & (speed < c.get("settle_speed", 0.03))
                & (self.timer >= int(c.get("settle_min_steps", 8)))
            )
        else:
            ready = (lateral < align_xy_tol) & ((tip[:, 2] - settle_z).abs() < c.get("settle_z_tol_fast", 0.006))
        to_press = a & (phase == SETTLE) & ready
        # recover mode: the perturbed handoff has been delivered (peg settled at offset+tilt) — release
        # the hold so PRESS recenters on the true axis and straightens (RCC) the peg.
        if self.m7_active and self.m7_recover:
            self.m7_released = self.m7_released | to_press
        self.z_press = torch.where(to_press, press_entry_z, self.z_press)
        self.spiral_r = torch.where(to_press, torch.zeros_like(self.spiral_r), self.spiral_r)
        self.spiral_th = torch.where(to_press, torch.zeros_like(self.spiral_th), self.spiral_th)
        self.best_depth = torch.where(to_press, torch.full_like(self.best_depth, -1.0), self.best_depth)
        self.stuck_timer = torch.where(to_press, torch.zeros_like(self.stuck_timer), self.stuck_timer)
        self.hop_count = torch.where(to_press, torch.zeros_like(self.hop_count), self.hop_count)
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
            progressed,
            torch.zeros_like(self.stuck_timer),
            torch.where(in_press, self.stuck_timer + 1, self.stuck_timer),
        )

        # wedge recovery (F/T based, §2.1). An (often unobserved) off-axis handoff can wedge the peg
        # at the funnel->bore rim: high lateral binding with no depth progress while still above the
        # capture depth. A full retract resets the spiral to r=0 and deterministically recreates the
        # SAME wedge, so first try cheap "hop-and-search" micro-recoveries: lift a few mm to UNLOAD the
        # wedge and advance the spiral to a NEW xy, staying in PRESS. Only after hop_max fruitless hops
        # do the full retract -> SETTLE retry; max_retries is the final backstop (-> insertion_failed).
        max_retries = int(c.get("max_retries", 5))
        binding = in_press & (fxy_mag > c.get("jam_force", 12.0)) & ~captured

        if bool(c.get("use_hop_search", False)):  # default OFF (c2.0 headline); controllers.yaml sets it
            # the stuck timer climbs whenever the tip makes no depth progress (binding implies stuck),
            # and is reset on each hop -> its threshold doubles as a cooldown between hops.
            hop_after = int(c.get("hop_after", 30))
            hop_max = int(c.get("hop_max", 5))
            wedged = in_press & ~captured & (self.stuck_timer > hop_after)
            hop = wedged & (self.hop_count < hop_max)
            self.z_press = torch.where(hop, self.z_press + c.get("hop_lift_m", 0.004), self.z_press)
            self.spiral_r = torch.where(
                hop,
                torch.clamp(self.spiral_r + c.get("hop_spiral_step", 0.0015), max=c["spiral_radius_max"]),
                self.spiral_r,
            )
            self.spiral_th = torch.where(hop, self.spiral_th + c.get("hop_az_step", 2.39996), self.spiral_th)
            self.stuck_timer = torch.where(hop, torch.zeros_like(self.stuck_timer), self.stuck_timer)
            self.hop_count = torch.where(hop, self.hop_count + 1, self.hop_count)
            retry = wedged & (self.hop_count >= hop_max) & (self.attempts < max_retries)
        else:
            retry_after = int(c.get("retry_after_steps", 150))
            retry = in_press & ((self.stuck_timer > retry_after) | binding) & (self.attempts < max_retries)

        self.attempts = torch.where(retry, self.attempts + 1, self.attempts)
        self.spiral_r = torch.where(retry, torch.zeros_like(self.spiral_r), self.spiral_r)
        self.best_depth = torch.where(retry, torch.full_like(self.best_depth, -1.0), self.best_depth)
        self.stuck_timer = torch.where(retry, torch.zeros_like(self.stuck_timer), self.stuck_timer)
        self.hop_count = torch.where(retry, torch.zeros_like(self.hop_count), self.hop_count)
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
