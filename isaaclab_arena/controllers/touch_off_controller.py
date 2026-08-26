# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""spec §4-1 ``touch_off``: measure the held part against a datum block instead of assuming it.

**The number this exists to replace.** Everything downstream of the grasp is currently commanded
from a COMPUTED protrusion: the part is assumed to sit at exactly the commanded grip station, so
``p_hat = datum_offset - (station - fingertip_offset)``. That assumption is what the insertion depth,
the QC protrusion window and the whole precision budget rest on, and it is wrong by however much the
part slipped, tilted or was gripped off-station. Touching a surveyed block turns that assumption
into a measurement.

**§2.1 / §0-5 honest.** The inputs are the ``ee_frame`` TCP (proprioception), the touch-off block's
surveyed pose (a fixture, known to the cell), and contact force. The workpiece's true pose is never
read -- that is ``/privileged/*`` and using it would make the measurement meaningless, since the
whole point is to recover what the machine cannot see.

**Three faces, three numbers.** Probing the block's TOP gives the axial protrusion; probing two
orthogonal SIDES gives the lateral offset of the part's axis in x and y. Together they are the
measured error vector ``e_hat`` the three-way gate routes on (§5), and the protrusion is what the
loader should servo instead of its nominal.

The contact test is the one the tactile CALIBRATE probe already uses: wrist-force magnitude over a
per-touch baseline, with a slow approach so the contact z is clean rather than rammed.
"""

from __future__ import annotations

import os
import torch

from isaaclab_arena.controllers.ee_control import ee_pose_action, read_ee_pose

# phase codes
GOTO, PROBE, RETRACT, DONE = range(4)


class TouchOffController:
    """Probe a held part against the datum block; write ``p_hat`` and ``e_lateral``.

    One face at a time, driven by ``faces`` (config ``touchoff.faces``). Each face runs
    GOTO -> PROBE -> RETRACT, then the index advances; after the last face the controller reports
    DONE and holds still.
    """

    def __init__(self, names: dict, block_pose, block_size, ee_cfg: dict, cfg: dict, geom: dict):
        self.names = names
        self.ee_cfg = ee_cfg
        self.cfg = cfg
        self.block_pose = tuple(float(v) for v in block_pose)
        self.block_size = tuple(float(v) for v in block_size)
        self.faces = [str(f) for f in cfg.get("faces", ["top", "x", "y"])]
        self.probe_speed = float(cfg.get("probe_speed", 0.005))
        self.contact_force = float(cfg.get("contact_force", 3.0))
        self.retract = float(cfg.get("retract", 0.010))
        self.approach_clear = float(cfg.get("approach_clear", 0.060))
        # 40 mm per step for travel. The block sits up to ~0.7 m from where the lift ends, so at the
        # 10 mm this started with the arm spends ~850 steps in transit and the episode times out
        # before the first face is touched.
        self.travel_step = float(cfg.get("travel_step", 0.040))
        self.max_travel = float(cfg.get("max_travel", 0.120))
        # How close the TCP must get to the stand-off before the probe starts. 8 mm was too tight
        # against a 10 mm travel step: the controller can straddle the target without ever landing
        # inside the tolerance, so GOTO never completes and the probe never begins.
        self.arrive_tol = float(cfg.get("arrive_tol", 0.015))
        self.geom = geom

        self.phase: torch.Tensor | None = None
        self.face_idx: torch.Tensor | None = None
        self.baseline: torch.Tensor | None = None
        self.contact_tcp: torch.Tensor | None = None  # (N, F, 3) TCP at each face's contact
        self.recorded: torch.Tensor | None = None  # (N, F) bool
        self.q_hold: torch.Tensor | None = None
        self.start_tcp: torch.Tensor | None = None
        # results, in the same units as everything else (metres)
        self.p_hat: torch.Tensor | None = None
        self.e_lateral: torch.Tensor | None = None
        # The measured axial grip station. This, not p_hat, is what the loader consumes: `station`
        # is threaded through the whole insertion path (tip offset, reachable depth, creep target),
        # so replacing the COMMANDED station with the measured one corrects every one of them at
        # once instead of patching the depth at the end.
        self.station_hat: torch.Tensor | None = None
        self.measured: torch.Tensor | None = None  # (N,) bool -- did the top face actually register

    # ------------------------------------------------------------------ setup
    def _ensure(self, env, tcp, quat):
        if self.phase is not None and self.phase.shape[0] == tcp.shape[0]:
            return
        n, dev = tcp.shape[0], tcp.device
        self.phase = torch.full((n,), GOTO, dtype=torch.long, device=dev)
        self.face_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.baseline = torch.zeros(n, device=dev)
        self.contact_tcp = torch.zeros(n, len(self.faces), 3, device=dev)
        self.recorded = torch.zeros(n, len(self.faces), dtype=torch.bool, device=dev)
        self.p_hat = torch.zeros(n, device=dev)
        self.e_lateral = torch.zeros(n, 2, device=dev)
        self.station_hat = torch.zeros(n, device=dev)
        self.measured = torch.zeros(n, dtype=torch.bool, device=dev)
        # The wrist orientation is LATCHED, exactly as the pick latches its grasp quat: re-deriving
        # it mid-probe would rotate the part between touches and make the three faces describe
        # different geometries.
        self.q_hold = quat.clone()
        self.start_tcp = tcp.clone()

    def reset(self, env_ids=None):
        for attr in ("phase", "face_idx", "baseline", "contact_tcp", "recorded", "q_hold", "start_tcp"):
            setattr(self, attr, None)
        self.p_hat = None
        self.e_lateral = None
        self.station_hat = None
        self.measured = None

    # -------------------------------------------------------------- geometry
    def _probe_targets(self, tcp, face: str, station: torch.Tensor):
        """(approach point, probe direction) for ``face``, in TCP coordinates.

        The TCP is commanded, not the part: the part's tip hangs ``station`` below the TCP along the
        held axis, so a target expressed for the tip is converted by adding that offset back.
        """
        bx, by, bz = self.block_pose
        sx, sy, sz = self.block_size
        top_z = bz + sz
        if face == "top":
            # Straight down onto the top face. Contact z tells us where the tip actually is.
            approach = torch.stack(
                [
                    torch.full_like(station, bx),
                    torch.full_like(station, by),
                    torch.full_like(station, top_z + self.approach_clear) + station,
                ],
                dim=-1,
            )
            direction = torch.tensor([0.0, 0.0, -1.0], device=tcp.device).expand_as(approach)
        elif face == "x":
            # Sideways onto the +x face, at a height that is on the block's side wall.
            approach = torch.stack(
                [
                    torch.full_like(station, bx + sx / 2.0 + self.approach_clear),
                    torch.full_like(station, by),
                    torch.full_like(station, bz + sz / 2.0) + station,
                ],
                dim=-1,
            )
            direction = torch.tensor([-1.0, 0.0, 0.0], device=tcp.device).expand_as(approach)
        elif face == "y":
            approach = torch.stack(
                [
                    torch.full_like(station, bx),
                    torch.full_like(station, by + sy / 2.0 + self.approach_clear),
                    torch.full_like(station, bz + sz / 2.0) + station,
                ],
                dim=-1,
            )
            direction = torch.tensor([0.0, -1.0, 0.0], device=tcp.device).expand_as(approach)
        else:
            raise ValueError(f"unknown touch-off face {face!r}; expected one of top/x/y")
        return approach, direction

    # ------------------------------------------------------------------ solve
    def _solve(self, station: torch.Tensor, radius: torch.Tensor, datum_offset: torch.Tensor):
        """Turn the recorded contact TCPs into ``p_hat`` and ``e_lateral``.

        ``p_hat`` is the protrusion of the datum feature past the fingers, which is what the loader
        needs: at the top-face touch the part's TIP is exactly on the block's top face, so the
        distance from the TCP down to that face IS the measured station. The protrusion follows from
        the variant's datum offset, and the whole point is that neither number is assumed.
        """
        bx, by, bz = self.block_pose
        sx, sy, sz = self.block_size
        top_z = bz + sz
        for i, face in enumerate(self.faces):
            got = self.recorded[:, i]
            if not bool(got.any()):
                continue
            contact = self.contact_tcp[:, i]
            if face == "top":
                measured_station = contact[:, 2] - top_z
                self.station_hat = torch.where(got, measured_station, self.station_hat)
                self.p_hat = torch.where(got, datum_offset - measured_station, self.p_hat)
                self.measured = self.measured | got
            elif face == "x":
                nominal = bx + sx / 2.0 + radius
                self.e_lateral[:, 0] = torch.where(got, contact[:, 0] - nominal, self.e_lateral[:, 0])
            elif face == "y":
                nominal = by + sy / 2.0 + radius
                self.e_lateral[:, 1] = torch.where(got, contact[:, 1] - nominal, self.e_lateral[:, 1])

    # ------------------------------------------------------------------- step
    def step(self, env, active: torch.Tensor, station: torch.Tensor, radius: torch.Tensor, datum_offset: torch.Tensor):
        """Return ``(action, finished)``. State only advances where ``active``."""
        from isaaclab_arena_environments.mdp.bdash_chuck_predicates import _filtered_norms, _net_norms

        tcp, quat = read_ee_pose(env)
        self._ensure(env, tcp, quat)
        n_faces = len(self.faces)

        # THE NET force on the workpiece, not the finger-filtered one. The filtered sensor reports
        # workpiece<->FINGERS, which is the grip and says nothing about touching a block: measured,
        # the probe drove through all three faces with the filtered signal never moving, so every
        # face was recorded as a give-up and `measured` stayed False. The net sensor sees every
        # contact, including the datum block -- the grip shows up as a constant, which is exactly
        # what the per-touch baseline below subtracts off.
        # PART-vs-FIXTURE force, not the net. The net sums every contact on every workpiece --
        # including the HELD part's own grip forces, which swing +-1.5 N with arm acceleration, so a
        # 1.5-3 N threshold sat inside the noise floor: measured, the top-face probe fired 2.86 mm
        # early with byte-identical readings under 0 mm and 8 mm injected error (a sensor with zero
        # sensitivity), and in another run never fired at all and pressed the part 20 mm through the
        # fingers. The fixture-filtered signal is exactly zero until the part touches the pad, which
        # is the event being measured. Falls back to the net only if the wiring is absent.
        if self.names.get("fixture_sensors"):
            force = _filtered_norms(env, tuple(self.names["fixture_sensors"])).sum(dim=1)
        else:
            force = _net_norms(env, tuple(self.names["workpiece_contact_sensors"])).sum(dim=1)
        idx = self.face_idx.clamp(max=n_faces - 1)

        # Per-face target. Built for every env at its own face index; a done env is clamped and then
        # masked out below, so it never moves.
        approach = torch.zeros_like(tcp)
        direction = torch.zeros_like(tcp)
        for i, face in enumerate(self.faces):
            a, d = self._probe_targets(tcp, face, station)
            sel = (idx == i).unsqueeze(-1)
            approach = torch.where(sel, a, approach)
            direction = torch.where(sel, d, direction)

        at_approach = torch.norm(tcp - approach, dim=-1) < self.arrive_tol
        if os.environ.get("BDASH_TOUCH_DIAG"):
            self._diag = getattr(self, "_diag", 0) + 1
            if self._diag % 20 == 1 and bool(active[0]):
                print(
                    f"[touchdiag] phase={int(self.phase[0])} face={int(self.face_idx[0])} "
                    f"dist={float(torch.norm(tcp[0] - approach[0])):.4f} "
                    f"tcp=({tcp[0, 0]:.3f},{tcp[0, 1]:.3f},{tcp[0, 2]:.3f}) "
                    f"goal=({approach[0, 0]:.3f},{approach[0, 1]:.3f},{approach[0, 2]:.3f}) "
                    f"force={float(force[0]):.2f} base={float(self.baseline[0]):.2f}",
                    flush=True,
                )
        # GOTO -> PROBE once parked at the stand-off, snapshotting the pre-touch force baseline.
        to_probe = active & (self.phase == GOTO) & at_approach
        self.baseline = torch.where(to_probe, force, self.baseline)
        self.phase = torch.where(to_probe, torch.full_like(self.phase, PROBE), self.phase)

        # PROBE -> RETRACT on contact over the baseline. The travelled distance is capped so a
        # missed face gives up instead of pushing the arm into the bench.
        travelled = torch.norm(tcp - approach, dim=-1)
        contact = (force - self.baseline) > self.contact_force
        # Give up on TRAVEL or on STALL. Travel alone is not reachable when the press hits a hard
        # stop before the cap -- the arm parks against the surface, `travelled` saturates below the
        # limit, and the probe pushes the held part through the fingers forever (measured: a
        # 20 mm in-gripper slip, and a timeout with the phase still PROBE). No descent for a second
        # while probing IS the contact event, whether or not the force sensor saw it.
        if not hasattr(self, "_stall_z") or self._stall_z.shape[0] != tcp.shape[0]:
            self._stall_z = tcp[:, 2].clone()
            self._stall_n = torch.zeros(tcp.shape[0], dtype=torch.long, device=tcp.device)
        moving = (tcp[:, 2] - self._stall_z).abs() > 0.0002
        probing_now = active & (self.phase == PROBE)
        self._stall_n = torch.where(probing_now & ~moving, self._stall_n + 1, torch.zeros_like(self._stall_n))
        self._stall_z = tcp[:, 2].clone()
        gave_up = (travelled > self.max_travel) | (self._stall_n > 30)
        hit = active & (self.phase == PROBE) & (contact | gave_up)
        hit_i = hit.nonzero(as_tuple=True)[0]
        if hit_i.numel() and os.environ.get("BDASH_TOUCH_DIAG"):
            k = int(hit_i[0])
            print(
                f"[touchhit] face={self.faces[int(idx[k])]} tcp_z={float(tcp[k, 2]) * 1e3:.1f}mm "
                f"travelled={float(travelled[k]) * 1e3:.1f}mm force={float(force[k]):.3f} "
                f"base={float(self.baseline[k]):.3f} contact={bool(contact[k])} stall={int(self._stall_n[k])}",
                flush=True,
            )
        if hit_i.numel():
            self.contact_tcp[hit_i, idx[hit_i]] = tcp[hit_i]
            self.recorded[hit_i, idx[hit_i]] = contact[hit_i]  # a give-up is NOT a measurement
        self.phase = torch.where(hit, torch.full_like(self.phase, RETRACT), self.phase)

        # RETRACT -> next face (or DONE).
        backed_off = torch.norm(tcp - approach, dim=-1) < 0.004
        stepped = active & (self.phase == RETRACT) & backed_off
        self.face_idx = torch.where(stepped, self.face_idx + 1, self.face_idx)
        self.phase = torch.where(
            stepped,
            torch.where(self.face_idx >= n_faces, torch.full_like(self.phase, DONE), torch.full_like(self.phase, GOTO)),
            self.phase,
        )

        if bool((self.phase == DONE).any()):
            self._solve(station, radius, datum_offset)

        # Action: GOTO/RETRACT go to the stand-off; PROBE creeps along the face normal; DONE holds.
        probe_target = approach + direction * self.max_travel
        target = torch.where((self.phase == PROBE).unsqueeze(-1), probe_target, approach)
        target = torch.where((self.phase == DONE).unsqueeze(-1), tcp, target)
        # FALSE means the fingers stay shut -- the argument is `gripper_open`. This was a
        # `torch.ones` named `closed`, i.e. inverted, so the probe would have dropped the part the
        # moment it took the arm. Never observed because this leg has not been measured yet; found
        # by the identical bug in the re-erect leg, which HAS been.
        hold_shut = torch.zeros(tcp.shape[0], dtype=torch.bool, device=tcp.device)
        # TWO SPEEDS, selected per env rather than by a global max. Travel and touch have opposite
        # requirements: the datum block can be most of a metre away (measured 0.687 m from where the
        # lift ends), so a 10 mm travel step needs ~850 steps just to arrive -- the first attempt
        # never reached the block at all and the probe sat in GOTO for the whole episode. The touch
        # itself has to stay slow, or the contact z is a ram rather than a measurement.
        act_travel = ee_pose_action(
            env, target, self.q_hold, gripper_open=hold_shut, cfg=self.ee_cfg, max_pos_step=self.travel_step
        )
        act_probe = ee_pose_action(
            env, target, self.q_hold, gripper_open=hold_shut, cfg=self.ee_cfg, max_pos_step=self.probe_speed
        )
        action = torch.where((self.phase == PROBE).unsqueeze(-1), act_probe, act_travel)
        return action, self.phase == DONE
