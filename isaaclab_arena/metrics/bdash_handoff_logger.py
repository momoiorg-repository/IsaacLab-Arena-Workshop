# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""B-DASH §3.5 per-episode handoff / terminal-error logger.

A first-class requirement of the brief is that we record *more than the success rate*: for every
episode we want the step at which each §3.4 predicate first fired, the geometry of the VLA→rule
handoff (lateral ``dx/dy`` and peg-axis tilt at the moment ``handoff`` first holds), the contact-force
statistics (running max + number of force-limit violations), and the terminal cause
(``success`` | ``insertion_failed`` | ``dropped`` | ``timeout`` | ``none``).

Implemented as an Isaac Lab :class:`RecorderTerm`:
  * ``record_post_step`` accumulates per-env running state from the simulator each control step;
  * ``record_pre_reset`` resolves the terminal cause from the termination manager, writes one JSONL
    record per finishing episode, and emits a per-episode result code for metric aggregation.

The thresholds/geometry come from ``configs/bdash/peg_insert/task.yaml`` (passed in via the cfg) so nothing here
is hard-coded. The active socket's clearance ``c`` is injected per-build for the ``inserted`` test.
"""

from __future__ import annotations

import datetime
import json
import math
import numpy as np
import os
import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase

# Predicate helpers (torch-only; importing this submodule does not pull in Isaac Sim).
from isaaclab_arena_environments.mdp import bdash_peg_predicates as vp

# terminal-cause result codes (kept in the JSONL too, as the human-readable string)
_RESULT_NONE = 0
_RESULT_SUCCESS = 1
_RESULT_INSERTION_FAILED = 2
_RESULT_DROPPED = 3
_RESULT_TIMEOUT = 4
_RESULT_STR = {
    _RESULT_NONE: "none",
    _RESULT_SUCCESS: "success",
    _RESULT_INSERTION_FAILED: "insertion_failed",
    _RESULT_DROPPED: "dropped",
    _RESULT_TIMEOUT: "timeout",
}


class BDashHandoffRecorder(RecorderTerm):
    """Accumulates §3.5 per-episode statistics and writes one JSONL record per episode."""

    def __init__(self, cfg: BDashHandoffRecorderCfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self._t = cfg.task_cfg  # loaded task.yaml dict
        self._clearance = cfg.clearance  # meters (active socket)
        self._names = self._t["scene_names"]
        self._geom = self._t["geometry"]
        self._tip_offset = tuple(self._geom["tip_offset"])
        self._mouth_h = float(self._geom["mouth_height"])
        self._force_max = float(self._t["force"]["force_max"])

        # output JSONL path (one file per run; created lazily on the first written record)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = cfg.output_path or os.path.join(cfg.output_dir, f"bdash_handoff_{cfg.run_tag}_{ts}.jsonl")
        self._fh = None
        self._disabled = False

        self.first_reset = True
        self._episode_counter = 0
        self._init_state()

    # ------------------------------------------------------------------ state
    def _init_state(self, env_ids=None):
        n = self._env.num_envs
        dev = self._env.device
        if env_ids is None:
            env_ids = slice(None)
            self.t_grasped = torch.full((n,), -1, dtype=torch.long, device=dev)
            self.t_handoff = torch.full((n,), -1, dtype=torch.long, device=dev)
            self.t_inserted = torch.full((n,), -1, dtype=torch.long, device=dev)
            self.handoff_dx = torch.zeros(n, device=dev)
            self.handoff_dy = torch.zeros(n, device=dev)
            self.handoff_tilt = torch.zeros(n, device=dev)
            self.force_max_seen = torch.zeros(n, device=dev)
            self.force_max_depth = torch.zeros(n, device=dev)  # tip depth (m) at the moment of peak force
            self.force_max_step = torch.zeros(n, dtype=torch.long, device=dev)  # step of the peak
            self.force_insert_max = torch.zeros(n, device=dev)  # peak force during the insertion window only
            self.force_violations = torch.zeros(n, dtype=torch.long, device=dev)
            self.max_depth = torch.full((n,), -1.0, device=dev)
            self.final_tilt = torch.zeros(n, device=dev)
            return
        self.t_grasped[env_ids] = -1
        self.t_handoff[env_ids] = -1
        self.t_inserted[env_ids] = -1
        self.handoff_dx[env_ids] = 0.0
        self.handoff_dy[env_ids] = 0.0
        self.handoff_tilt[env_ids] = 0.0
        self.force_max_seen[env_ids] = 0.0
        self.force_max_depth[env_ids] = 0.0
        self.force_max_step[env_ids] = 0
        self.force_insert_max[env_ids] = 0.0
        self.force_violations[env_ids] = 0
        self.max_depth[env_ids] = -1.0
        self.final_tilt[env_ids] = 0.0

    # --------------------------------------------------------------- per step
    def record_post_step(self):
        env = self._env
        step = env.episode_length_buf.long()  # per-env step count this episode

        g = vp.grasped(
            env,
            robot_name=self._names["robot"],
            peg_finger_sensor=self._names["peg_finger_sensor"],
            **self._t["grasped"],
        )
        h = vp.handoff(
            env,
            peg_name=self._names["peg"],
            socket_name=self._names["socket"],
            robot_name=self._names["robot"],
            peg_finger_sensor=self._names["peg_finger_sensor"],
            tip_offset=self._tip_offset,
            mouth_height=self._mouth_h,
            **self._t["handoff"],
            **self._t["grasped"],
        )
        ins = vp.inserted(
            env,
            peg_name=self._names["peg"],
            socket_name=self._names["socket"],
            clearance=self._clearance,
            tip_offset=self._tip_offset,
            mouth_height=self._mouth_h,
            min_depth=float(self._t["inserted"]["min_depth"]),
        )

        # first-occurrence step latches (only set when still -1)
        self.t_grasped = torch.where((self.t_grasped < 0) & g, step, self.t_grasped)
        self.t_inserted = torch.where((self.t_inserted < 0) & ins, step, self.t_inserted)

        new_handoff = (self.t_handoff < 0) & h
        if new_handoff.any():
            tip = vp._peg_tip_w(env, self._names["peg"], self._tip_offset)
            axis_xy, _ = vp._hole_frame(env, self._names["socket"], self._mouth_h)
            dx = tip[:, 0] - axis_xy[:, 0]
            dy = tip[:, 1] - axis_xy[:, 1]
            # peg long axis (local +z) vs world vertical -> tilt angle
            q = env.scene[self._names["peg"]].data.root_quat_w
            zloc = torch.zeros_like(tip)
            zloc[:, 2] = 1.0
            axis_w = vp._quat_rotate(q, zloc)
            tilt = torch.acos(axis_w[:, 2].abs().clamp(-1.0, 1.0))
            self.handoff_dx = torch.where(new_handoff, dx, self.handoff_dx)
            self.handoff_dy = torch.where(new_handoff, dy, self.handoff_dy)
            self.handoff_tilt = torch.where(new_handoff, tilt, self.handoff_tilt)
            self.t_handoff = torch.where(new_handoff, step, self.t_handoff)

        # insertion-depth and peg-tilt diagnostics (terminal-error analysis, brief §3.5)
        tip = vp._peg_tip_w(env, self._names["peg"], self._tip_offset)
        axis_xy, mouth_z = vp._hole_frame(env, self._names["socket"], self._mouth_h)
        depth = mouth_z - tip[:, 2]
        self.max_depth = torch.maximum(self.max_depth, depth)

        # force statistics on the peg (net contact force norm); track the tip depth and step at the
        # peak so we can tell an insertion-press spike from a reset/grasp/table transient.
        f = vp._net_force_norm(env, self._names["peg_sensor"])
        new_force_max = f > self.force_max_seen
        self.force_max_depth = torch.where(new_force_max, depth, self.force_max_depth)
        self.force_max_step = torch.where(new_force_max, step, self.force_max_step)
        self.force_max_seen = torch.maximum(self.force_max_seen, f)
        # peak force during the *insertion window only* (peg grasped, tip from just above the mouth to
        # just above the bore floor) — this is the figure the brief's single-digit-N target refers to,
        # isolated from the reset/grasp/table-rest transients that pollute the whole-episode max.
        in_insert_window = g & (depth > -0.005) & (depth < 0.024)
        self.force_insert_max = torch.where(
            in_insert_window, torch.maximum(self.force_insert_max, f), self.force_insert_max
        )
        self.force_violations = self.force_violations + (f > self._force_max).long()

        q = env.scene[self._names["peg"]].data.root_quat_w
        zloc = torch.zeros_like(tip)
        zloc[:, 2] = 1.0
        axis_w = vp._quat_rotate(q, zloc)
        self.final_tilt = torch.acos(axis_w[:, 2].abs().clamp(-1.0, 1.0))

        if os.environ.get("BDASH_DEBUG") and int(step[0].item()) < 6:
            tip = vp._peg_tip_w(env, self._names["peg"], self._tip_offset)
            axis_xy, mouth_z = vp._hole_frame(env, self._names["socket"], self._mouth_h)
            depth = (mouth_z - tip[:, 2])[0].item()
            lateral = torch.norm(tip[:, :2] - axis_xy, dim=-1)[0].item()
            ff = vp._filtered_force_norm(env, self._names["peg_finger_sensor"])[0].item()
            gw = vp.gripper_width(env, self._names["robot"])[0].item()
            sock = env.scene[self._names["socket"]].data.root_pos_w[0]
            print(
                f"[BDASH_DEBUG] step={int(step[0].item())} tip_z={tip[0,2].item():.4f} "
                f"sock_z={sock[2].item():.4f} depth={depth:.4f} lat={lateral*1000:.2f}mm "
                f"net_f={f[0].item():.2f}N finger_f={ff:.2f}N grip_w={gw*1000:.2f}mm "
                f"g={int(g[0].item())} h={int(h[0].item())} ins={int(ins[0].item())}",
                flush=True,
            )

        return None, None

    # -------------------------------------------------------------- per reset
    def _terminal_codes(self, env_ids) -> torch.Tensor:
        """Resolve the terminal cause for each finishing env from the termination manager."""
        n = len(env_ids)
        dev = self._env.device
        codes = torch.full((n,), _RESULT_NONE, dtype=torch.long, device=dev)
        tm = getattr(self._env, "termination_manager", None)
        if tm is None:
            return codes
        active = tm.active_terms

        def term(name):
            return tm.get_term(name)[env_ids] if name in active else torch.zeros(n, dtype=torch.bool, device=dev)

        # lowest-to-highest priority so the highest-priority cause wins
        codes = torch.where(term("time_out"), torch.full_like(codes, _RESULT_TIMEOUT), codes)
        codes = torch.where(term("object_dropped"), torch.full_like(codes, _RESULT_DROPPED), codes)
        codes = torch.where(term("insertion_failed"), torch.full_like(codes, _RESULT_INSERTION_FAILED), codes)
        codes = torch.where(term("success"), torch.full_like(codes, _RESULT_SUCCESS), codes)
        return codes

    def record_pre_reset(self, env_ids):
        # First reset: nothing has happened yet (all envs reset together). Mirror SuccessRecorder.
        if self.first_reset:
            assert len(env_ids) == self._env.num_envs
            self.first_reset = False
            self._init_state(env_ids)
            return None, None

        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        env_ids = torch.as_tensor(env_ids, device=self._env.device).long()

        codes = self._terminal_codes(env_ids)
        # Reclassify a timeout that *reached the insertion attempt* (handoff fired but the peg never
        # seated) as insertion_failed, so §3.5 failures aren't all lumped as "timeout": under the
        # low-force servo the force-only `insertion_failed` predicate rarely fires, yet per §3.4 an
        # insertion that times out IS an insertion failure. A bare `timeout` now means the episode
        # never got as far as handoff (e.g. a failed grasp).
        reached_insert = self.t_handoff[env_ids] >= 0
        codes = torch.where(
            (codes == _RESULT_TIMEOUT) & reached_insert, torch.full_like(codes, _RESULT_INSERTION_FAILED), codes
        )
        dt = float(self._env.step_dt)

        def secs(step_tensor, i):
            s = int(step_tensor[env_ids][i].item())
            return None if s < 0 else round(s * dt, 4)

        for i, eid in enumerate(env_ids.tolist()):
            self._episode_counter += 1
            record = {
                "episode": self._episode_counter,
                "env_id": int(eid),
                "clearance_mm": round(self._clearance * 1000.0, 4),
                "result": _RESULT_STR[int(codes[i].item())],
                "t_grasped_s": secs(self.t_grasped, i),
                "t_handoff_s": secs(self.t_handoff, i),
                "t_inserted_s": secs(self.t_inserted, i),
                "handoff_dx_mm": round(float(self.handoff_dx[env_ids][i].item()) * 1000.0, 3),
                "handoff_dy_mm": round(float(self.handoff_dy[env_ids][i].item()) * 1000.0, 3),
                "handoff_tilt_deg": round(math.degrees(float(self.handoff_tilt[env_ids][i].item())), 3),
                "force_max_N": round(float(self.force_max_seen[env_ids][i].item()), 3),
                "force_max_depth_mm": round(float(self.force_max_depth[env_ids][i].item()) * 1000.0, 2),
                "force_max_step": int(self.force_max_step[env_ids][i].item()),
                "force_insert_max_N": round(float(self.force_insert_max[env_ids][i].item()), 3),
                "force_violations": int(self.force_violations[env_ids][i].item()),
                "max_depth_mm": round(float(self.max_depth[env_ids][i].item()) * 1000.0, 2),
                "final_tilt_deg": round(math.degrees(float(self.final_tilt[env_ids][i].item())), 2),
                "episode_steps": int(self._env.episode_length_buf[eid].item()),
            }
            self._write(record)

        self._init_state(env_ids)
        return self.name, codes

    # ------------------------------------------------------------------- io
    def _write(self, record: dict):
        # §3.5 logging must never crash a rollout: if the configured dir isn't writable (e.g. it was
        # created root-owned by an earlier `docker exec` run), fall back to a temp dir, and only
        # disable logging (with a warning) if no location works.
        if self._disabled:
            return
        if self._fh is None:
            self._fh = self._open_log()
            if self._fh is None:
                self._disabled = True
                return
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def _open_log(self):
        import tempfile

        fallback = os.path.join(tempfile.gettempdir(), "bdash", os.path.basename(self._path))
        last_err = None
        for path in (self._path, fallback):
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                fh = open(path, "a")
                if path != self._path:
                    print(
                        f"[BDashHandoffLogger] '{self._path}' not writable ({last_err}); "
                        f"writing §3.5 log to '{path}' instead.",
                        flush=True,
                    )
                    self._path = path
                return fh
            except OSError as e:
                last_err = e
        print(f"[BDashHandoffLogger] no writable log location ({last_err}); §3.5 logging disabled.", flush=True)
        return None


@configclass
class BDashHandoffRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = BDashHandoffRecorder
    name: str = "bdash_handoff"
    task_cfg: dict = None
    clearance: float = 0.002
    output_dir: str = "logs/bdash"
    output_path: str | None = None
    run_tag: str = "run"


class BDashHandoffLogger(MetricBase):
    """Metric wrapper that installs :class:`BDashHandoffRecorder` and reports the logged success rate.

    The rich per-episode data lands in the JSONL file; the aggregated number returned here is the
    fraction of logged episodes whose terminal cause was ``success`` (so the eval grid still gets a
    single scalar per cell, while the JSONL carries the predicate/handoff/force breakdown).
    """

    name = "bdash_handoff_success_rate"
    recorder_term_name = "bdash_handoff"

    def __init__(self, task_cfg: dict, clearance: float, output_dir: str = "logs/bdash", run_tag: str = "run"):
        self._task_cfg = task_cfg
        self._clearance = clearance
        self._output_dir = output_dir
        self._run_tag = run_tag

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return BDashHandoffRecorderCfg(
            name=self.recorder_term_name,
            task_cfg=self._task_cfg,
            clearance=self._clearance,
            output_dir=self._output_dir,
            run_tag=self._run_tag,
        )

    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        if len(recorded_metric_data) == 0:
            return 0.0
        codes = np.concatenate([np.atleast_1d(a) for a in recorded_metric_data])
        return float(np.mean(codes == _RESULT_SUCCESS))
