# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH chuck-loading scripted teacher: grasp an arbitrarily-posed workpiece and lift it clear.

This is the expert whose demonstrations the VLA is trained on, so what it reads and how it moves are
both load-bearing.

**It reads the target workpiece's ground-truth pose.** That is deliberate, and it is not the mistake
that killed v3. v3 failed because the teacher aligned the wrist to a variable the policy could not
*see* -- a Ø8 mm square peg's yaw, worth a couple of pixels -- so the action depended on something
absent from the observation and the policy regressed to the mean. Here the variable is the direction
of a Ø25 x 90 mm bar: 26 px in the side cameras and ~100 px in the wrist camera at exactly the
moment the wrist has to turn. Whether that is genuinely recoverable is settled empirically by
``scripts/bdash/precheck_axis_observability.py`` on the recorded pixels, with the v3 dataset as a
calibrated negative control -- not by routing the teacher through an estimator, which would only
move the question.

**Upright parts keep the frozen v4/v6 contract.** They are bodies of revolution, so wrist yaw is
unobservable *and* irrelevant, and they are grasped with the fixed down-quat exactly as before. This
matters more than it looks: an upright part still carries a few degrees of tilt, and feeding that
tilt's direction through ``atan2`` would make the commanded wrist depend on a 3-degree wobble no
camera can resolve -- reintroducing the v3 disease through the back door.

Registered as policy ``bdash_chuck_teacher``.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import math
import os
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.controllers.budget_gate import GO, REFIX, REJECT
from isaaclab_arena.controllers.budget_gate import decide as gate_decide
from isaaclab_arena.controllers.ee_control import grasp_quat_from_axis, read_ee_pose
from isaaclab_arena.controllers.scripted_chuck_pick import ScriptedChuckPick
from isaaclab_arena.controllers.scripted_pick import CLOSE
from isaaclab_arena.policy.policy_base import PolicyBase

# MEASURED from franka_description/meshes/collision/finger.stl: the gripping FACE is the tip
# 17.9 mm of a 53.9 mm link (link z 0.0360-0.0539) and the ee_frame TCP sits 8.9 mm inboard of the
# fingertip, so the fingertip is this far BELOW the commanded TCP.
_FINGERTIP_BELOW_TCP = 0.0089
# Keep the fingertips clear of the chuck face. They cannot enter the bore at all -- the same mesh
# reaches 26.2 mm outward from its gripping face, so on a Ø25 shaft a finger spans radius
# 12.5-38.7 mm against a bore radius of 20 mm.
_FACE_MARGIN = 0.002
#: RETIRED to 0 (2026-08-24). The "1.1 mm tracking gap" was measured while the arm was pressing an
#: unreachable target against the work-stopper ring -- the gap was the RING, not the PD. With the
#: target moved inside the reachable region the feed-forward OVERSHOT instead: three of five W-A
#: landed 1.7 mm past the command, back against the ring, cocking the part to 3.1-4.7 deg. Free-
#: space landings without it measure within +-1 mm of the command, which the QC windows absorb.
_CREEP_TRACKING_GAP = 0.0
# TCP speed under which the part is treated as stopped, for the handover trigger (m/s).
_HANDOVER_SPEED = 0.01
# Orientation error under which the tool counts as square, for SETTLE COMPLETE (rad).
# Well inside `rot_tol` (0.06 rad = 3.4 deg), which alone is 7x the 0.5 deg QC bar.
_SETTLE_ROT_TOL = 0.004
# Backstop: settle no longer than this even if the pose never fully converges.
_SETTLE_MAX = 60
# Control-lag the trigger extrapolates over, in steps. Measured overshoot was a uniform +1.1 mm
# against a 0.5 mm creep step, i.e. about two steps in flight.
_TRIGGER_LOOKAHEAD_STEPS = 2
# ~1 s at step_dt 16.7 ms: long enough for a released part to fall and stop bouncing.
_DROP_SETTLE_STEPS = 60

# spec §9 (added 2026-08-15): the blended action is written by several layers, and the blend hides
# which one won -- the tensor looks identical whoever produced it. THREE separate defects in this
# file were one layer silently cancelling another, and none of them were visible in any log:
#   * freeze x relative action -- holding still by freezing a relative delta re-applied it every
#     step and drove W-B 43 mm past target
#   * release x jaw close      -- the gripper opened before the clamp existed and the part fell
#                                 through the through-bore
#   * creep x SETTLE           -- an ungated creep command overrode the very motion that carries
#                                 the tip through InsertionController's SETTLE -> PRESS window, so
#                                 W-A never left SETTLE in 12 of 12 episodes
# Every override therefore goes through `_take_action`, which records the owner per env per step.
# THE RULE: a change that adds an action override MUST pass a phase condition as its mask -- not a
# bare depth/distance test, and never an unconditional write. `tests/test_bdash_action_owner.py`
# fails the build if a raw `torch.where` override is added to this file instead.
ACTION_OWNERS = ("pick", "reerect", "touchoff", "insert", "creep", "hold")
OWNER_PICK, OWNER_REERECT, OWNER_TOUCHOFF, OWNER_INSERT, OWNER_CREEP, OWNER_HOLD = range(len(ACTION_OWNERS))
# Column 6 (the binary gripper) is tracked separately: it is a different actuator on a different
# schedule, and the release x jaw-close defect above lived entirely in it.
GRIP_OWNERS = ("pick", "release")
GRIP_PICK, GRIP_RELEASE = range(len(GRIP_OWNERS))


@dataclass
class BDashChuckTeacherArgs:
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> BDashChuckTeacherArgs:
        return cls()


@register_policy
class BDashChuckTeacherPolicy(PolicyBase):

    name = "bdash_chuck_teacher"
    config_class = BDashChuckTeacherArgs

    def __init__(self, config: BDashChuckTeacherArgs):
        super().__init__(config)
        self._pick: ScriptedChuckPick | None = None
        self._names: tuple[str, ...] = ()
        self._stations: torch.Tensor | None = None  # (K, 2): [side-lying, upright] axial station
        self._upright_cos: float = math.cos(math.radians(30.0))
        self._lift_side: float = 0.0
        self._debug = bool(os.environ.get("BDASH_DEBUG"))
        # grasp target frozen at CLOSE, so the wrist never servos to the part it already holds
        self._latched_point: torch.Tensor | None = None
        self._latched_quat: torch.Tensor | None = None
        self._latch_stale: torch.Tensor | None = None
        self._latched_axis: torch.Tensor | None = None
        self._latched_yaw: torch.Tensor | None = None
        # last-step diagnostics, read by scripts/bdash/check_chuck_teacher.py
        self.last_side_lying: torch.Tensor | None = None
        self.last_axis: torch.Tensor | None = None
        self.last_grasp: torch.Tensor | None = None
        self.last_yaw: torch.Tensor | None = None
        self.force_yaw_zero: bool = False
        # spec §4-3, load stage only. All None/absent on the pick stage.
        self._ins = None
        self._in_insertion: torch.Tensor | None = None
        self._chuck_xy: torch.Tensor | None = None
        self._insert_depths: torch.Tensor | None = None
        self._jaws = None
        self._env = None
        self._release_to_seat: torch.Tensor | None = None
        self._handover: torch.Tensor | None = None
        self._face_height: float = 0.080
        self._creep_band: float = 0.008
        self._seat_force: float = 8.0
        self._overshoot_comp: torch.Tensor | None = None
        self._creep_step: float = 0.0005
        self._handover_timer: torch.Tensor | None = None
        self._drop_timer: torch.Tensor | None = None
        self._prev_depth: torch.Tensor | None = None
        self._hold_tcp: torch.Tensor | None = None
        self._prev_tcp: torch.Tensor | None = None
        self.last_depth_nominal: torch.Tensor | None = None
        self.last_depth_reachable: torch.Tensor | None = None
        # spec §9: who wrote the action this step. (N,) int, indices into ACTION_OWNERS/GRIP_OWNERS.
        self._touch = None
        self._touch_done: torch.Tensor | None = None
        self._reerect = None
        self._reerect_done: torch.Tensor | None = None
        self._reerect_axis: torch.Tensor | None = None
        self._reerect_planned: torch.Tensor | None = None
        # (N,) bool: envs whose arm a BORROWED leg (re-erect, touch-off) is driving this step. The
        # pick's stall watchdog must stand down for exactly these, and the flag is rebuilt from
        # scratch every step so a leg that ends cannot leave the watchdog suppressed.
        self._borrowed: torch.Tensor | None = None
        self._reerect_start_gap: torch.Tensor | None = None
        # spec §5 gate + P1/P2/P3 policy modes (BDASH_POLICY_MODE). P1 = never measure, always
        # direct; P2 = never measure, ALWAYS refix via the pad; P3 = measure, then gate.
        self._mode: str = ""
        self._gate_decision: torch.Tensor | None = None
        self._gate_decided: torch.Tensor | None = None
        self._gate_e: torch.Tensor | None = None
        self._refix_done: torch.Tensor | None = None
        self._refix_planned: torch.Tensor | None = None
        self._reerect_stand_cos: torch.Tensor | None = None
        self._reerect_place_xy: torch.Tensor | None = None
        self._reerect_regrip_cos: torch.Tensor | None = None
        self._station_meas: torch.Tensor | None = None
        self._action_owner: torch.Tensor | None = None
        self._grip_owner: torch.Tensor | None = None

    # ------------------------------------------------------------------ setup
    def _ensure(self, env):
        if self._pick is not None:
            return
        from isaaclab_arena_environments.mdp.bdash_chuck_config import load_controllers_cfg

        self._env = env  # the jaws are SCENE state and reset() only gets env_ids
        task = env.unwrapped.cfg.isaaclab_arena_env.task
        cc = load_controllers_cfg()
        pick_cfg = cc["pick"]
        self._pick = ScriptedChuckPick(task.names, cc["ee_control"], pick_cfg)
        self._names = tuple(task.workpiece_names)
        self._upright_cos = math.cos(math.radians(float(pick_cfg["upright_angle_deg"])))

        # Per-workpiece axial grasp station, one column per pose class.
        #   side-lying -> a fixed distance from the leading end, on the Ø25 shaft for every variant
        #   upright    -> a fixed distance below the TOP, so a tall part cannot foul the hand
        side = float(pick_cfg["grip_station_side"])
        # Per-variant since 2026-08-15 (controllers.yaml explains why). A scalar is still accepted
        # so an older config, or the peg-style flat value, keeps working.
        below_top_cfg = pick_cfg["grip_below_top_upright"]

        def _below_top(variant: str) -> float:
            if isinstance(below_top_cfg, dict):
                return float(below_top_cfg[variant])
            return float(below_top_cfg)

        self._lift_side = float(pick_cfg.get("grip_lift_side", 0.0))
        dev = env.unwrapped.device
        # TODO(§0-3 continuous dimensions): these are cached ONCE, on the first step, and
        # `_ensure` returns early forever after. They are per-VARIANT today. When dimensions become
        # a per-spawn draw, this cache is the one place that will silently go stale — it must then
        # be refreshed per reset (or read straight from a per-env tensor written by the scatter).
        self._stations = torch.tensor(
            [[side, float(p["length"]) - _below_top(p["variant"])] for p in task.profiles],
            device=dev,
            dtype=torch.float32,
        )
        self._lengths = torch.tensor([float(p["length"]) for p in task.profiles], device=dev, dtype=torch.float32)
        self._radii = torch.tensor([float(p["max_radius"]) for p in task.profiles], device=dev, dtype=torch.float32)
        hand = task.cfg["target"]["hand"]
        self._hand_span = float(hand["half_span"])
        self._hand_width = float(hand["half_width"])
        self._hand_clear_z = float(hand["clear_z"])

        # spec §4-3. Built ONLY for the load stage: with `_ins` left None every branch below is
        # skipped and the teacher is bit-for-bit the pick-stage expert that produced GATE 1.
        if getattr(task, "stage", "pick") == "full":
            from isaaclab_arena.controllers.chuck_insertion_controller import ChuckInsertionController

            # InsertionController addresses the scene through the peg task's vocabulary
            # ("socket" is whatever it inserts INTO, at `root_z + mouth_height`). Alias rather
            # than rename: the peg path is frozen, and `scene_names` is what the chuck predicates,
            # the sampler and the recorder all read. `peg_sensor` is aliased for completeness --
            # ChuckInsertionController overrides `_read_wrench` and never reads it -- so that a
            # future call site gets a real sensor instead of a KeyError. `peg` and
            # `peg_finger_sensor` are only touched under BDASH_CHEAT_TIP / BDASH_REALIGN, which
            # this task never sets.
            ins_names = dict(task.names)
            ins_names["socket"] = task.names["chuck"]
            ins_names["peg_sensor"] = task.contact_sensor_names[0]
            self._ins = ChuckInsertionController(
                ins_names,
                mouth_height=float(task.geom["chuck_face_height"]),
                ee_cfg=cc["ee_control"],
                ins_cfg=cc["insertion"],
                grip_offset=0.0,  # replaced per env in get_action once the target is known
            )
            # FIXTURE-filtered, not the unfiltered net sensor: see ChuckInsertionController's
            # _read_wrench. The net force on a 0.4 kg held part is mostly the gripper carrying it.
            self._ins.sensor_names = tuple(task.fixture_sensor_names)
            # spec §4-1. Built only when enabled, so the default path is byte-for-byte the pipeline
            # that produced the existing numbers and the comparison is a clean A/B.
            self._mode = (os.environ.get("BDASH_POLICY_MODE") or "").upper()
            if os.environ.get("BDASH_TOUCH_OFF") or self._mode == "P3":
                from isaaclab_arena.controllers.touch_off_controller import TouchOffController

                # The probe reads the part-vs-fixture filtered force (see its step()); hand it the
                # sensor names the insertion already uses for the same reason.
                task.names.setdefault("fixture_sensors", list(task.fixture_sensor_names))
                self._touch = TouchOffController(
                    task.names,
                    # THE 仮置き台 IS THE DATUM. It was a separate 60x60 block at (0.18, 0.32)
                    # until 2026-08-22, which is the same shape doing the same job -- a surveyed
                    # flat-topped box -- 0.705 m from the chuck instead of 0.364 m. Touch-off runs
                    # immediately after the re-grip, i.e. while the arm is still AT the pad, so the
                    # separate block cost a 0.7 m round trip for nothing. (That travel is what made
                    # the probe spend ~850 steps in transit and time the episode out, which was
                    # patched by raising travel_step from 10 to 40 mm; the right fix was the layout.)
                    task.cfg["scene"]["reerect_pose"],
                    task.assets_cfg["reerect_pad"]["size"],
                    cc["ee_control"],
                    cc.get("touchoff", {}),
                    task.geom,
                )
                self._datum_offsets = torch.tensor(
                    [float(p["datum_offset"]) for p in task.profiles], device=dev, dtype=torch.float32
                )
            # spec §4-2. Built only when the scene declares a set-down point, so a config without
            # one is byte-for-byte the pipeline that produced the existing upright numbers.
            if "reerect_pose" in task.cfg["scene"]:
                from isaaclab_arena.controllers.reerect_controller import ReerectController

                self._reerect = ReerectController(
                    task.names,
                    task.cfg["scene"]["reerect_pose"],
                    cc["ee_control"],
                    cc.get("reerect", {}),
                )
            gate_cfg = cc.get("gate", {})
            go_mm = gate_cfg.get("go_below_mm", {})
            self._gate_go = torch.tensor(
                [float(go_mm.get(p["variant"], 1.0)) * 1e-3 for p in task.profiles], device=dev, dtype=torch.float32
            )
            self._gate_reject = float(gate_cfg.get("reject_above_mm", 15.0)) * 1e-3
            self._ring_h = float((task.assets_cfg["chuck"].get("work_stopper") or {}).get("height", 0.0))
            self._chuck_xy = torch.tensor(
                [float(v) for v in task.cfg["scene"]["chuck_pose"][:2]], device=dev, dtype=torch.float32
            )
            # Commanded insertion depth per variant (task.yaml chuck_load.insert_depth_m). This is
            # the PRESS -> RELEASE threshold, and it is per-variant, which is why it is not a
            # constant in controllers.yaml.
            depths = task.cfg["chuck_load"]["insert_depth_m"]
            self._insert_depths = torch.tensor(
                [float(depths[p["variant"]]) for p in task.profiles], device=dev, dtype=torch.float32
            )
            self._in_insertion = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=dev)

            # spec §4-3 閉爪. The chuck's own actuator, not the arm's.
            from isaaclab_arena.controllers.chuck_jaws import ChuckJaws
            from isaaclab_arena_environments.mdp import bdash_chuck_assets

            chuck_geom = bdash_chuck_assets.chuck_geometry()
            self._jaws = ChuckJaws(
                jaw_names=[f"bdash_chuck_jaw_{k}" for k in range(3)],
                chuck_xy=tuple(float(v) for v in task.cfg["scene"]["chuck_pose"][:2]),
                jaw_z=float(task.cfg["scene"]["chuck_pose"][2])
                + float(chuck_geom["body_height"])
                - float(chuck_geom["jaw_height"]),
                radius_closed=float(chuck_geom["jaw_radius_low"]),
                cfg=cc.get("jaws", {}),
            )
            # Which variants must be RELEASED to seat: their datum feature cannot follow the part
            # into the bore while the fingers hold it, so the last stretch is a guided drop.
            self._face_height = float(task.geom["chuck_face_height"])
            self._seat_force = float(task.cfg["chuck_load"].get("seat_force", 8.0))
            comp = cc.get("overshoot_comp", {}) or {}
            self._overshoot_comp = torch.tensor(
                [float(comp.get(p["variant"], 0.0)) for p in task.profiles], device=dev, dtype=torch.float32
            )
            self._creep_band = float(cc["insertion"]["creep_band"])
            self._creep_step = float(cc["insertion"]["creep_max_pos_step"])
            self._release_to_seat = torch.tensor([d > 0.5 for d in task._load_targets()["force_seated"]], device=dev)

    # -------------------------------------------------------------------- §9
    def _begin_action(self, action: torch.Tensor) -> None:
        """Start the step with the pick owning every env. Call once, right after ``_pick.step``."""
        owner = torch.full((action.shape[0],), OWNER_PICK, dtype=torch.int8, device=action.device)
        self._action_owner = owner
        self._grip_owner = torch.full_like(owner, GRIP_PICK)

    def _take_action(self, action: torch.Tensor, new_action: torch.Tensor, mask: torch.Tensor, owner: int):
        """Override the pose columns where ``mask``, recording ``owner`` for those envs (spec §9).

        ``mask`` must be a PHASE CONDITION -- an FSM state, or a state conjoined with a threshold --
        never a bare threshold and never all-True. See the ACTION_OWNERS comment for the three
        defects this rule exists to catch.
        """
        assert mask.dtype == torch.bool and mask.shape == (action.shape[0],), "override mask must be (N,) bool"
        self._action_owner = torch.where(mask, torch.full_like(self._action_owner, owner), self._action_owner)
        return torch.where(mask.unsqueeze(-1), new_action, action)

    def _take_gripper(self, action: torch.Tensor, value: float, mask: torch.Tensor, owner: int):
        """Same contract as :meth:`_take_action`, for column 6 (the binary gripper)."""
        assert mask.dtype == torch.bool and mask.shape == (action.shape[0],), "override mask must be (N,) bool"
        self._grip_owner = torch.where(mask, torch.full_like(self._grip_owner, owner), self._grip_owner)
        action = action.clone()
        action[:, 6] = torch.where(mask, torch.full_like(action[:, 6], value), action[:, 6])
        return action

    def reerect_stats(self) -> dict:
        """Per-episode diagnostics for the §4-2 leg. Reporting only; nothing reads these back."""
        if self._reerect is None or self._reerect.phase is None:
            return {}
        import math as _math

        out = dict(self._reerect.stats())
        out["reerect_done"] = bool(self._reerect_done[0]) if self._reerect_done is not None else False
        out["reerect_planned"] = bool(self._reerect_planned[0]) if self._reerect_planned is not None else False
        # The angle the turn was PLANNED to cover. A leg that runs its full step budget while this
        # reads ~0 is a leg whose plan never took, which is indistinguishable from a leg that turned
        # and slipped if only the outcome is logged.
        q0, q1 = self._reerect.q_start[:1], self._reerect.q_end[:1]
        dot = float((q0 * q1).sum().abs().clamp(max=1.0))
        out["reerect_plan_deg"] = round(_math.degrees(2.0 * _math.acos(dot)), 2)
        if getattr(self, "_reerect_start_gap", None) is not None:
            out["reerect_start_gap"] = round(float(self._reerect_start_gap[0]), 4)
        if getattr(self, "_reerect_regrip_cos", None) is not None:
            c = float(self._reerect_regrip_cos[0])
            out["reerect_regrip_deg"] = round(_math.degrees(_math.acos(min(1.0, max(-1.0, c)))), 2)
        if getattr(self, "_reerect_stand_cos", None) is not None:
            c = float(self._reerect_stand_cos[0])
            # Tilt from VERTICAL either way up, plus the sign that says which way up.
            out["place_tilt_deg"] = round(_math.degrees(_math.acos(min(1.0, abs(c)))), 2)
            out["place_correct_end"] = c > 0.0
            out["place_stood"] = abs(c) > _math.cos(_math.radians(30.0))
            out["reerect_stand_deg"] = out["place_tilt_deg"]
        if getattr(self, "_reerect_place_xy", None) is not None:
            out["place_xy_err_mm"] = round(float(self._reerect_place_xy[0]) * 1e3, 2)
        return out

    def touch_off_stats(self) -> dict:
        """What the §4-1 probe actually measured, for the JSONL sidecar.

        Logged because the failure mode is silent: a probe that never touches leaves `station_hat`
        at zero, and a zero station makes the loader's depth arithmetic report a part as seated
        while the arm is still nowhere near the chuck.
        """
        if self._touch is None or self._touch.measured is None:
            return {}
        out = {
            "touch_measured": bool(self._touch.measured[0]),
            "touch_done": bool(self._touch_done[0]) if self._touch_done is not None else False,
            "touch_phase": int(self._touch.phase[0]) if self._touch.phase is not None else -1,
            "touch_face": int(self._touch.face_idx[0]) if self._touch.face_idx is not None else -1,
        }
        if self._touch.station_hat is not None:
            out["station_hat"] = round(float(self._touch.station_hat[0]), 5)
        if self._touch.p_hat is not None:
            out["p_hat"] = round(float(self._touch.p_hat[0]), 5)
        return out

    def action_owner_stats(self) -> dict:
        """Owner of env 0's action this step, for the JSONL sidecar (spec §9)."""
        if self._action_owner is None:
            return {}
        return {
            "action_owner": ACTION_OWNERS[int(self._action_owner[0])],
            "grip_owner": GRIP_OWNERS[int(self._grip_owner[0])],
        }

    # ------------------------------------------------------------------- §4-3
    def _blend_reerect(self, uenv, rows, target, station, pick_action, pick_finished):
        """Turn a side-lying part upright, set it down, and send the pick round again.

        Returns the possibly-overridden action, a ``pick_finished`` that stays False until the part
        has been re-erected AND re-picked, and the station -- which changes, because the second pick
        grips at the upright station rather than the side one.

        §9. The turn's endpoints are planned ONCE, at the moment the leg is entered, from the
        CLOSE-time latched quaternion, the axis latched with it, and the commanded grip offset. From
        then on the trajectory is a slerp in a phase counter. Nothing the arm can observe about the
        part it is holding changes where the turn ends, which is the ruling's condition, and it is
        enforced structurally: ``ReerectController.step`` is not given the part's pose at all.
        """
        n = uenv.num_envs
        dev = pick_action.device
        if self._reerect_done is None or self._reerect_done.shape[0] != n:
            self._reerect_done = torch.zeros(n, dtype=torch.bool, device=dev)
            self._reerect_planned = torch.zeros(n, dtype=torch.bool, device=dev)

        # Only parts that SPAWNED on their side take this route. Read from the reset-time flag, not
        # from the live axis: after the turn the live axis says upright, so a live test would end
        # the leg the instant it succeeded and re-enter it if the part wobbled.
        side = getattr(uenv, "bdash_side_lying", None)
        if side is None:
            return pick_action, pick_finished, station
        side = side[rows, target] if side.dim() > 1 else side
        arm = pick_finished & side & ~self._reerect_done
        if not bool(arm.any()):
            return pick_action, pick_finished, station

        # Plan on entry, once per env. `self.last_axis` is the axis latched with the grasp quat, so
        # the plan is a function of the CLOSE-time pose and the commanded station only.
        fresh = arm & ~self._reerect_planned
        if bool(fresh.any()):
            # REPORTING ONLY (never read back into the trajectory): how far the part is from the TCP
            # at the moment the leg takes over. The leg is open loop by §9, so it will run its whole
            # motion whether or not anything is in the fingers -- and an empty hand turning in mid
            # air looks, in every other log line, exactly like a successful turn. This number is
            # what tells the two apart.
            names = self._names
            pos_all = torch.stack([uenv.scene[n].data.root_pos_w for n in names], dim=1)
            tcp = read_ee_pose(uenv)[0]
            self._reerect_start_gap = (pos_all[rows, target] - tcp).norm(dim=-1)
            self._reerect.plan(self._latched_quat, self._latched_axis, station, fresh)
            self._reerect_planned = self._reerect_planned | fresh

        # Snapshot BEFORE the step, because "the leg finished" and "the part is back in the hand"
        # are different events one step apart. Using the post-step flag would let `pick_finished`
        # go true on the very step the part is released and standing free on the bench, which
        # latches the insertion on with an empty gripper.
        was_done = self._reerect_done.clone()
        self._borrowed = self._borrowed | arm
        action, want_open, finished = self._reerect.step(uenv, arm)

        if os.environ.get("BDASH_REERECT_TRACE") and bool(arm[0]):
            # Per-step trace of the one thing the leg cannot see for itself. It is open loop by §9,
            # so it turns whether or not anything is still in the fingers -- and every other log line
            # looks identical either way. Printing, not feeding back.
            from isaaclab_arena.controllers.reerect_controller import PHASE_NAMES
            from isaaclab_arena_environments.mdp.bdash_chuck_predicates import gripper_width

            _tcp = read_ee_pose(uenv)[0]
            _pos = torch.stack([uenv.scene[n].data.root_pos_w for n in self._names], dim=1)[rows, target]
            _gap = float((_pos[0] - _tcp[0]).norm())
            _fz = float(_pos[0, 2])
            print(
                f"[reerect] phase={PHASE_NAMES[int(self._reerect.phase[0])]:8s} "
                f"t={int(self._reerect.timer[0]):3d}/{int(self._reerect.turn_len[0]):3d} "
                f"gap={_gap * 1e3:6.1f}mm part_z={_fz:.4f} tcp_z={float(_tcp[0, 2]):.4f} "
                f"grip_cmd={float(action[0, 6]):+.1f} width={float(gripper_width(uenv)[0]) * 1e3:5.1f}mm",
                flush=True,
            )
        pick_action = self._take_action(pick_action, action, arm, OWNER_REERECT)
        pick_action = self._take_gripper(pick_action, 1.0, arm & want_open, GRIP_RELEASE)

        # A leg that GAVE UP has not stood the part up, so re-running the pick on it would send the
        # arm back to the tray for a part that is still in the fingers -- the same collision that
        # produced `load_failed` before the watchdog fix. Let the flow continue instead: the
        # insertion that follows fails honestly on depth, and the record carries `reerect_gave_up`
        # and the phase it stuck in, which is the fact worth having.
        gave_up = self._reerect.gave_up if self._reerect.gave_up is not None else torch.zeros_like(arm)
        just_done = arm & finished & ~gave_up
        self._reerect_done = self._reerect_done | (arm & gave_up)
        if bool(just_done.any()):
            # REPORTING ONLY: how straight the part is STANDING, before the re-pick touches it.
            # Splits the two ways the seated angle can go wrong -- the part landing crooked on the
            # pad, or the re-grip tilting a part that had landed straight. Measured, the upright
            # path seats at 0.2-0.5 deg median while the re-erected one came in at 4.3 deg, so the
            # tilt is specific to this leg; which half of it owns the error is what this says.
            # CP1b, the teacher's PLACE quality. Measured at the instant the leg lets go and the
            # part has settled, which is exactly the state the VLA's demonstrations will terminate
            # in, so this is the bar the recorded data is worth. Privileged and REPORTING ONLY --
            # nothing reads it back, and the leg has already finished by the time it is taken.
            #
            # Four numbers, because "it placed the part" hides four different ways of not doing so:
            # it fell over, it went down the wrong way up, it stands but cocked, or it stands but
            # off the pad. The re-grip that follows fails differently for each.
            quat_all = torch.stack([uenv.scene[n].data.root_quat_w for n in self._names], dim=1)
            pos_all = torch.stack([uenv.scene[n].data.root_pos_w for n in self._names], dim=1)
            lz = torch.zeros(uenv.num_envs, 3, device=dev, dtype=quat_all.dtype)
            lz[:, 2] = 1.0
            stand_axis = quat_apply(quat_all[rows, target], lz)
            # SIGNED, not abs: the axis runs leading -> trailing, so it points UP exactly when the
            # leading end is the one on the pad. That sign IS the correct-end test.
            self._reerect_stand_cos = stand_axis[:, 2]
            station_xy = torch.tensor(
                [float(v) for v in self._reerect.station_pose[:2]], device=dev, dtype=pos_all.dtype
            )
            self._reerect_place_xy = (pos_all[rows, target][:, :2] - station_xy).norm(dim=-1)
            # Hand the part back to the pick controller as if it had never been touched. The pick's
            # own reset drops its phase to the start; the grasp latches are rebuilt on the next step
            # because `phase < CLOSE` again, so the second approach tracks the part's live pose --
            # which is allowed, and is not the forbidden feedback: the part is on the bench now, not
            # in the fingers, so its pose is no longer a function of the arm's own output.
            self._pick.reset(torch.nonzero(just_done, as_tuple=False).flatten())
            self._reerect_done = self._reerect_done | just_done
            # The LATCHES ARE NOT CLEARED HERE, and that is deliberate. Resetting the pick drops its
            # phase below CLOSE, so `get_action`'s latch block already overwrites them with the live
            # values on the very next step -- clearing them buys nothing and costs correctness: they
            # are still read LATER IN THIS SAME STEP by `_handover_to_chuck`, which is downstream of
            # this call. Setting them to None crashed there with "Cannot access data pointer of
            # Tensor that doesn't have storage", one step after every successful re-erect, and Isaac
            # Sim swallowed the traceback so it presented as a frozen sim.

        # An env that still owes a re-erect is not finished picking, whatever the pick FSM says --
        # and neither is one that finished the leg on THIS step, whose pick has just been reset and
        # has not yet re-approached the part now standing on the bench.
        pick_finished = pick_finished & (was_done | ~side)
        return pick_action, pick_finished, station

    def _blend_gate_refix(self, uenv, rows, target, station, pick_action, pick_finished):
        """spec §5: decide GO / REFIX / REJECT, and run the refix route where it is chosen.

        P2 never measures and refixes EVERY part -- the "always pay the jig tax" baseline. P3 gates
        on the touch-off's measured axial error against the per-variant budget. The refix itself is
        the place-only pass of the re-erect machinery (plan_refix): stand the part on the 仮置き台,
        let go, re-grasp at the commanded station -- which resets the grip error to process-nominal
        (CP1b: 100% stood, 0.00 deg tilt, over 50 places).

        A REJECTed part is held still, closed, and reported; the harness ends the episode on the
        decision. The hold is BORROWED -- the arm is deliberately not following the pick's waypoint,
        and without the mark the pick watchdog reads the stillness as a stall and retries a pick
        that already succeeded.

        One reerect-controller instance serves both this and the side-lying leg; E1 runs upright
        only, so the two are never active together. If side-lying ever joins a gated run, the
        instance needs splitting.
        """
        n = uenv.num_envs
        dev = pick_action.device
        if self._refix_done is None or self._refix_done.shape[0] != n:
            self._refix_done = torch.zeros(n, dtype=torch.bool, device=dev)
            self._refix_planned = torch.zeros(n, dtype=torch.bool, device=dev)
        if self._gate_decision is None or self._gate_decision.shape[0] != n:
            self._gate_decision = torch.full((n,), GO, dtype=torch.long, device=dev)
            self._gate_decided = torch.zeros(n, dtype=torch.bool, device=dev)

        if self._mode == "P3" and self._touch is not None and self._touch_done is not None:
            undecided = self._touch_done & ~self._gate_decided
            if bool(undecided.any()) and self._gate_e is not None:
                measured = (
                    (self._touch.measured & self._touch_done)
                    if self._touch.measured is not None
                    else torch.zeros(n, dtype=torch.bool, device=dev)
                )
                dec = gate_decide(
                    self._gate_e,
                    measured,
                    self._gate_go[target][rows],
                    torch.full((n,), self._gate_reject, device=dev, dtype=self._gate_e.dtype),
                )
                # Latched: the decision is made ONCE per part, on the measurement that was taken.
                # Re-evaluating later would read a cleared measurement as "unmeasured -> refix" and
                # loop the part through the pad forever.
                self._gate_decision = torch.where(undecided, dec, self._gate_decision)
                self._gate_decided = self._gate_decided | undecided
            need = pick_finished & (self._gate_decision == REFIX) & ~self._refix_done
            rej = pick_finished & (self._gate_decision == REJECT)
        else:  # P2
            need = pick_finished & ~self._refix_done
            rej = torch.zeros(n, dtype=torch.bool, device=dev)

        if bool(rej.any()):
            self._borrowed = self._borrowed | rej
            pick_action = self._take_action(pick_action, torch.zeros_like(pick_action), rej, OWNER_HOLD)
            pick_action = self._take_gripper(pick_action, -1.0, rej, GRIP_PICK)
        pick_finished = pick_finished & ~rej

        if bool(need.any()):
            fresh = need & ~self._refix_planned
            if bool(fresh.any()):
                # `station` here is the MEASURED value where touch-off produced one (P3), so the
                # set-down height is computed from where the part actually sits in the fingers --
                # the measurement is what makes the place gentle. P2 plans from the commanded value
                # and eats the drop, which is part of what that baseline costs.
                self._reerect.plan_refix(self._latched_quat, station, fresh)
                self._refix_planned = self._refix_planned | fresh
            self._borrowed = self._borrowed | need
            action, want_open, finished = self._reerect.step(uenv, need)
            pick_action = self._take_action(pick_action, action, need, OWNER_REERECT)
            pick_action = self._take_gripper(pick_action, 1.0, need & want_open, GRIP_RELEASE)

            gave = self._reerect.gave_up if self._reerect.gave_up is not None else torch.zeros_like(need)
            done = need & finished
            if bool(done.any()):
                self._pick.reset(torch.nonzero(done, as_tuple=False).flatten())
                self._refix_done = self._refix_done | done
                # The second pass inserts directly: the re-grasp reset the error, so the decision
                # for these envs becomes GO -- and the stale measurement must not re-apply, so it
                # is cleared rather than trusted.
                self._gate_decision = torch.where(
                    done & ~gave, torch.full_like(self._gate_decision, GO), self._gate_decision
                )
                if self._touch is not None and self._touch.measured is not None:
                    self._touch.measured = self._touch.measured & ~done
            pick_finished = pick_finished & ~need
        return pick_action, pick_finished, station

    def gate_stats(self) -> dict:
        """Per-episode gate diagnostics for the E1 JSONL. Reporting only."""
        if not self._mode:
            return {}
        from isaaclab_arena.controllers.budget_gate import DECISION_NAMES

        out = {"policy_mode": self._mode}
        if self._gate_e is not None:
            out["gate_e_mm"] = round(float(self._gate_e[0]) * 1e3, 2)
        if self._gate_decided is not None and bool(self._gate_decided[0]):
            out["gate_decision"] = DECISION_NAMES[int(self._gate_decision[0])]
        if self._refix_done is not None:
            out["refixed"] = bool(self._refix_done[0])
        return out

    def _blend_insertion(self, env, uenv, rows, target, station, pick_action, pick_finished):
        """Hand the arm over to the chuck-insertion controller once the pick is done.

        The idiom is the peg pipeline's (bdash_scripted_policy.py:90-156) and it is deliberately
        the same: step BOTH controllers every frame, gate the insertion side with an ``active``
        mask so it accumulates no state before the handoff, and select with ``torch.where``. The
        pick keeps holding its transport waypoint after DONE (scripted_pick.py:157), so it never
        fights the takeover.

        The latch is monotone: once an env is inserting it stays inserting. There is no way back to
        the tray in this stage -- returning a part is the gate's ``no-go`` route (spec §5), which is
        not implemented.
        """
        if self._in_insertion is None or self._in_insertion.shape[0] != uenv.num_envs:
            self._in_insertion = torch.zeros(uenv.num_envs, dtype=torch.bool, device=pick_action.device)

        self._borrowed = torch.zeros(uenv.num_envs, dtype=torch.bool, device=pick_action.device)

        # Carry the lift over the bore. Without this the pick's TRANSPORT waypoint is the lift
        # itself, so the phase is satisfied on entry and the part never leaves the tray.
        #
        # A part that still owes a re-erect goes to the SET-DOWN STATION instead, because the bore
        # is not where it is going: routing it over the chuck first costs a 0.36 m round trip and
        # carries a part the chuck cannot accept directly across the open bore on the way. The
        # second pass -- after the part is standing and has been picked again -- takes the chuck
        # branch, since `_reerect_done` is set by then.
        transport = self._chuck_xy.expand(uenv.num_envs, 2)
        if self._reerect is not None and self._reerect_done is not None:
            side = getattr(uenv, "bdash_side_lying", None)
            if side is not None:
                side = side[rows, target] if side.dim() > 1 else side
                station_xy = torch.tensor(
                    [float(v) for v in self._reerect.station_pose[:2]],
                    device=transport.device,
                    dtype=transport.dtype,
                ).expand(uenv.num_envs, 2)
                transport = torch.where((side & ~self._reerect_done).unsqueeze(-1), station_xy, transport)
        self._pick.transport_xy = transport

        # RE-ERECT LEG (spec §4-2), the FIRST thing after the lift, because everything after it --
        # touch-off, transport, insertion -- must see the part in the pose it will be loaded in.
        #
        # Why it has to exist at all: a side-lying part is gripped ACROSS its axis 30 mm from the
        # leading end, and insertion can only reach `station - 10.9 mm` before the fingers arrive at
        # the chuck face. That is 19.1 mm against a commanded 40-72 mm, so the hole is unreachable
        # from the pose the part was picked in, no matter how well the arm servos. Measured before
        # this leg existed: 20 of 20 side-lying episodes never entered the bore.
        #
        # The leg ends by RESETTING THE PICK, which is the whole trick: once the part is standing on
        # the bench it is an ordinary upright part, so the second pass through the same pick
        # controller grips it at the upright station and every leg downstream is the upright path
        # that is already measured. Nothing below this point knows the part was ever on its side.
        if self._reerect is not None:
            pick_action, pick_finished, station = self._blend_reerect(
                uenv, rows, target, station, pick_action, pick_finished
            )

        # TOUCH-OFF LEG (spec §4-1), between the lift and the transport. The part is already held,
        # so this is the one moment its true grip station can be measured without a camera: touch
        # the surveyed datum block and read the TCP. `station` is then the MEASURED value for
        # everything downstream -- tip offset, reachable depth, creep target -- rather than the
        # commanded one, which is only right if the part never slipped.
        if self._touch is not None:
            if self._touch_done is None or self._touch_done.shape[0] != uenv.num_envs:
                self._touch_done = torch.zeros(uenv.num_envs, dtype=torch.bool, device=pick_action.device)
            probing = pick_finished & ~self._touch_done
            if bool(probing.any()):
                radius = self._radii[target][rows]
                datum = self._datum_offsets[target][rows]
                self._borrowed = self._borrowed | probing
                touch_action, touch_finished = self._touch.step(env, probing, station, radius, datum)
                self._touch_done = self._touch_done | (probing & touch_finished)
                pick_action = self._take_action(pick_action, touch_action, probing, OWNER_TOUCHOFF)
            # Use the measurement only where it actually registered: a probe that never felt the
            # block must fall back to the commanded station rather than to whatever zero means.
            if self._touch.measured is not None:
                meas = self._touch.measured & self._touch_done
                # The gate's input: measured-minus-commanded, captured BEFORE the replacement below
                # overwrites the commanded value. Grip error passes 1:1 into protrusion, so this
                # single number is the budget the gate routes on.
                if self._gate_e is None or self._gate_e.shape[0] != uenv.num_envs:
                    self._gate_e = torch.zeros(uenv.num_envs, device=station.device, dtype=station.dtype)
                if os.environ.get("BDASH_TOUCH_DIAG") and bool(meas[0]):
                    print(
                        f"[gatein] station_hat={float(self._touch.station_hat[0]) * 1e3:.2f}mm "
                        f"station_cmd={float(station[0]) * 1e3:.2f}mm "
                        f"e={float((self._touch.station_hat - station)[0]) * 1e3:+.2f}mm",
                        flush=True,
                    )
                self._gate_e = torch.where(meas, self._touch.station_hat - station, self._gate_e)
                station = torch.where(meas, self._touch.station_hat, station)
                self._station_meas = self._touch.station_hat
            # The insertion may not start until the probe is finished, or the arm would be pulled
            # toward the chuck mid-touch and the contact z would be meaningless.
            pick_finished = pick_finished & self._touch_done

        # spec §5: the three-way gate and its two non-GO routes (P2 refixes everything, P3 routes on
        # the measurement). Sits between the measurement and the insertion because that is what it
        # IS -- the decision about whether the measured grip may proceed to the chuck.
        if self._mode in ("P2", "P3") and self._reerect is not None:
            pick_action, pick_finished, station = self._blend_gate_refix(
                uenv, rows, target, station, pick_action, pick_finished
            )

        # Proprioceptive tip offset and PRESS -> RELEASE threshold, computed AFTER the touch-off and
        # gate legs so a measured station actually reaches the insertion. This block used to sit
        # before them, which quietly re-imposed the COMMANDED station every step: the measurement
        # replaced the local variable and nothing downstream ever saw it. (The fingers cannot enter
        # the bore -- a finger spans radius 12.5-38.7 mm against a 20 mm bore -- so reachable depth
        # is `station - _FINGERTIP_BELOW_TCP - _FACE_MARGIN`, clamped, shortfall exposed.)
        self._ins.grip_offset = station
        nominal = self._insert_depths[target][rows]
        # The held-depth limit is set by the FINGERS LANDING ON THE WORK STOPPER, not by the chuck
        # face: the stopper is a ring standing `ring_h` proud of the face, and the fingertips stop
        # at ring-top + fingertip offset. Measured: TCP froze at 98.8 mm = 90 + 8.9, exactly, on
        # every W-A attempt -- the old face-based formula overstated reach by the ring height and
        # commanded depths the arm then pressed against forever.
        reachable = station - _FINGERTIP_BELOW_TCP - self._ring_h - _FACE_MARGIN
        self.last_depth_nominal = nominal
        self.last_depth_reachable = reachable
        self._ins.cfg["seated_depth"] = torch.minimum(nominal, reachable)

        # LATCH ONLY OVER THE CHUCK. A borrowed leg (the probe) can leave the arm parked at the
        # pad with the TCP at ~105 mm; latching there starts the insertion with a held part sweeping
        # toward the chuck at bore height, and it clips the fixture on the way in -- measured, the
        # GO path spent 1273 steps in SETTLE with the part wedged 14.5 mm into the bore mouth. The
        # pick is DONE and still holds its transport waypoint, so simply not latching until the arm
        # is back on it lets the existing pick action fly the arm home first.
        tcp_now = read_ee_pose(uenv)[0]
        at_transport = (torch.norm(tcp_now[:, :2] - self._chuck_xy.unsqueeze(0), dim=-1) < 0.05) & (
            tcp_now[:, 2] > 0.14
        )
        self._in_insertion = self._in_insertion | (pick_finished & at_transport)
        # Stand the pick's stall watchdog down wherever the arm is no longer following the PICK's
        # waypoint -- insertion, and equally the borrowed legs (re-erect, touch-off).
        #
        # Leaving the borrowed legs out was a real defect, not a tidiness point. The re-erect leg
        # drives the arm from the chuck to the set-down station, i.e. AWAY from the transport
        # waypoint the pick is still holding, so the pick sees its distance grow, calls a stall, and
        # drops `phase` back to APPROACH -- which pulls `pick_finished` low, freezes the leg
        # mid-TRANSIT, and sends the arm back toward the tray with the part still in the fingers.
        # Measured: the leg never left TRANSIT in any episode (72 active steps, retries = 1,
        # stall_steps = 4) while its plan was a correct 90 deg turn that never got to run.
        #
        # The same hole was open for the touch-off leg, which walks the arm to the datum block for
        # the same reason; it had not been caught because that leg has not been measured yet.
        self._pick.watchdog_suppress = self._in_insertion | self._borrowed
        # Drop out of `active` once the handover starts: the arm must HOLD STILL while the chuck
        # takes the part. Left active, the insertion controller keeps pressing for the whole
        # `close_steps` stroke -- measured, that drove W-B 27 mm past its commanded depth (a +1.3 mm
        # protrusion error became -27 mm) because it kept descending while the jaws travelled.
        # This is the real machine's order too: the robot stops, then the chuck clamps.
        pressing = self._in_insertion & ~self._handover_mask(uenv, pick_action)
        ins_action = self._ins.step(env, active=pressing)
        action = self._take_action(pick_action, ins_action, self._in_insertion, OWNER_INSERT)
        return self._handover_to_chuck(env, uenv, rows, target, station, action)

    def _handover_mask(self, uenv, like: torch.Tensor) -> torch.Tensor:
        """The envs whose part is being handed to the chuck. Lazily sized on first use."""
        if self._handover is None or self._handover.shape[0] != uenv.num_envs:
            self._handover = torch.zeros(uenv.num_envs, dtype=torch.bool, device=like.device)
        return self._handover

    def _handover_to_chuck(self, env, uenv, rows, target, station, action):
        """spec §4-3 tail: give the part to the chuck, then let go.

        The bore is a through hole, so the ORDER is not free and it is not the same for every
        variant:

        * depth-seated (W-A/W-B): the fingers can hold the part at its commanded depth, so the jaws
          close FIRST and the gripper opens after. Releasing first would drop the part through.
        * flange-seated (W-C): the flange lands on the chuck face at 72 mm, and the fingers cannot
          follow it below the face (they span radius 12.5-38.7 mm against a bore radius of 20 mm).
          So the gripper opens at the deepest it can reach and the part drops the last stretch,
          guided by the bore, before the jaws close on it. Measured without this: W-C sat 5.3 mm
          proud of nominal in 11 of 12 episodes and QC passed once.

        The whole sequence hangs off `_handover`, a monotone latch set when the insertion controller
        reports it has arrived -- not off a ground-truth pose, so nothing privileged enters the
        control path (§0-5).
        """
        from isaaclab_arena.controllers.ee_control import read_ee_pose

        if self._handover is None or self._handover.shape[0] != uenv.num_envs:
            self._handover = torch.zeros(uenv.num_envs, dtype=torch.bool, device=action.device)
        if self._handover_timer is None or self._handover_timer.shape[0] != uenv.num_envs:
            self._handover_timer = torch.zeros(uenv.num_envs, dtype=torch.long, device=action.device)
        if self._drop_timer is None or self._drop_timer.shape[0] != uenv.num_envs:
            self._drop_timer = torch.zeros(uenv.num_envs, dtype=torch.long, device=action.device)
        # Latch on the commanded depth, not on the controller reaching RELEASE. RELEASE fires
        # AFTER `depth > seated_depth`, so by then the part is already a millimetre or two past
        # target and freezing there banks that overshoot: measured, W-B settled at -2.3 to -3.8 mm
        # of protrusion error against a +-1.5 mm window. The depth here is the same proprioceptive
        # quantity `cp.loaded` uses, so the trigger and the termination cannot disagree.
        tcp, tcp_quat = read_ee_pose(env)
        # The tool held LEVEL. `rot_tol` (0.06 rad = 3.4 deg) lets the insertion finish with the
        # wrist that far off vertical, which alone exceeds the 0.5 deg QC bar; the workpiece axis
        # follows the tool, so squaring the tool squares the part.
        level = torch.zeros(uenv.num_envs, 4, device=tcp.device, dtype=tcp.dtype)
        level[:, 1] = 1.0
        z_axis = torch.zeros(uenv.num_envs, 3, device=tcp.device, dtype=tcp.dtype)
        z_axis[:, 2] = 1.0
        level = quat_mul(quat_from_angle_axis(self._latched_yaw, z_axis), level)
        if self._prev_tcp is None or self._prev_tcp.shape[0] != uenv.num_envs:
            self._prev_tcp = tcp.clone()
        face_z = uenv.scene[self._pick.names["chuck"]].data.root_pos_w[:, 2] + self._face_height
        depth = face_z - (tcp[:, 2] - station)
        speed = torch.norm(tcp - self._prev_tcp, dim=-1) / max(uenv.step_dt, 1e-6)
        self._prev_tcp = tcp.clone()
        # Measured calibration, written by scripts/bdash/calibrate_overshoot.py (never by hand,
        # spec §9). Negative protrusion error means the part sits deeper than nominal, so the
        # commanded depth comes down by that much.
        target_depth = self._ins.cfg["seated_depth"] + self._overshoot_comp[target][rows]

        # DEPTH ALONE fires the handover. Requiring low speed as well was a design error: a part
        # with nothing to stop it never slows, so the conjunction never held and the descent ran on.
        # Measured, W-B reached 82.5-83.0 mm against a commanded 40 mm and left an 80 mm bore
        # entirely; W-C only ever satisfied it because its flange hits the chuck face. Low speed is
        # the right test for SETTLE COMPLETE (below), not for "have I arrived".
        # PREDICTIVE TRIGGER. Firing on `depth >= target` fires one step LATE by construction, and
        # the arm carries on for the command already in flight: measured, every variant overshot by
        # a uniform +1.1 mm, about two creep steps of tracking lag. Extrapolating the measured depth
        # rate over that lag cancels it, and it self-corrects -- a slower descent extrapolates less.
        if self._prev_depth is None or self._prev_depth.shape[0] != uenv.num_envs:
            self._prev_depth = depth.clone()
        rate = (depth - self._prev_depth) / max(uenv.step_dt, 1e-6)
        self._prev_depth = depth.clone()
        predicted = depth + rate.clamp(min=0.0) * uenv.step_dt * _TRIGGER_LOOKAHEAD_STEPS
        force_seat = self._release_to_seat[target][rows]
        fz = torch.zeros_like(depth) if self._ins.last_wrench is None else self._ins.last_wrench[:, 2].abs()
        # A flange-seated variant arrives when the chuck pushes back, not at a commanded depth: its
        # seat is a hard stop, and the depth that reaches it is unreachable while the part is held.
        # WITHIN ONE CREEP STEP counts as arrived. The predictive trigger extrapolates the descent
        # rate to cancel in-flight lag, but during CREEP the rate is ~0.3-0.5 mm/step, so the
        # prediction barely leads -- measured, W-A crawled to 48.8 and 49.0 mm against a 50.0 mm
        # target and the trigger never fired: the arm sat one PD tracking-lag short of the target
        # forever, jaws never commanded, nothing terminated. One creep step is the resolution the
        # approach actually has, and firing inside it costs at most 0.5 mm of protrusion -- a
        # quarter of the tightest QC window.
        # ...and ONLY while already inside the creep band. The prediction extrapolates the raw
        # descent rate, which during the insertion APPROACH is several mm/step -- extrapolated over
        # the lookahead that is tens of mm, and a transient spike can cross the target from 40 mm
        # away. Measured: lowering the arrival threshold by a single 0.5 mm exposed exactly that --
        # three runs froze mid-approach at 7.5/13.6/22.1 mm depth, the arm parked by a trigger that
        # fired while it was still diving. Depth inside the band is slow by construction (the creep
        # override caps it), so inside the band the extrapolation is honest and outside it it is
        # not evidence at all.
        near = predicted.ge(target_depth - 2.0 * self._creep_step) & depth.ge(target_depth - self._creep_band)
        # Force-seated variants ALSO arrive by depth. Their seat sits BELOW the held-depth limit
        # (the fingers stop on the ring before the flange reaches it), so held force never comes:
        # measured, W-C pressed its ring-stop at fz = 0 forever. Depth-arrival at the clamped
        # reachable target hands over, and the drop below closes the last gap onto the seat.
        arrived = self._in_insertion & (near | (force_seat & fz.gt(self._seat_force)))

        # TWO-STAGE APPROACH. Inside `creep_band` of the target, stop letting the insertion
        # controller pick the step and drive the tip straight at the commanded depth with a small
        # step cap, so the worst overshoot is one creep step (0.5 mm) rather than one normal step
        # (`fine_max_pos_step`, 3 mm -- twice W-B's +-1.5 mm window).
        # PRESS only. Gating on depth alone was wrong: `InsertionController` has to pass a narrow
        # SETTLE -> PRESS window (|tip_z - settle_z| < settle_z_tol, i.e. +-4 mm about 35 mm above
        # the mouth), and a creep command issued while it is still in SETTLE overrides the very
        # motion that carries it through that window. Measured, W-A never left SETTLE
        # (`ins_phase` 1 and `best_depth` at its -1000 sentinel in 12 of 12), so its depth was
        # whatever the pick's lift happened to leave -- 47 mm, unmoved by a 3.6 mm swing in the
        # commanded depth. W-B and W-C escaped it only because their larger grip stations put the
        # tip through the window before the creep band was reached.
        from isaaclab_arena.controllers.insertion_controller import PRESS

        creeping = (
            self._in_insertion
            & ~self._handover
            & (self._ins.phase == PRESS)
            & depth.gt(target_depth - self._creep_band)
        )
        if bool(creeping.any()):
            from isaaclab_arena.controllers.ee_control import ee_pose_action

            creep_tcp = tcp.clone()
            # FEED-FORWARD the measured tracking gap. The PD settles ~1.1 mm short of an absolute
            # z command (measured twice: as a uniform +1.1 mm overshoot when the trigger fired
            # late, and as a 48.8-49.0 mm stall against a 50.0 mm target when it fired never --
            # same constant, opposite signs, one cause). Commanding 1.1 mm past the target makes
            # the equilibrium land ON it; the run-to-run spread around that constant was 0.2 mm.
            creep_tcp[:, 2] = face_z + station - target_depth - _CREEP_TRACKING_GAP
            creep_action = ee_pose_action(
                env, creep_tcp, level, torch.zeros_like(creeping), self._pick.ee_cfg, self._creep_step
            )
            action = self._take_action(action, creep_action, creeping, OWNER_CREEP)
        if os.environ.get("BDASH_PRESS_TRACE") and bool(self._in_insertion[0]):
            # The stall between SETTLE and the seat is invisible in the per-episode record: depth
            # freezes mid-bore with zero force, PRESS never exits, nothing terminates. Per-step,
            # printed, never read back.
            print(
                f"[press] depth={float(depth[0]) * 1e3:6.1f}mm tgt={float(target_depth[0]) * 1e3:5.1f} "
                f"ins_ph={int(self._ins.phase[0])} creep={bool(creeping[0])} hand={bool(self._handover[0])} "
                f"tcp_z={float(tcp[0, 2]) * 1e3:6.1f} fz={float(fz[0]):.2f} arrived={bool(arrived[0])}",
                flush=True,
            )
        was_handing = self._handover.clone()
        self._handover = self._handover | arrived
        # Hold the pose the insertion finished at. `_ins.step` is no longer driving these envs, so
        # without this the blend below would fall back to the PICK action, which still points at
        # its transport waypoint above the tray.
        just_started = self._handover & ~was_handing
        if self._hold_tcp is None or self._hold_tcp.shape[0] != uenv.num_envs:
            self._hold_tcp = tcp.clone()
        self._hold_tcp = torch.where(just_started.unsqueeze(-1), tcp, self._hold_tcp)

        # SETTLE. The arm is frozen; hold it there for `settle_steps` before anything moves, so the
        # part stops swinging and squares up in the grip. Releasing or clamping into a moving part
        # sets whatever tilt it happened to have: measured without this, seat angle ran 0.12-1.5 deg
        # against a QC bar of 0.5 deg, and the 0.12 deg episodes prove the pose is achievable.
        self._handover_timer = torch.where(
            self._handover, self._handover_timer + 1, torch.zeros_like(self._handover_timer)
        )
        # SETTLE COMPLETE = stopped AND square, with the step count only as a backstop. This is
        # where low speed belongs: the arm is already frozen, so waiting for it to be still is a
        # test that can actually pass.
        from isaaclab.utils.math import quat_error_magnitude

        square = quat_error_magnitude(level, tcp_quat) < _SETTLE_ROT_TOL
        settled_pose = self._handover & ((speed.lt(_HANDOVER_SPEED) & square) | (self._handover_timer >= _SETTLE_MAX))

        # RECOMPUTE every step from the frozen ABSOLUTE pose. `ee_pose_action` returns a RELATIVE
        # displacement, so holding the arm still is not "freeze the action" -- freezing a non-zero
        # delta re-applies it every step and the arm keeps going. Measured: freezing across 25
        # settle + 20 close steps drove W-B 43 mm past target (-43 mm protrusion error), and the
        # earlier 20-step version drove it 27 mm. That, not the RELEASE trigger, was the whole
        # systematic protrusion error. Recomputing against a fixed target converges to zero motion,
        # which is what "hold still" has to mean with a relative action space.
        if bool(self._handover.any()):
            from isaaclab_arena.controllers.ee_control import ee_pose_action

            hold_action = ee_pose_action(
                env, self._hold_tcp, level, torch.zeros_like(self._handover), self._pick.ee_cfg
            )
            action = self._take_action(action, hold_action, self._handover, OWNER_HOLD)

        # HOLD-DESCEND for every variant: the part is carried to its seat and the chuck takes it
        # there. Nothing is released before the clamp exists, so nothing falls through the bore.
        # Force-seated variants are RELEASED to fall the last stretch onto the stopper seat: their
        # seat is below the held-depth limit (fingers on the ring), so holding on cannot finish the
        # job by construction. Depth-seated variants stay held -- a released plain cylinder in a
        # through bore just falls out.
        needs_drop = self._handover & self._release_to_seat[target][rows].bool()
        # ONE `command` PER STEP. There used to be two -- a held-parts call here and the combined
        # call below -- and `command` moves the radius one rate-step toward whichever target its
        # mask says. For a held part both calls said "close" and the stroke merely ran at double
        # rate; for a DROP-seated part the first said "open" and the second said "close", so the
        # jaws stepped one out, one in, every step, forever. Measured: W-C sat perfectly seated
        # (61.5 mm, +0.5 mm, 0.00 deg) for a thousand steps while the jaws flapped in place and
        # `chuck_closed` never came. The bug was unreachable while `needs_drop` was hardwired to
        # zeros; enabling the drop path exposed it.
        # ...once a dropped part has had time to settle, the jaws close on it too. `close_steps`
        # is the stroke length, so reusing it as the settle budget keeps one number in one place.
        # A dropped part needs time to fall and come to rest before the jaws take it. Counted from
        # the settle finishing, not from the handover, so it does not race the settle.
        #
        # TODO(§4-3 エア着座確認): the real cell confirms seating with an air check, and the spec
        # mirrors it in sim as a SEAT-FACE GAP test, explicitly cleared for runtime use. Close the
        # jaws on that instead of on a clock. The timer below is provisional.
        #
        # 1 s, not `close_steps`. W-C falls ~21 mm, which is 65 ms of free fall but far longer to
        # stop bouncing; at `close_steps` (20 steps = 0.33 s) the clamp was catching it mid-motion
        # and freezing a pose 3.7 mm past the flange seat -- past a face the Ø45 flange cannot enter,
        # so it was provably still moving.
        self._drop_timer = torch.where(needs_drop, self._drop_timer + 1, torch.zeros_like(self._drop_timer))
        settled = needs_drop & (self._drop_timer > _DROP_SETTLE_STEPS)
        close_mask = (self._handover & settled_pose & ~needs_drop) | settled
        self._jaws.command(env, close_mask=close_mask)
        # Published for the load_failed termination: once the COMMIT is commanded, contact force on
        # the part is the cycle doing its job, not a jam. The commit is the clamp for a held part
        # and the RELEASE for a drop-seated one -- the drop lands on its seat with an impact spike
        # that is the seating itself, and it happens before any jaw moves (measured: W-C aborted on
        # load_failed at the moment of touchdown, five steps before the clamp would have fired).
        # Monotone within the episode.
        commit = close_mask | needs_drop
        prev_cmd = getattr(uenv, "bdash_jaws_commanded", None)
        if prev_cmd is None or prev_cmd.shape[0] != commit.shape[0]:
            prev_cmd = torch.zeros_like(commit)
        uenv.bdash_jaws_commanded = prev_cmd | commit

        # The FIXED JOINT half of `close + fixed joint`: once the stroke is home the chuck owns the
        # part. Without it the jaws are geometry with no grip force and the gripper opening drops
        # the part through the bore.
        self._jaws.clamp(env, self._names, uenv.bdash_target_idx, self._handover & self._jaws.closed)
        uenv.bdash_jaws_closed = self._jaws.closed
        # Open the gripper: immediately for a part that must drop, and once the jaws hold it for the
        # rest. Column 6 of the action is the binary gripper (+1 open, -1 closed).
        # Open the gripper once the jaws are NEARLY home rather than fully home. A 3-jaw chuck
        # centres the part as it closes, but it cannot while the gripper still holds it -- the two
        # fight and the part sets at whatever tilt the gripper had. Measured holding to full close:
        # seat angle 0.12-1.5 deg against a QC bar of 0.5 deg. Releasing a hair early leaves the
        # jaws close enough to catch it (they are inside the Ø25 shaft's radius + the gap below)
        # while giving it the freedom to square up.
        # Open the gripper only once the chuck has actually taken the part (`closed`, which is when
        # `clamp` above engages) -- not merely when the jaws are near. They hold nothing until the
        # joint exists, so letting go early drops the part through a through-bore. The exception is
        # a flange-seated variant, which has to be released BEFORE the jaws close so it can fall the
        # last stretch onto the face.
        release = needs_drop | (self._handover & self._jaws.closed)
        return self._take_gripper(action, 1.0, release, GRIP_RELEASE)

    # ------------------------------------------------------------------ §0-11
    def _upright_yaw(self, uenv, target, rows, grasp_point) -> torch.Tensor:
        """Wrist yaw for an upright target: closing direction perpendicular to the tallest neighbour.

        Torch mirror of ``bdash_chuck_randomization.upright_yaw_rule``. Both must agree, because the
        layout sampler uses the rule to decide whether a candidate is reachable (§0-10) while the
        teacher uses it to actually grasp — a divergence would make the guarantee vacuous.
        ``tests/test_bdash_hand_clearance.py`` pins the rule; the agreement is asserted there too.

        With ``c(psi) = (sin psi, -cos psi)``, ``c . d = 0`` gives ``psi = atan2(d_y, d_x)``, folded
        to ``[-pi/2, pi/2]`` — the same closing LINE, picked deterministically so the rule stays a
        single-valued function of the scene (spec §3 learnability).
        """
        pos_all = torch.stack([uenv.scene[n].data.root_pos_w for n in self._names], dim=1)  # (N,K,3)
        quat_all = torch.stack([uenv.scene[n].data.root_quat_w for n in self._names], dim=1)
        local_z = torch.zeros_like(pos_all)
        local_z[..., 2] = 1.0
        axis_all = quat_apply(quat_all, local_z)

        side_all = getattr(uenv, "bdash_side_lying", None)
        if side_all is None:
            side_all = axis_all[..., 2].abs() <= self._upright_cos
        lengths, radii = self._lengths.unsqueeze(0), self._radii.unsqueeze(0)
        top_side = pos_all[..., 2] + torch.clamp(lengths * axis_all[..., 2], min=0.0) + radii
        top_up = pos_all[..., 2] + lengths * axis_all[..., 2].abs()
        top_z = torch.where(side_all, top_side, top_up)  # (N,K)

        hand_bottom = (grasp_point[:, 2] + self._hand_clear_z).unsqueeze(1)
        interferes = top_z > hand_bottom
        interferes[rows, target] = False  # the target is not its own obstacle

        # tallest interfering neighbour; -inf keeps non-interfering ones from ever winning argmax
        masked = torch.where(interferes, top_z, torch.full_like(top_z, float("-inf")))
        nb = masked.argmax(dim=1)
        d = pos_all[rows, nb, :2] - grasp_point[:, :2]
        psi = torch.atan2(d[:, 1], d[:, 0])
        psi = torch.remainder(psi + math.pi / 2.0, math.pi) - math.pi / 2.0  # fold to [-pi/2, pi/2]
        return torch.where(interferes.any(dim=1) & (d.norm(dim=-1) > 1e-9), psi, torch.zeros_like(psi))

    # ------------------------------------------------------------------ act
    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        self._ensure(env)
        uenv = env.unwrapped
        target = uenv.bdash_target_idx  # (N,) latched at reset
        rows = torch.arange(uenv.num_envs, device=target.device)

        pos = torch.stack([uenv.scene[n].data.root_pos_w for n in self._names], dim=1)[rows, target]
        quat = torch.stack([uenv.scene[n].data.root_quat_w for n in self._names], dim=1)[rows, target]

        local_z = torch.zeros_like(pos)
        local_z[:, 2] = 1.0
        axis = quat_apply(quat, local_z)  # leading end -> trailing end, unit

        upright = axis[:, 2].abs() > self._upright_cos
        station = self._stations[target][rows, upright.long()]
        grasp_point = pos + station.unsqueeze(-1) * axis
        # Lift the commanded TCP for a side-lying part so the fingertips (8.9 mm BELOW the TCP)
        # clear the tray floor -- see controllers.yaml pick.grip_lift_side.
        grasp_point[:, 2] = grasp_point[:, 2] + torch.where(
            upright, torch.zeros_like(grasp_point[:, 2]), torch.full_like(grasp_point[:, 2], self._lift_side)
        )
        # S2 ERROR INJECTION (E1 test-bed, default off): BDASH_GRASP_OFF_AXIAL_MM shifts the grasp
        # ALONG the part's axis, which is the error touch-off exists to measure -- an axial grip
        # offset passes 1:1 into protrusion if uncorrected. Upright targets only (E1 runs upright),
        # positive = gripped that many mm closer to the leading end. The harness varies it per
        # episode to realise the injected error distribution.
        off_ax = float(os.environ.get("BDASH_GRASP_OFF_AXIAL_MM", "0") or "0") * 1e-3
        if off_ax:
            # FIRST grasp only. The injected error stands in for the VLA's grasp; the re-grasp after
            # a refix (or a re-erect) is the CLASSICAL precise grip, and injecting into it makes the
            # refix unfixable by construction -- measured: P3 routed an 8 mm error to the pad,
            # re-grasped, and came back with the same 8 mm, because this knob re-applied it.
            fixed_since = torch.zeros_like(upright)
            if self._refix_done is not None and self._refix_done.shape[0] == upright.shape[0]:
                fixed_since = fixed_since | self._refix_done
            if self._reerect_done is not None and self._reerect_done.shape[0] == upright.shape[0]:
                fixed_since = fixed_since | self._reerect_done
            grasp_point[:, 2] = grasp_point[:, 2] - torch.where(
                upright & ~fixed_since, torch.full_like(grasp_point[:, 2], off_ax), torch.zeros_like(grasp_point[:, 2])
            )

        # Side-lying: turn the wrist so the fingers close ACROSS the axis — forced by the geometry.
        # Upright: the part is yaw-symmetric, so the yaw is FREE and spec §0-11 spends it on
        # clearance, pointing the hand's long axis away from the tallest neighbour. Chosen once
        # before the grasp and frozen by the CLOSE latch below; the residual tilt of an upright part
        # is never fed through atan2, which would make the command depend on a wobble no camera can
        # resolve (the v3 failure mode).
        q_down = torch.zeros(uenv.num_envs, 4, device=pos.device, dtype=pos.dtype)
        q_down[:, 1] = 1.0
        yaw = self._upright_yaw(uenv, target, rows, grasp_point)
        if self.force_yaw_zero:
            # ABLATION SWITCH, measurement only -- never set on a recording run. §0-11's yaw is
            # chosen precisely when a tall neighbour is present, so "psi != 0" is also a marker for
            # a crowded scene: the observed 99.1% (psi=0) vs 80.7% (psi!=0) split cannot say which
            # of the two is doing the work. Holding the yaw at zero while the layouts and the target
            # draw stay bit-identical (same seed, clearance enforcement off) separates them.
            yaw = torch.zeros_like(yaw)
        z_axis = torch.zeros_like(pos)
        z_axis[:, 2] = 1.0
        q_upright = quat_mul(quat_from_angle_axis(yaw, z_axis), q_down)
        grasp_quat = torch.where(upright.unsqueeze(-1), q_upright, grasp_quat_from_axis(axis))

        # LATCH the target once the fingers close. Up to that point tracking the part's live pose is
        # what lets the teacher follow a part that shifts during the approach. After it, the part IS
        # the gripper -- so continuing to servo the wrist to the part's measured axis closes a
        # feedback loop onto the arm's own output, and `grasp_quat_from_axis` has a branch cut at
        # a_x = 0 where the commanded yaw jumps by pi. A held part drifting across that cut makes the
        # wrist whip, which threw parts clear of the tray (measured: grasped, then found at
        # y = +0.69 and y = -0.38 against a tray spanning y in [0.075, 0.265]). Upright parts were
        # immune because they are commanded a constant quat -- which is exactly the observed
        # upright-vs-side-lying split (93% vs 67%, Fisher p = 0.0056).
        held = self._pick.phase is not None and bool((self._pick.phase >= CLOSE).any())
        if self._latched_point is None or self._latched_point.shape[0] != uenv.num_envs:
            self._latched_point = grasp_point.clone()
            self._latched_quat = grasp_quat.clone()
            self._latched_axis = axis.clone()
        if self._latched_yaw is None or self._latched_yaw.shape[0] != uenv.num_envs:
            self._latched_yaw = yaw.clone()
        if self._latch_stale is not None and bool(self._latch_stale.any()):
            # NB: local name must not shadow `rows` -- get_action passes it onward to the blenders.
            stale_rows = self._latch_stale.unsqueeze(-1)
            self._latched_point = torch.where(stale_rows, grasp_point, self._latched_point)
            self._latched_quat = torch.where(stale_rows, grasp_quat, self._latched_quat)
            self._latched_axis = torch.where(stale_rows, axis, self._latched_axis)
            self._latched_yaw = torch.where(self._latch_stale, yaw, self._latched_yaw)
            self._latch_stale[:] = False
        if held:
            hold = (self._pick.phase >= CLOSE).unsqueeze(-1)
            grasp_point = torch.where(hold, self._latched_point, grasp_point)
            grasp_quat = torch.where(hold, self._latched_quat, grasp_quat)
            self._latched_point = torch.where(hold, self._latched_point, grasp_point)
            self._latched_quat = torch.where(hold, self._latched_quat, grasp_quat)
            # Latched for the §4-2 turn, which needs to know WHICH END of the part is the leading
            # one and may not ask the part once it is held: the held part's axis is a function of
            # the arm's own output, so reading it to decide the turn is exactly the feedback the
            # ruling forbids. Frozen at CLOSE with everything else the turn is planned from.
            self._latched_axis = torch.where(hold, self._latched_axis, axis)
            # Latched for the SAME reason the quat is, but it also fixes a reporting trap: once the
            # part is lifted, `_upright_yaw` re-evaluates with the grasp point high above the tray,
            # finds no neighbour tall enough to matter and returns 0. Reading the live value at the
            # end of an episode therefore reports psi = 0 for every success no matter what was
            # commanded, which looks exactly like "large yaw causes failure".
            self._latched_yaw = torch.where(self._pick.phase >= CLOSE, self._latched_yaw, yaw)
        else:
            self._latched_point = grasp_point.clone()
            self._latched_quat = grasp_quat.clone()
            self._latched_axis = axis.clone()
            self._latched_yaw = yaw.clone()

        # REPORTING ONLY: the part's tilt the moment the re-grip closes on it. With the standing
        # tilt (recorded when the leg finished) and the final seated angle, this splits the error
        # three ways -- landed crooked, tilted by the re-grip, or cocked inside the bore -- and the
        # three have nothing in common as fixes. W-B seats at 3.6-4.3 deg against an upright-path
        # median of 0.3 deg, so something in here owns ~4 deg and guessing which is not worth a run.
        if self._reerect_done is not None and self._reerect_regrip_cos is None:
            closing = self._reerect_done & (self._pick.phase is not None) & (self._pick.phase >= CLOSE)
            if bool(closing.any()):
                self._reerect_regrip_cos = axis[:, 2].abs()

        self._pick.grasp_override = grasp_point
        self._pick.grasp_quat_override = grasp_quat
        self.last_side_lying = ~upright
        self.last_axis = axis
        self.last_grasp = grasp_point
        # The §0-11 angle actually commanded. Logged because §0-10's guarantee is computed by the
        # SAMPLER with its own copy of the rule: the folds are asserted to agree, but the inputs
        # (live neighbour poses at grasp time vs the analytic layout at reset) are not, so a
        # disagreement here would mean the certified target is approached at an uncertified yaw.
        self.last_yaw = self._latched_yaw

        action, pick_finished = self._pick.step(env)
        self._begin_action(action)

        if self._ins is not None:
            action = self._blend_insertion(env, uenv, rows, target, station, action, pick_finished)
        # spec §9: publish on the env so the recorder can log it alongside the action it explains.
        # Set on BOTH stages -- on the pick stage it is a constant "pick", which is exactly the
        # claim being recorded (nothing else wrote the action).
        uenv.bdash_action_owner = self._action_owner
        uenv.bdash_grip_owner = self._grip_owner

        if self._debug:
            # After the step, so the phase is this step's -- and so it is never the lazily-sized None.
            from isaaclab_arena.controllers.ee_control import read_ee_pose

            tcp, _ = read_ee_pose(env)
            robot = uenv.scene[self._pick.names["robot"]]
            jp = robot.data.joint_pos[0]
            lim = robot.data.soft_joint_pos_limits[0]
            # distance to the nearer limit, per joint, in rad -- a stalled DESCEND that is an IK
            # problem shows a joint pinned near 0; a collision stall shows joints with room to spare.
            margin = torch.minimum(jp - lim[:, 0], lim[:, 1] - jp)
            print(
                f"[joints] margins={' '.join(f'{m:+.3f}' for m in margin[:7])} "
                f"min={margin[:7].min():.4f}@j{int(margin[:7].argmin()) + 1}",
                flush=True,
            )
            print(
                f"[chuckteacher] target={target[0].item()} upright={bool(upright[0])} "
                f"axis=({axis[0, 0]:+.3f},{axis[0, 1]:+.3f},{axis[0, 2]:+.3f}) "
                f"station={station[0]:.3f} grasp=({grasp_point[0, 0]:.3f},{grasp_point[0, 1]:.3f},"
                f"{grasp_point[0, 2]:.4f}) tcp=({tcp[0, 0]:.3f},{tcp[0, 1]:.3f},{tcp[0, 2]:.3f}) "
                f"phase={int(self._pick.phase[0])}",
                flush=True,
            )
        return action

    # ----------------------------------------------------------------- state
    @property
    def phase(self) -> torch.Tensor | None:
        """Per-env FSM phase, for the de-risk harness (APPROACH..DONE)."""
        return None if self._pick is None else self._pick.phase

    def stall_stats(self) -> dict:
        """Watchdog state for the JSONL sidecar (spec §4-5): retries, stall depth, phase."""
        if self._pick is None or self._pick.phase is None:
            return {}
        stall_z = float(self._pick.stall_tcp_z[0])
        return {
            "final_phase": int(self._pick.phase[0]),
            "retries": int(self._pick.attempts[0]),
            "stall_steps": int(self._pick.stall_timer[0]),
            "gave_up": bool(self._pick.gave_up[0]),
            "empty_close": bool(self._pick.empty_close[0]),
            # None rather than NaN: JSON has no NaN, and json.dumps would emit a bare `NaN` token
            # that strict parsers reject.
            "stall_tcp_z": None if stall_z != stall_z else round(stall_z, 4),
            "stall_phase": int(self._pick.stall_phase[0]),
            "yaw": None if self.last_yaw is None else round(float(self.last_yaw[0]), 4),
            **self._insertion_stats(),
        }

    def _insertion_stats(self) -> dict:
        """Load-stage diagnostics. Empty on the pick stage, so the pick JSONL schema is unchanged."""
        if self._ins is None or self._ins.phase is None:
            return {}
        return {
            "ins_phase": int(self._ins.phase[0]),
            "ins_attempts": int(self._ins.attempts[0]),
            "ins_gave_up": bool(self._ins.gave_up[0]),
            "ins_best_depth": round(float(self._ins.best_depth[0]), 4),
            "ins_fz": None if self._ins.last_wrench is None else round(float(self._ins.last_wrench[0, 2]), 3),
            "ins_fxy": None if self._ins.last_wrench is None else round(float(self._ins.last_wrench[0, :2].norm()), 3),
            "in_insertion": bool(self._in_insertion[0]) if self._in_insertion is not None else False,
            "depth_nominal": None if self.last_depth_nominal is None else round(float(self.last_depth_nominal[0]), 4),
            "depth_reachable": (
                None if self.last_depth_reachable is None else round(float(self.last_depth_reachable[0]), 4)
            ),
        }

    def finished(self) -> torch.Tensor | None:
        from isaaclab_arena.controllers.scripted_pick import DONE

        if self._pick is None or self._pick.phase is None:
            return None
        pick_done = self._pick.phase >= DONE
        if self._ins is None or self._ins.phase is None:
            return pick_done
        # Load stage: the pick being done only means the part is over the bore. Both must finish.
        from isaaclab_arena.controllers.insertion_controller import DONE as INS_DONE

        return pick_done & (self._ins.phase >= INS_DONE)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._pick is None:
            return
        self._pick.reset(env_ids)
        self._pick.grasp_override = None
        self._pick.grasp_quat_override = None
        # Latches: FULL reset drops them wholesale (rebuilt on the next step). A PARTIAL reset must
        # NOT -- the old wholesale drop re-initialized every env's latch from its live pose, and an
        # env holding a part mid-transport had its grasp reference silently replaced. Measured in
        # the first vectorized recording probe: 41 resets over 10 target demos knocked the teacher
        # from its ~95% single-env grasp rate down to 24% -- every neighbour's episode end was a
        # small sabotage. Partial resets mark rows stale instead; get_action re-seeds exactly those
        # rows from the fresh episode's live values and leaves running envs alone.
        if env_ids is None:
            self._latched_point = None
            self._latched_quat = None
            self._latched_axis = None
            self._latched_yaw = None
            self._latch_stale = None
        else:
            if self._latch_stale is None and self._latched_point is not None:
                self._latch_stale = torch.zeros(
                    self._latched_point.shape[0], dtype=torch.bool, device=self._latched_point.device
                )
            if self._latch_stale is not None:
                self._latch_stale[env_ids] = True
        if self._touch is not None:
            # The probe's memory is one episode long BY MEANING -- its measurement describes the
            # grip it touched, and a grip dies with its episode. Left uncleared, episode N+1 saw
            # touch_done=True, skipped its own probe entirely, and gated on episode N's number:
            # measured, an 8 mm error sailed through as GO wearing the previous part's +2.86.
            self._touch.reset(env_ids)
            if self._touch_done is not None:
                self._touch_done[slice(None) if env_ids is None else env_ids] = False
        if self._reerect is not None:
            self._reerect.reset(env_ids)
            idx = slice(None) if env_ids is None else env_ids
            if self._reerect_done is not None:
                self._reerect_done[idx] = False
            if self._reerect_planned is not None:
                self._reerect_planned[idx] = False
            if self._refix_done is not None:
                self._refix_done[idx] = False
                self._refix_planned[idx] = False
            if self._gate_decided is not None:
                self._gate_decided[idx] = False
                self._gate_decision[idx] = GO
            if self._gate_e is not None:
                # Cleared, or a probe that gives up NEXT episode reports THIS episode's number: the
                # stats read whatever survives here, and a stale ê wearing a fresh episode's label
                # cost a debugging hour against a sensor that was in fact working.
                self._gate_e[idx] = 0.0
            self._reerect_stand_cos = None
            self._reerect_place_xy = None
            self._reerect_regrip_cos = None
        if self._ins is not None:
            self._ins.reset(env_ids)
            idx = slice(None) if env_ids is None else env_ids
            if self._in_insertion is not None:
                self._in_insertion[idx] = False
            if self._handover is not None:
                self._handover[idx] = False
            if self._handover_timer is not None:
                self._handover_timer[idx] = 0
            if self._drop_timer is not None:
                self._drop_timer[idx] = 0
            self._hold_tcp = None
            self._prev_tcp = None
            self._prev_depth = None
            # The jaws MUST be reopened here. They are scene state, not policy state: left closed,
            # the next episode starts with `chuck_closed` already true and terminates on step 0
            # without the arm having touched anything. Measured when this was missing: episodes
            # "succeeded" at lifted_step 0 with grasped_step None.
            if self._jaws is not None and self._env is not None:
                self._jaws.reset(env_ids)
                self._jaws.command(self._env, close_mask=torch.zeros_like(self._in_insertion))
                self._env.unwrapped.bdash_jaws_closed = self._jaws.closed
                self._env.unwrapped.bdash_jaws_commanded = torch.zeros_like(self._in_insertion)

    # ------------------------------------------------------------------- CLI
    # PolicyBase declares these abstract, and policy_runner.py calls add_args_to_parser BEFORE the
    # simulation app is up -- so omitting them fails the run at startup rather than at first use.
    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> BDashChuckTeacherPolicy:
        return BDashChuckTeacherPolicy(BDashChuckTeacherArgs.from_cli_args(args))
