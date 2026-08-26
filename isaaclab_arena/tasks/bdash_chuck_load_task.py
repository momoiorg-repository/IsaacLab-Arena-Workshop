# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH multi-variant chuck-loading task.

A Franka picks one of several cylindrical workpieces out of a coarse tray -- at arbitrary pose,
including lying on its side -- and (after GATE 1) loads it into a vertical 3-jaw chuck. Success and
failure are decided by the predicates in
:mod:`isaaclab_arena_environments.mdp.bdash_chuck_predicates`, in the same style as
:class:`~isaaclab_arena.tasks.bdash_pick_insert_task.BDashPickInsertTask`: predicates are never
registered anywhere, they are passed as ``func=`` into ``TerminationTermCfg`` with every threshold
injected as a ``params`` entry at build time.

Two things differ structurally from the peg task:

**Many candidate workpieces.** There are K of them (6 with ``--variants all``), so the episode
latches a target at reset onto ``env.bdash_target_idx`` and every predicate is scoped to it. A
filtered ``ContactSensor`` is one-body-to-many-filters, so a single regex sensor spanning all the
workpieces *and* filtering against the fingers is not expressible -- hence one sensor per workpiece,
and hence a scene config built with :func:`make_configclass`, since K is not known until the
environment has parsed ``--variants``.

**Two stages.** ``stage="pick"`` is the VLA's slice: the episode ends when the target workpiece is
lifted clear of the tray, which is where the recorded demos are cut. ``stage="full"`` adds the
chuck-loading terminations and lands after GATE 1.
"""

from __future__ import annotations

import copy
import numpy as np
import os
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

import isaaclab_arena_environments.mdp as mdp
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena_environments.mdp import bdash_chuck_assets
from isaaclab_arena_environments.mdp import bdash_chuck_materials as bcm
from isaaclab_arena_environments.mdp import bdash_chuck_predicates as cp
from isaaclab_arena_environments.mdp import bdash_chuck_randomization as bcr
from isaaclab_arena_environments.mdp.bdash_chuck_config import load_controllers_cfg, load_materials_cfg

# default Franka finger prim paths (robot is spawned at {ENV_REGEX_NS}/Robot)
_DEFAULT_FINGER_PRIMS = [
    "{ENV_REGEX_NS}/Robot/panda_leftfinger",
    "{ENV_REGEX_NS}/Robot/panda_rightfinger",
]

STAGES = ("pick", "full")


class BDashChuckLoadTask(TaskBase):

    def __init__(
        self,
        workpieces: list[Asset],
        variants: list[str],
        tray: Asset,
        chuck_body: Asset,
        background_scene: Asset,
        task_cfg: dict,
        assets_cfg: dict,
        stage: str = "pick",
        finger_prim_paths: list[str] | None = None,
        lighting_event: EventTermCfg | None = None,
        texture_event: EventTermCfg | None = None,
        seed: int = 0,
        run_tag: str = "bdash",
        log_dir: str = "logs/bdash",
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

        self.workpieces = workpieces
        self.variants = list(variants)
        self.tray = tray
        self.chuck_body = chuck_body
        self.background_scene = background_scene
        self.stage = stage
        self.run_tag = run_tag
        self.log_dir = log_dir

        # load_task_cfg is lru_cached, so never mutate it in place: the per-build workpiece and
        # sensor names depend on --variants and would leak into the next environment built.
        self.cfg = copy.deepcopy(task_cfg)
        self.names = self.cfg["scene_names"]
        self.geom = self.cfg["geometry"]

        count = len(workpieces)
        finger_prefix = self.names["workpiece_finger_sensor_prefix"]
        self.workpiece_names = tuple(w.name for w in workpieces)
        self.finger_sensor_names = tuple(f"{finger_prefix}_{k}" for k in range(count))
        self.names["workpieces"] = list(self.workpiece_names)
        self.names["workpiece_finger_sensors"] = list(self.finger_sensor_names)

        # Per-variant geometry, ordered like `workpieces` so a predicate can index it with the
        # latched target. Sourced from the asset registry rather than restated, so it cannot drift
        # away from the generated meshes.
        self.profiles = [bdash_chuck_assets.workpiece_profile(v) for v in self.variants]
        # The layout sampler keeps the grasp station clear of the tray walls, so it needs the same
        # station the teacher will actually use. Read it from controllers.yaml rather than
        # restating it here, so the two cannot drift apart.
        self.grasp_station = float(load_controllers_cfg().get("pick", {}).get("grip_station_side", 0.030))

        # One ContactSensor per workpiece, filtered against the two fingers. K is only known now,
        # so the scene config has to be built dynamically.
        finger_prim_paths = finger_prim_paths or _DEFAULT_FINGER_PRIMS
        sensor_fields = [
            (name, ContactSensorCfg, wp.get_contact_sensor_cfg(contact_against_prim_paths=finger_prim_paths))
            for name, wp in zip(self.finger_sensor_names, workpieces)
        ]
        # The load stage needs a SECOND sensor per workpiece, unfiltered: `force_violation` reads
        # `net_forces_w` (everything the part touches -- bore wall, jaw, chuck face), while the
        # finger sensors above read `force_matrix_w` against the fingers only and can never see a
        # jam against the chuck. `scene_names.workpiece_contact_sensor_prefix` has been declared in
        # task.yaml since the first build with zero Python references; this is what it was for.
        #
        # Only for stage="full": a contact sensor per workpiece is not free, and the pick stage --
        # which is what every recorded demo and the whole of GATE 1 runs on -- has no use for it.
        self.contact_sensor_names: tuple[str, ...] = ()
        self.fixture_sensor_names: tuple[str, ...] = ()
        self.fixture_filter_names: tuple[str, ...] = ()
        if self.stage == "full":
            contact_prefix = self.names["workpiece_contact_sensor_prefix"]
            self.contact_sensor_names = tuple(f"{contact_prefix}_{k}" for k in range(count))
            self.names["workpiece_contact_sensors"] = list(self.contact_sensor_names)
            sensor_fields += [
                (name, ContactSensorCfg, wp.get_contact_sensor_cfg())
                for name, wp in zip(self.contact_sensor_names, workpieces)
            ]
            # A THIRD sensor per workpiece, filtered against the chuck body and the three jaws
            # separately. `force_matrix_w` gives one column per filter, so this names WHICH fixture
            # a stalled insertion is resting on. The unfiltered sensor above only gives a
            # magnitude, and a magnitude cannot distinguish "sitting on the chuck face" from
            # "sitting on a jaw" -- which is exactly the question a stalled load poses.
            # The 仮置き台 is a fixture too: the touch-off probe reads THIS sensor for the part
            # touching the pad's top face, and with the pad absent from the filter that contact was
            # invisible -- the probe pressed a held part onto a surface its sensor could not see.
            fixture_prims = (
                [self.chuck_body.prim_path]
                + [f"{{ENV_REGEX_NS}}/bdash_chuck_jaw_{k}" for k in range(3)]
                + ["{ENV_REGEX_NS}/bdash_reerect_pad"]
            )
            self.fixture_sensor_names = tuple(f"wp_fixture_{k}" for k in range(count))
            self.names["workpiece_fixture_sensors"] = list(self.fixture_sensor_names)
            self.fixture_filter_names = ("chuck_body", "jaw_0", "jaw_1", "jaw_2", "reerect_pad")
            sensor_fields += [
                (name, ContactSensorCfg, wp.get_contact_sensor_cfg(contact_against_prim_paths=fixture_prims))
                for name, wp in zip(self.fixture_sensor_names, workpieces)
            ]
        self.scene_config = make_configclass("BDashChuckSceneCfg", sensor_fields)()

        # The scatter event owns workpiece poses from here on, so the per-asset reset terms must
        # stand down or they will fight it (same reasoning as bdash_pick_insert_task).
        for wp in workpieces:
            wp.disable_reset_pose()

        self.assets_cfg = assets_cfg
        self.events_cfg = self._make_events_cfg(assets_cfg, seed, lighting_event, texture_event)
        self.termination_cfg = self._make_termination_cfg()
        default_description = (
            "Pick up a workpiece from the tray and load it into the chuck."
            if self.stage == "full"
            else "Pick up a workpiece from the tray and lift it clear."
        )
        self.task_description = task_description or default_description

    # ------------------------------------------------------------------ scene
    def get_scene_cfg(self):
        return self.scene_config

    # ------------------------------------------------------------- terminations
    def get_termination_cfg(self):
        return self.termination_cfg

    def _grasp_params(self) -> dict:
        grasped = self.cfg["grasped"]
        return {
            "robot_name": self.names["robot"],
            "finger_sensors": self.finger_sensor_names,
            "width_min": float(grasped["width_min"]),
            "width_max": float(grasped["width_max"]),
            "grasp_force": float(grasped["grasp_force"]),
        }

    def _load_params(self) -> dict:
        """Per-variant acceptance geometry for the load stage, derived, never restated.

        Two different indexing conventions live here and they are not interchangeable --
        ``test_bdash_chuck_predicates.py`` pins both. ``seat_angle_rad`` is applied COLUMN-WISE
        (one entry per workpiece, broadcast across the K axis inside :func:`cp.seated`), while
        ``protrusion_ok``'s nominals/tolerances are GATHERED by the latched target index.
        """
        tolerances = [bdash_chuck_assets.workpiece_tolerance(v) for v in self.variants]
        fallback_deg = float(self.cfg["seated"]["seat_angle_deg"])
        return {
            "seat_angle_rad": tuple(np.radians(float(t.get("seat_angle_deg", fallback_deg))) for t in tolerances),
            "datum_offsets": tuple(float(p["datum_offset"]) for p in self.profiles),
            # nominal protrusion = datum offset - commanded insert depth, so the two can never
            # drift apart (task.yaml:183-185 spells the arithmetic out per variant).
            # Nominal protrusion = datum offset - the depth actually commanded. Using
            # `chuck_load.insert_depth_m` here was wrong once a WORK STOPPER existed: that yaml
            # value is the flange-on-face figure, while the seat is now the stopper's top, so the
            # two disagreed by the stopper height and W-C read a 10.6 mm error while sitting
            # perfectly on its seat. `_load_targets` already derives the commanded depth from the
            # section profile and the stopper; take the nominal from the same place so the two
            # cannot drift.
            "nominals": tuple(
                float(p["datum_offset"]) - float(d)
                for p, d in zip(self.profiles, self._load_targets()["target_depths"])
            ),
            "protrusion_tol": tuple(float(t["protrusion_mm"]) / 1000.0 for t in tolerances),
            **self._load_targets(),
        }

    def _load_targets(self) -> dict:
        """Where each variant must end up, and how it knows it has arrived (spec §4-3, §12.3).

        Both are DERIVED from the section profile, not restated. A variant whose last section is
        wider than the bore cannot enter it, so that shoulder lands on the chuck face and the part
        seats by FORCE at whatever depth the geometry gives; anything narrower passes through and is
        driven to a commanded DEPTH instead. W-C is the flange case (Ø45 against a Ø40 bore, seating
        at 72 mm -- the same number `chuck_load.insert_depth_m` states, now derived rather than
        asserted); W-B's Ø32 shoulder passes the bore, so it is a depth target despite being stepped.
        """
        bore_radius = (
            float(self.cfg["chuck_load"]["bore_clearance"])
            + float(self.assets_cfg["workpieces"][self.variants[0]]["sections"][0][0]) / 2.0
        )
        pick_cfg = load_controllers_cfg().get("pick", {})
        below_top = pick_cfg.get("grip_below_top_upright", 0.035)
        stations, depths, force_seated = [], [], []
        for profile, variant in zip(self.profiles, self.variants):
            sections = self.assets_cfg["workpieces"][variant]["sections"]
            seats = float(sections[-1][0]) / 2.0 > bore_radius
            # The seat is the WORK STOPPER's top when one is fitted, not the chuck face, so the
            # depth a flange-seated variant reaches is short by the stopper's height.
            stopper = self.assets_cfg["chuck"].get("work_stopper") or {}
            derived = sum(float(length) for _, length in sections[:-1]) - float(stopper.get("height", 0.0))
            derived = derived if seats else None
            depths.append(derived if derived is not None else float(self.cfg["chuck_load"]["insert_depth_m"][variant]))
            force_seated.append(1.0 if seats else 0.0)
            bt = below_top[variant] if isinstance(below_top, dict) else below_top
            stations.append(float(profile["length"]) - float(bt))
        return {
            "stations": tuple(stations),
            "target_depths": tuple(depths),
            "force_seated": tuple(force_seated),
        }

    def _make_termination_cfg(self):
        if self.stage == "full":
            return self._make_load_termination_cfg()
        lifted = self.cfg["lifted"]
        success = TerminationTermCfg(
            func=cp.lifted,
            params={
                "workpiece_names": self.workpiece_names,
                "tray_name": self.tray.name,
                "rim_height": float(self.geom["tray_rim_height"]),
                "height": float(lifted["height"]),
                "speed_max": float(lifted["speed_max"]),
                "tip_offset": tuple(self.geom["tip_offset"]),
                **self._grasp_params(),
            },
        )
        # root_height_below_minimum takes a single asset_cfg and so cannot follow a per-episode
        # target; the chuck predicate reduces over the workpieces instead.
        object_dropped = TerminationTermCfg(
            func=cp.dropped,
            params={
                "workpiece_names": self.workpiece_names,
                "min_z": float(self.background_scene.object_min_z),
            },
        )
        return TerminationsCfg(success=success, object_dropped=object_dropped)

    def _make_load_termination_cfg(self):
        """spec §4-3: the episode ends when the target is SEATED in the chuck, not when it is lifted.

        The term must still be called ``success``: ``SuccessRecorder`` asserts that name is in
        ``termination_manager.active_terms`` and a run that renames it hangs at the first reset.

        The gripper is still holding at this point and that is deliberate. The jaws are kinematic
        and nothing drives them (assets.yaml:49-52, and the ``jaws:`` block in controllers.yaml is
        marked unconsumed), so there is nothing to hand the part over TO -- and the bore is a
        through hole, so a released part falls straight out. ``seated`` is a purely geometric
        acceptance test and needs no clamp to be true; closing the jaws and the air-seat check are
        the next block, not this one.
        """
        load = self._load_params()
        # spec §2.3: the commit is the jaw close, and QC judges after it. `loaded` remains the
        # controller's cue to hand the part over -- it is just no longer the end of the episode.
        success = TerminationTermCfg(func=cp.chuck_closed, params={})
        loaded_cue = TerminationTermCfg(
            func=cp.loaded,
            params={
                "chuck_name": self.chuck_body.name,
                "face_height": float(self.geom["chuck_face_height"]),
                "stations": load["stations"],
                "target_depths": load["target_depths"],
                "force_seated": load["force_seated"],
                "fixture_sensors": self.fixture_sensor_names,
                "seat_force": float(self.cfg["chuck_load"].get("seat_force", 8.0)),
                "robot_name": self.names["robot"],
                "min_grip_width": float(load_controllers_cfg().get("pick", {}).get("min_grip_width", 0.018)),
            },
        )
        # Jam: the net contact force on the part exceeds the limit. This needs the UNFILTERED
        # sensors added in __init__ for this stage -- the finger-filtered ones cannot see a part
        # binding against the bore wall, which is the failure this term exists to catch.
        load_failed = TerminationTermCfg(
            func=cp.load_failed,
            params={
                "contact_sensors": self.contact_sensor_names,
                "force_max": float(self.cfg["force"]["force_max"]),
            },
        )
        object_dropped = TerminationTermCfg(
            func=cp.dropped,
            params={
                "workpiece_names": self.workpiece_names,
                "min_z": float(self.background_scene.object_min_z),
            },
        )
        del loaded_cue  # built for its params to be validated at build time; not a termination
        return FullTerminationsCfg(success=success, load_failed=load_failed, object_dropped=object_dropped)

    # ----------------------------------------------------------------- events
    def _make_events_cfg(self, assets_cfg, seed, lighting_event, texture_event):
        tray_x, tray_y, tray_yaw = bcr.tray_frame(self.cfg)
        variant_ids = [bdash_chuck_assets.BDASH_VARIANTS.index(v) for v in self.variants]

        scatter = EventTermCfg(
            func=bcr.scatter_tray_workpieces,
            mode="reset",
            params={
                "asset_names": list(self.workpiece_names),
                "pieces": self.profiles,
                "tray_xy": (float(tray_x), float(tray_y)),
                "tray_yaw": float(tray_yaw),
                "seed": int(seed),
                "layout": bcr.layout_kwargs_from_cfg(self.cfg, assets_cfg, self.grasp_station),
            },
        )
        target_cfg = self.cfg["target"]
        pick_cfg = load_controllers_cfg().get("pick", {})
        select_target = EventTermCfg(
            func=bcr.select_target_workpiece,
            mode="reset",
            params={
                "num_workpieces": len(self.workpiece_names),
                "variant_ids": variant_ids,
                "mode": str(target_cfg["mode"]),
                "side_lying_prob": float(target_cfg.get("side_lying_prob", 0.5)),
                "seed": int(seed),
                # §0-10: restrict the target to a workpiece the OPEN HAND can descend onto. Passed
                # explicitly (rather than reached through env.cfg) so the event stays a pure function
                # of its params, and so `pieces` carries dimensions -- which become per-spawn once
                # §0-3 continuous randomisation lands.
                "pieces": self.profiles,
                "asset_names": list(self.workpiece_names),
                "hand": {
                    "half_span": float(target_cfg["hand"]["half_span"]),
                    "half_width": float(target_cfg["hand"]["half_width"]),
                    "clear_z": float(target_cfg["hand"]["clear_z"]),
                },
                "grasp": {
                    "station_side": float(pick_cfg.get("grip_station_side", 0.030)),
                    # dict{variant: m} or a bare float; the sampler resolves it per piece so its
                    # certified grasp station is the one the teacher will actually use.
                    "below_top_upright": pick_cfg.get("grip_below_top_upright", 0.035),
                },
                "enforce_hand_clearance": bool(target_cfg.get("enforce_hand_clearance", True)),
            },
        )
        # Arm-start jitter must run AFTER reset_all. The embodiment contributes its own
        # randomize_franka_joint_state, but embodiment event terms merge before task ones, so
        # reset_scene_to_default lands on top and puts the joints straight back to their defaults --
        # which is why record_vla_demos.py's --arm_init_std has never actually diversified anything.
        randomize_arm_start = EventTermCfg(
            func=franka_stack_events.randomize_joint_by_gaussian_offset,
            mode="reset",
            params={
                "mean": 0.0,
                "std": float(self.cfg["arm_start"]["jitter_std"]),
                "asset_cfg": SceneEntityCfg(self.names["robot"]),
            },
        )
        # spec: appearance randomisation, frozen ranges in materials.yaml. Disabled by setting
        # `workpiece_randomization.enabled: false` there, or per-run with BDASH_PLAIN_MATERIALS=1,
        # which is what the A/B comparison slice uses.
        materials_cfg = load_materials_cfg()["workpiece_randomization"]
        randomize_appearance = None
        if materials_cfg.get("enabled", True) and not os.environ.get("BDASH_PLAIN_MATERIALS"):
            randomize_appearance = EventTermCfg(
                func=bcm.randomize_workpiece_appearance,
                mode="reset",
                params={
                    "asset_names": list(self.workpiece_names),
                    "cfg": materials_cfg,
                    "seed": int(seed),
                    "held_out": bool(os.environ.get("BDASH_HELDOUT_MATERIALS")),
                },
            )

        return EventsCfg(
            scatter_workpieces=scatter,
            select_target=select_target,
            randomize_appearance=randomize_appearance,
            randomize_arm_start=randomize_arm_start,
            lighting_event=lighting_event,
            texture_event=texture_event,
        )

    def get_events_cfg(self):
        return self.events_cfg

    # ---------------------------------------------------------------- metrics
    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("B-DASH chuck loading does not use Mimic datagen in this scope.")

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.tray, offset=np.array([0.5, -0.6, 0.5]))


@configclass
class TerminationsCfg:
    """Pick-stage terminal conditions. ``success`` = the target workpiece is lifted clear."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class FullTerminationsCfg:
    """Load-stage terminal conditions. ``success`` = the target is seated in the chuck bore.

    Mirrors the peg task's shape (bdash_pick_insert_task.py:220-227): one success term, one
    mechanism-specific failure term, one dropped term. ``success`` keeps its name because
    ``SuccessRecorder`` asserts on it.
    """

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    load_failed: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class EventsCfg:
    """Reset events, in the order they must run.

    Field order IS execution order (the config fields are merged in declaration order and the
    EventManager preserves that), and the order matters twice over: ``reset_all`` has to come first
    or it would undo the scatter, and ``randomize_arm_start`` has to come after it or it would be
    undone in turn.
    """

    reset_all: EventTermCfg = MISSING
    scatter_workpieces: EventTermCfg = MISSING
    select_target: EventTermCfg = MISSING
    # Appearance is redrawn per episode. Visual only -- no collider, mass or friction is touched --
    # so it can sit anywhere after the parts exist without disturbing the physics ordering above.
    randomize_appearance: EventTermCfg = None
    randomize_arm_start: EventTermCfg = None
    lighting_event: EventTermCfg = None
    texture_event: EventTermCfg = None

    def __init__(
        self,
        scatter_workpieces,
        select_target,
        randomize_appearance=None,
        randomize_arm_start=None,
        lighting_event=None,
        texture_event=None,
    ):
        self.reset_all = EventTermCfg(
            func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
        )
        self.scatter_workpieces = scatter_workpieces
        self.select_target = select_target
        self.randomize_appearance = randomize_appearance
        self.randomize_arm_start = randomize_arm_start
        self.lighting_event = lighting_event
        self.texture_event = texture_event
