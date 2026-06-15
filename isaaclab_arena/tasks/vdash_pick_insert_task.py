# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""V-DASH pick→insert task (brief §3.4 / §3.5).

A Franka picks a Ø8 mm peg and inserts it into a kinematic socket of clearance ``c``. Unlike the
generic :class:`AssemblyTask` (success = root-proximity only), this task decides success and failure
from the §3.4 *predicates* (:mod:`isaaclab_arena_environments.mdp.vdash_predicates`) and records the
§3.5 per-episode handoff / terminal-error log.

Key differences from ``AssemblyTask`` (see plan §8 reconciliation):
  * ``fixed_asset`` = **socket** (kinematic, bolted to the table); ``held_asset`` = **peg** (the env
    that ships with the repo has these swapped);
  * two ``ContactSensor`` s are added on the peg — one filtered against the two gripper fingers
    (``peg_finger_contact`` → grasp force) and one for the net contact force (``peg_contact`` →
    force-limit violations);
  * terminations are ``success`` = ``inserted`` (depth > min_depth ∧ lateral < c), ``insertion_failed``
    = force-limit exceeded, ``object_dropped`` = peg fell, ``time_out``;
  * all thresholds come from ``configs/vdash/task.yaml`` and the active socket's clearance is injected
    into the ``inserted`` test at build time.
"""

from __future__ import annotations

import copy

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_arena_environments.mdp as mdp
from isaaclab_arena_environments.mdp import vdash_predicates as vp
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.metrics.vdash_handoff_logger import VDashHandoffLogger
from isaaclab_arena.tasks.events import randomize_poses_and_align_auxiliary_assets
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

# default Franka finger prim paths (robot is spawned at {ENV_REGEX_NS}/Robot)
_DEFAULT_FINGER_PRIMS = [
    "{ENV_REGEX_NS}/Robot/panda_leftfinger",
    "{ENV_REGEX_NS}/Robot/panda_rightfinger",
]


class VDashPickInsertTask(TaskBase):

    def __init__(
        self,
        socket: Asset,
        peg: Asset,
        background_scene: Asset,
        task_cfg: dict,
        clearance_m: float,
        finger_prim_paths: list[str] | None = None,
        pose_range: dict[str, tuple[float, float]] | None = None,
        min_separation: float = 0.0,
        lighting_event: EventTermCfg | None = None,
        texture_event: EventTermCfg | None = None,
        run_tag: str = "run",
        log_dir: str = "logs/vdash",
        episode_length_s: float | None = None,
        task_description: str | None = None,
        overhead_cam: bool = False,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.socket = socket          # fixed asset (kinematic)
        self.peg = peg                # held asset
        self.background_scene = background_scene
        # The §3.4 scene keys for peg/socket are the *registered asset names* (e.g. ``vdash_peg``,
        # ``vdash_socket_c1``), which vary per build — override the yaml placeholders with the real
        # names on a copy (``load_task_cfg`` is cached, so never mutate it in place).
        self.cfg = copy.deepcopy(task_cfg)
        self.cfg["scene_names"]["peg"] = peg.name
        self.cfg["scene_names"]["socket"] = socket.name
        self.clearance_m = clearance_m
        self.names = self.cfg["scene_names"]
        self.geom = self.cfg["geometry"]
        self.run_tag = run_tag
        self.log_dir = log_dir
        finger_prim_paths = finger_prim_paths or _DEFAULT_FINGER_PRIMS

        # Two contact sensors on the peg: one filtered vs the fingers (grasp force), one net force.
        self.scene_config = SceneCfg(
            peg_finger_contact=self.peg.get_contact_sensor_cfg(contact_against_prim_paths=finger_prim_paths),
            peg_contact=self.peg.get_contact_sensor_cfg(contact_against_prim_paths=[]),
        )
        if overhead_cam:  # M8: only add the (render-cost) camera for the classic-vision baseline
            self.scene_config.overhead_cam = _overhead_camera_cfg()

        # Reset restores canonical poses via ``reset_scene_to_default``. For L1+ we additionally
        # randomize socket/peg poses (and disable per-asset resets so they don't fight the
        # randomizer); at L0 (empty pose_range) poses stay fixed at their canonical values.
        pose_range = pose_range or {}
        if pose_range:
            for asset in (self.socket, self.peg):
                asset.disable_reset_pose()
        self.events_cfg = self._make_events_cfg(pose_range, min_separation, lighting_event, texture_event)
        self.termination_cfg = self._make_termination_cfg()
        self.task_description = task_description or (
            f"Pick up the peg and insert it into the socket (clearance {clearance_m * 1000:.2f} mm)"
        )

    # ------------------------------------------------------------------ scene
    def get_scene_cfg(self):
        return self.scene_config

    # ------------------------------------------------------------- terminations
    def get_termination_cfg(self):
        return self.termination_cfg

    def _make_termination_cfg(self):
        success = TerminationTermCfg(
            func=vp.inserted,
            params={
                "peg_name": self.names["peg"],
                "socket_name": self.names["socket"],
                "clearance": self.clearance_m,
                "tip_offset": tuple(self.geom["tip_offset"]),
                "mouth_height": float(self.geom["mouth_height"]),
                "min_depth": float(self.cfg["inserted"]["min_depth"]),
            },
        )
        insertion_failed = TerminationTermCfg(
            func=vp.insertion_failed,
            params={
                "peg_sensor": self.names["peg_sensor"],
                "force_max": float(self.cfg["force"]["force_max"]),
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.names["peg"]),
            },
        )
        return TerminationsCfg(
            success=success,
            insertion_failed=insertion_failed,
            object_dropped=object_dropped,
        )

    # ----------------------------------------------------------------- events
    def _make_events_cfg(self, pose_range, min_separation, lighting_event, texture_event):
        return EventsCfg(
            pose_range=pose_range,
            min_separation=min_separation,
            fixed_asset_cfg=SceneEntityCfg(self.names["socket"]),
            asset_cfgs=[SceneEntityCfg(self.names["socket"]), SceneEntityCfg(self.names["peg"])],
            lighting_event=lighting_event,
            texture_event=texture_event,
        )

    def get_events_cfg(self):
        return self.events_cfg

    # ---------------------------------------------------------------- metrics
    def get_metrics(self) -> list[MetricBase]:
        return [
            SuccessRateMetric(),
            VDashHandoffLogger(
                task_cfg=self.cfg,
                clearance=self.clearance_m,
                output_dir=self.log_dir,
                run_tag=self.run_tag,
            ),
        ]

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("V-DASH pick-insert does not use Mimic datagen in this scope.")

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.socket,
            offset=np.array([0.6, -0.6, 0.5]),
        )


@configclass
class SceneCfg:
    """Extra scene sensors contributed by the task (merged into the InteractiveScene)."""

    peg_finger_contact: ContactSensorCfg = MISSING
    peg_contact: ContactSensorCfg = MISSING
    overhead_cam: CameraCfg = None  # M8: fixed top-down RGB+depth cam for the classic-vision baseline


def _overhead_camera_cfg() -> CameraCfg:
    """Fixed top-down RGB+depth camera over the workspace (M8 classic-vision baseline, brief §3 M8).

    Looks straight down (identity quat, opengl convention) from 0.75 m above the workspace centre; the
    ~10 mm focal length frames the ~0.4 m workspace so the Ø20 mm grip cube is ~30 px. RGB feeds the
    colour segmentation; ``distance_to_image_plane`` lifts the masked pixels to 3D. Per-env prim."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/overhead_cam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, focus_distance=75.0, horizontal_aperture=5.376, vertical_aperture=5.376
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.45, 0.0, 0.75), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
    )


@configclass
class TerminationsCfg:
    """§3.4 terminal conditions. ``success``=inserted, ``insertion_failed``=force-limit exceeded."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    insertion_failed: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class EventsCfg:
    """Reset + at-reset pose randomization (L0/L1). Optional L2 lighting / L3 texture terms are
    supplied by :mod:`isaaclab_arena_environments.mdp.vdash_randomization`; left ``None`` they are
    skipped by the EventManager (distractors are added as scene assets by the environment)."""

    reset_all: EventTermCfg = MISSING
    randomize_asset_positions: EventTermCfg = None
    lighting_event: EventTermCfg = None
    texture_event: EventTermCfg = None

    def __init__(
        self, pose_range, fixed_asset_cfg, asset_cfgs, min_separation=0.0, lighting_event=None, texture_event=None
    ):
        self.reset_all = EventTermCfg(
            func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
        )
        # Pose randomization only when a range is given (L1+); at L0 reset_scene_to_default keeps the
        # canonical poses and this term stays None (skipped by the EventManager).
        self.randomize_asset_positions = None
        if pose_range:
            self.randomize_asset_positions = EventTermCfg(
                func=randomize_poses_and_align_auxiliary_assets,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "min_separation": min_separation,
                    "asset_cfgs": asset_cfgs,
                    "fixed_asset_cfg": fixed_asset_cfg,
                    "auxiliary_asset_cfgs": [],
                    "randomization_mode": "held_and_fixed_only",
                },
            )
        self.lighting_event = lighting_event
        self.texture_event = texture_event
