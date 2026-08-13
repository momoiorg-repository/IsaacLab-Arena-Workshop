# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH multi-variant chuck-loading environment.

Composes the Franka embodiment, a vertical 3-jaw power chuck (body plus three kinematic jaws), a
coarse tray of mixed workpieces, a touch-off datum block, a V-block, a background table and a dome
light. Physics uses the same FORGE-grade callback as the B-DASH assembly task
(:func:`~isaaclab_arena_environments.mdp.env_callbacks.bdash_assembly_env_cfg_callback`), which is
tuned for contact-rich insertion and applies unchanged here.

Scene layout and spawn ranges come from ``configs/bdash/chuck_load/task.yaml``; asset geometry comes from
``configs/bdash/chuck_load/assets.yaml`` via :mod:`isaaclab_arena_environments.mdp.bdash_chuck_assets`. Nothing is
restated in this file, so the scene cannot drift from the meshes that were generated.

Workpieces spawn at arbitrary pose, including lying on their side. A side-lying cylinder cannot
enter a vertical chuck, so those parts must be re-erected through the V-block -- which is what
makes the three-way gate a mechanism requirement rather than a demo flourish.

``--task none`` builds the scene with no task attached, which is how the camera framing is checked
before any demo is recorded (camera settings bake into the dataset and cannot be changed after).
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

_JAW_COUNT = 3


class BDashChuckLoadEnvironment(ExampleEnvironmentBase):

    name: str = "bdash_chuck_load"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp
        from isaaclab_arena_environments.mdp import bdash_chuck_assets
        from isaaclab_arena_environments.mdp.bdash_chuck_config import load_task_cfg

        task_cfg = load_task_cfg()
        scene_cfg = task_cfg["scene"]
        chuck_geom = bdash_chuck_assets.chuck_geometry()

        variants = self._resolve_variants(args_cli.variants)

        # --- assets ------------------------------------------------------------------------------
        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        chuck_body = self.asset_registry.get_asset_by_name("bdash_chuck_body")()
        # instance_name is mandatory for repeated assets: Scene.add_asset keys on asset.name and
        # overwrites duplicates, so three jaws sharing one name would collapse into a single prim.
        jaws = [
            self.asset_registry.get_asset_by_name("bdash_chuck_jaw")(instance_name=f"bdash_chuck_jaw_{k}")
            for k in range(_JAW_COUNT)
        ]
        tray = self.asset_registry.get_asset_by_name("bdash_tray")()
        vblock = self.asset_registry.get_asset_by_name("bdash_vblock")()
        touchoff = self.asset_registry.get_asset_by_name("bdash_touchoff_block")()

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            initial_joint_pose=scene_cfg["initial_joint_pose"],
        )
        embodiment.scene_config.robot = mdp.FRANKA_PANDA_BDASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        teleop_device = (
            self.device_registry.get_device_by_name(args_cli.teleop_device)()
            if args_cli.teleop_device is not None
            else None
        )

        # --- fixture poses -----------------------------------------------------------------------
        background.set_initial_pose(
            Pose(position_xyz=tuple(scene_cfg["table_pose"]), rotation_wxyz=(0.707, 0.0, 0.0, 0.707))
        )
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, scene_cfg["ground_plane_z"])))

        chuck_xy = scene_cfg["chuck_pose"]
        chuck_body.set_initial_pose(Pose(position_xyz=tuple(chuck_xy), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        self._place_jaws(jaws, chuck_xy, chuck_geom, open_extra=args_cli.jaw_open)

        tray.set_initial_pose(Pose(position_xyz=tuple(scene_cfg["tray_pose"]), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        vblock.set_initial_pose(Pose(position_xyz=tuple(scene_cfg["vblock_pose"]), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        touchoff.set_initial_pose(
            Pose(position_xyz=tuple(scene_cfg["touchoff_pose"]), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )

        # --- workpieces in the tray --------------------------------------------------------------
        workpieces = self._spawn_workpieces(variants, task_cfg, scene_cfg)

        assets = [background, ground_plane, light, chuck_body, *jaws, tray, vblock, touchoff, *workpieces]
        scene = Scene(assets=assets)

        # The full task (predicates, touch-off logging, per-variant terminations) lands separately;
        # `none` exists so the scene can be built and camera-framed before it is written.
        if args_cli.task != "none":
            raise NotImplementedError(
                f"task {args_cli.task!r} is not implemented yet; only --task none is available so far"
            )
        task = NoTask()

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=bdash_env_cfg_callback,
        )

    # ------------------------------------------------------------------------------------------
    def _place_jaws(self, jaws: list, chuck_xy, geom: dict, open_extra: float) -> None:
        """Seat the three jaws in their radial slots at 120 deg.

        The jaw mesh has its gripping face at local x=0 and extends toward +x, so placing it at
        radius r puts the face exactly r from the chuck axis: r = jaw_radius_low is fully closed on
        the Ø25 shaft, and ``open_extra`` retracts it outward to load/unload.
        """
        from isaaclab_arena.utils.pose import Pose

        cx, cy, cz = chuck_xy
        # jaws sit in slots cut down from the chuck face
        jaw_z = cz + geom["body_height"] - geom["jaw_height"]
        radius = geom["jaw_radius_low"] + open_extra
        for k, jaw in enumerate(jaws):
            angle = k * 2.0 * math.pi / _JAW_COUNT
            jaw.set_initial_pose(
                Pose(
                    position_xyz=(cx + radius * math.cos(angle), cy + radius * math.sin(angle), jaw_z),
                    rotation_wxyz=(math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)),
                )
            )

    def _spawn_workpieces(self, variants: list[str], task_cfg: dict, scene_cfg: dict) -> list:
        """Place ``count_per_variant`` of each requested variant in the tray.

        Poses here are only the initial layout; the per-reset randomization event re-scatters them.
        A fraction are laid on their side, which is the case the chuck cannot accept directly.
        """
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.mdp import bdash_chuck_assets

        spawn = task_cfg["tray_spawn"]
        tray_x, tray_y, tray_z = scene_cfg["tray_pose"]
        drop = spawn["drop_height"]
        per_variant = int(spawn["count_per_variant"])

        out = []
        index = 0
        total = len(variants) * per_variant
        for variant in variants:
            asset_name = bdash_chuck_assets.BDASH_WORKPIECE_BY_VARIANT[variant]
            for copy_idx in range(per_variant):
                # unique instance_name per copy — see the jaw comment above
                obj = self.asset_registry.get_asset_by_name(asset_name)(
                    instance_name=f"{asset_name}_{copy_idx}",
                )
                # deterministic initial spread; the reset event randomizes from here
                frac = (index + 0.5) / total
                x = (
                    tray_x
                    + spawn["pose_range"]["x"][0]
                    + frac * (spawn["pose_range"]["x"][1] - spawn["pose_range"]["x"][0])
                )
                y = tray_y + (spawn["pose_range"]["y"][1] if index % 2 == 0 else spawn["pose_range"]["y"][0])
                side_lying = (index / max(total - 1, 1)) < float(spawn["side_lying_frac"])
                if side_lying:
                    # rotate 90 deg about +Y so the axis lies in the tray plane
                    rot = (math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0), 0.0)
                    z = tray_z + 0.0125 + 0.002  # resting on the shaft radius
                else:
                    rot = (1.0, 0.0, 0.0, 0.0)
                    z = tray_z + drop
                obj.set_initial_pose(Pose(position_xyz=(x, y, z), rotation_wxyz=rot))
                out.append(obj)
                index += 1
        return out

    @staticmethod
    def _resolve_variants(spec: str) -> list[str]:
        from isaaclab_arena_environments.mdp import bdash_chuck_assets

        if spec in ("all", "", None):
            return list(bdash_chuck_assets.BDASH_VARIANTS)
        wanted = [v.strip().upper() for v in spec.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in bdash_chuck_assets.BDASH_WORKPIECE_BY_VARIANT]
        if unknown:
            raise ValueError(
                f"unknown workpiece variant(s) {unknown}; known: {list(bdash_chuck_assets.BDASH_VARIANTS)}"
            )
        return wanted

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--task", type=str, default="none", help="task to attach ('none' builds scene only)")
        parser.add_argument("--variants", type=str, default="all", help="workpiece variants, e.g. 'W-A,W-C' or 'all'")
        parser.add_argument("--background", type=str, default="table")
        parser.add_argument("--embodiment", type=str, default="franka")
        parser.add_argument("--log_dir", type=str, default="logs/bdash")
        parser.add_argument("--run_tag", type=str, default="bdash")
        parser.add_argument(
            "--jaw_open",
            type=float,
            default=0.010,
            help="extra radial retraction of the jaws beyond the closed Ø25 position (m)",
        )
        parser.add_argument("--teleop_device", type=str, default=None)


def bdash_env_cfg_callback(env_cfg):
    """FORGE-grade assembly physics, plus this task's side-camera framing.

    The chuck-loading scene is much wider than the bdash peg workspace, so the shared
    ``FrankaCameraCfg`` focal length of 8.0 clips it. Overriding here rather than in
    ``isaaclab_arena/embodiments/franka/franka.py`` is deliberate: that default is what the
    B-DASH v6 model was trained at and its published results are frozen against it.

    ``BDASH_SIDE_FOCAL`` overrides the configured value for framing experiments.
    """
    from isaaclab_arena_environments import mdp
    from isaaclab_arena_environments.mdp.bdash_chuck_config import load_task_cfg

    env_cfg = mdp.bdash_assembly_env_cfg_callback(env_cfg)

    focal = float(os.environ.get("BDASH_SIDE_FOCAL") or load_task_cfg()["cameras"]["side_focal_length"])
    for cam_name in ("left_cam", "right_cam"):
        cam = getattr(env_cfg.scene, cam_name, None)
        if cam is not None and getattr(cam, "spawn", None) is not None:
            cam.spawn.focal_length = focal
    return env_cfg
