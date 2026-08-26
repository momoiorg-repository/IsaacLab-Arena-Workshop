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
import copy
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
        from isaaclab_arena_environments.mdp.bdash_chuck_config import (
            load_assets_cfg,
            load_materials_cfg,
            load_task_cfg,
        )

        task_cfg = load_task_cfg()
        scene_cfg = task_cfg["scene"]
        chuck_geom = bdash_chuck_assets.chuck_geometry()

        variants = self._resolve_variants(args_cli.variants)

        # --- assets ------------------------------------------------------------------------------
        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        # Lighting is FROZEN with the materials (materials.yaml): a material only reaches the image
        # through the light, so recording under one and inferring under another undoes the material
        # choice entirely. Read from config rather than restated here for that reason.
        dome_cfg = load_materials_cfg()["lighting"]["dome"]
        light_spawner_cfg = sim_utils.DomeLightCfg(
            color=tuple(float(c) for c in dome_cfg["color"]), intensity=float(dome_cfg["intensity"])
        )
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        chuck_body = self.asset_registry.get_asset_by_name("bdash_chuck_body")()
        # instance_name is mandatory for repeated assets: Scene.add_asset keys on asset.name and
        # overwrites duplicates, so three jaws sharing one name would collapse into a single prim.
        jaws = [
            self.asset_registry.get_asset_by_name("bdash_chuck_jaw")(instance_name=f"bdash_chuck_jaw_{k}")
            for k in range(_JAW_COUNT)
        ]
        workbench = self.asset_registry.get_asset_by_name("bdash_workbench")()
        tray = self.asset_registry.get_asset_by_name("bdash_tray")()
        reerect_pad = self.asset_registry.get_asset_by_name("bdash_reerect_pad")()

        embodiment_kwargs = dict(
            enable_cameras=args_cli.enable_cameras,
            initial_joint_pose=scene_cfg["initial_joint_pose"],
        )
        # Wrist camera pose, chuck task only (configs/bdash/chuck_load/task.yaml `cameras.wrist_offset`,
        # already converted to the ROS convention OffsetCfg expects). The embodiment default stays
        # untouched -- the peg task's frozen v6 recordings were made with it.
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(**embodiment_kwargs)
        # Wrist camera pose MUST be written onto the BUILT camera cfg, not passed as camera_offset:
        # FrankaCameraCfg assembles its cameras in __post_init__, which has already run by the time
        # the constructor stores _camera_offset -- measured 2026-08-26: the passed offset was
        # silently ignored and the sim still showed the default (0.11, -0.031, -0.074). The focal
        # override below always used the direct-mutation path, which is why it worked.
        wrist_off = (task_cfg.get("cameras") or {}).get("wrist_offset")
        if wrist_off and args_cli.enable_cameras:
            cam = embodiment.camera_config.wrist_cam
            cam.offset.pos = tuple(float(v) for v in wrist_off["pos"])
            cam.offset.rot = tuple(float(v) for v in wrist_off["rot_wxyz_ros"])
            print(f"[bdash] wrist cam offset -> pos={cam.offset.pos} rot(ros)={cam.offset.rot}", flush=True)
        wf = os.environ.get("BDASH_WRIST_FOCAL") or (task_cfg.get("cameras") or {}).get("wrist_focal_length")
        if wf and args_cli.enable_cameras:
            embodiment.camera_config.wrist_cam.spawn.focal_length = float(wf)
            print(f"[bdash] wrist cam focal -> {float(wf)}", flush=True)
        embodiment.scene_config.robot = mdp.FRANKA_PANDA_BDASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        teleop_device = (
            self.device_registry.get_device_by_name(args_cli.teleop_device)()
            if args_cli.teleop_device is not None
            else None
        )

        # --- fixture poses -----------------------------------------------------------------------
        # The generated WORKBENCH is the work surface; its top face is z = 0 and everything else in
        # the configs is measured from there. The stock table stays only as the visual base under it,
        # dropped by the slab thickness so the two do not fight over the same z -- the small table's
        # 0.91 m width in y is exactly what this replaces.
        bench_cfg = load_assets_cfg()["workbench"]
        workbench.set_initial_pose(Pose(position_xyz=(*bench_cfg["centre"], 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        table_pose = list(scene_cfg["table_pose"])
        table_pose[2] -= float(bench_cfg["size"][2])
        background.set_initial_pose(Pose(position_xyz=tuple(table_pose), rotation_wxyz=(0.707, 0.0, 0.0, 0.707)))
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, scene_cfg["ground_plane_z"])))

        chuck_xy = scene_cfg["chuck_pose"]
        chuck_body.set_initial_pose(Pose(position_xyz=tuple(chuck_xy), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        self._place_jaws(jaws, chuck_xy, chuck_geom, open_extra=args_cli.jaw_open)

        from isaaclab_arena_environments.mdp import bdash_chuck_randomization as _bcr

        tray.set_initial_pose(
            Pose(
                position_xyz=tuple(scene_cfg["tray_pose"]),
                rotation_wxyz=_bcr.yaw_quat(math.radians(float(scene_cfg.get("tray_yaw_deg", 0.0)))),
            )
        )
        reerect_pad.set_initial_pose(
            Pose(position_xyz=tuple(scene_cfg["reerect_pose"]), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )

        # --- workpieces in the tray --------------------------------------------------------------
        workpiece_variants = [v for v in variants for _ in range(int(task_cfg["tray_spawn"]["count_per_variant"]))]
        workpieces = self._spawn_workpieces(variants, task_cfg, scene_cfg)

        assets = [
            background,
            ground_plane,
            light,
            workbench,
            chuck_body,
            *jaws,
            tray,
            reerect_pad,
            *workpieces,
        ]
        scene = Scene(assets=assets)

        # `none` builds the scene with nothing attached, which is how the camera framing is checked
        # before any demo is recorded (camera settings bake into the dataset and cannot be redone).
        if args_cli.task == "none":
            task = NoTask()
        elif args_cli.task in ("pick", "full"):
            from isaaclab_arena.tasks.bdash_chuck_load_task import BDashChuckLoadTask
            from isaaclab_arena_environments.mdp import bdash_peg_randomization

            profile = bdash_peg_randomization.profile(args_cli.level)
            if args_cli.task == "full":
                # A side-lying part physically cannot enter a vertical bore -- it has to be re-erected
                # on the V-block first (spec §4-2), and that skill is not implemented. Without this
                # the load stage would draw side-lying targets it can never seat, and the success
                # rate would be a mixture of "can the teacher load" and "did it draw an upright one".
                # REMOVE when the §4-2 re-erect is measured good enough to draw side-lying
                # targets at the load stage. The leg exists now (place onto the 仮置き台) but the
                # load rate behind it is not yet worth drawing them by default.
                task_cfg = copy.deepcopy(task_cfg)
                # BDASH_ALLOW_SIDE_LYING=1 lifts the restriction, so the cost of NOT having the
                # V-block re-erect can be measured instead of argued about.
                if os.environ.get("BDASH_SIDE_LYING_ONLY"):
                    task_cfg["target"]["mode"] = "side_lying_only"
                elif not os.environ.get("BDASH_ALLOW_SIDE_LYING"):
                    task_cfg["target"]["mode"] = "upright_only"
            task = BDashChuckLoadTask(
                workpieces=workpieces,
                variants=workpiece_variants,
                tray=tray,
                chuck_body=chuck_body,
                background_scene=background,
                task_cfg=task_cfg,
                assets_cfg=load_assets_cfg(),
                stage="pick" if args_cli.task == "pick" else "full",
                lighting_event=profile.lighting_event,
                texture_event=profile.texture_event,
                seed=int(getattr(args_cli, "seed", 0) or 0),
                run_tag=f"{args_cli.run_tag}_{args_cli.level}",
                log_dir=args_cli.log_dir,
                episode_length_s=float(task_cfg["termination"]["episode_length_s"]),
            )
        else:
            raise NotImplementedError(
                f"task {args_cli.task!r} is not implemented; available: none (scene only), pick "
                "(grasp + lift, the VLA's slice). 'full' chuck loading lands after GATE 1."
            )

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

        These are only the *initial* poses baked into the scene; with ``--task pick`` the per-reset
        scatter event re-draws them every episode. They are sampled from that same distribution
        (with a fixed seed) rather than laid out by hand, so a scene built for camera framing shows
        exactly what a recorded episode will show -- including that the stepped variants come to
        rest tilted rather than flat.
        """
        import random

        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.mdp import bdash_chuck_assets
        from isaaclab_arena_environments.mdp import bdash_chuck_randomization as bcr
        from isaaclab_arena_environments.mdp.bdash_chuck_config import load_assets_cfg, load_controllers_cfg

        per_variant = int(task_cfg["tray_spawn"]["count_per_variant"])
        frame = bcr.tray_frame(task_cfg)
        ordered = [v for v in variants for _ in range(per_variant)]
        station = float(load_controllers_cfg().get("pick", {}).get("grip_station_side", 0.030))

        placements, _ = bcr.sample_tray_layout(
            random.Random(0),
            pieces=[bdash_chuck_assets.workpiece_profile(v) for v in ordered],
            **bcr.layout_kwargs_from_cfg(task_cfg, load_assets_cfg(), station),
        )

        out = []
        for index, (variant, place) in enumerate(zip(ordered, placements)):
            asset_name = bdash_chuck_assets.BDASH_WORKPIECE_BY_VARIANT[variant]
            # unique instance_name per copy — see the jaw comment above
            obj = self.asset_registry.get_asset_by_name(asset_name)(
                instance_name=f"{asset_name}_{index % per_variant}",
            )
            x, y, z = place["pos"]
            wx, wy = bcr.tray_to_world(x, y, frame)
            obj.set_initial_pose(
                Pose(
                    position_xyz=(wx, wy, z),
                    rotation_wxyz=bcr.quat_mul_wxyz(bcr.yaw_quat(frame[2]), place["quat"]),
                )
            )
            out.append(obj)
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
        parser.add_argument(
            "--task",
            type=str,
            default="none",
            choices=["none", "pick", "full"],
            help=(
                "'none' builds the scene only (camera framing); 'pick' is the VLA slice (grasp + "
                "lift) and is what every recorded demo uses; 'full' adds the chuck load and ends "
                "on `seated` (spec §4-3)"
            ),
        )
        parser.add_argument("--variants", type=str, default="all", help="workpiece variants, e.g. 'W-A,W-C' or 'all'")
        parser.add_argument(
            "--level",
            type=str,
            default="L1",
            choices=["L0", "L1", "L2", "L3"],
            help="scene-diversity level; L2+ add lighting/texture randomization (tray scatter is always on)",
        )
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

    cameras = load_task_cfg()["cameras"]
    focal = float(os.environ.get("BDASH_SIDE_FOCAL") or cameras["side_focal_length"])
    for cam_name in ("left_cam", "right_cam"):
        cam = getattr(env_cfg.scene, cam_name, None)
        if cam is not None and getattr(cam, "spawn", None) is not None:
            cam.spawn.focal_length = focal

    # Aim the external cams from the solved layout. The offsets are relative to the Robot prim,
    # whose root is the fixed base at the origin, so a world pose is also the offset.
    # `top_cam` / `overview_cam` are the task-level names; `right_cam` / `left_cam` are the
    # embodiment's field names and cannot be renamed here without changing the observation keys of
    # the frozen peg pipeline, which shares FrankaCameraCfg.
    for cam_name, key in (("right_cam", "top_cam"), ("left_cam", "overview_cam")):
        cam = getattr(env_cfg.scene, cam_name, None)
        if cam is None or key not in cameras:
            continue
        eye = tuple(float(v) for v in cameras[key]["eye"])
        if cameras[key].get("rot"):
            # An explicit, hand-framed orientation. Used VERBATIM -- re-deriving it from a look-at
            # point would quietly discard the roll the framing was chosen with. (w, x, y, z) in the
            # USD/OpenGL convention: the camera looks along its own -Z.
            rot = tuple(float(v) for v in cameras[key]["rot"])
        else:
            # Derived from a look-at point. `up` fixes the ROLL, and it only matters when the view
            # is near-vertical: there the world-Z reference degenerates and the image orientation
            # becomes arbitrary. Default it to the TRAY'S axis so an overhead frame shows the tray
            # square rather than skewed.
            tray_yaw = math.radians(float(load_task_cfg()["scene"].get("tray_yaw_deg", 0.0)))
            up = cameras[key].get("up") or [math.cos(tray_yaw), math.sin(tray_yaw), 0.0]
            rot = _look_at_quat(eye, cameras[key]["look_at"], tuple(float(v) for v in up))
        cam.offset = type(cam.offset)(pos=eye, rot=rot, convention="opengl")
    return env_cfg


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Orientation (w, x, y, z) of an OPENGL-convention camera at ``eye`` looking at ``target``.

    OpenGL cameras look along their own -Z with +Y up, so the camera's +Z is the backward
    direction. Derived rather than written down because the layout is solved, not chosen: a
    hand-written quaternion would silently stop pointing at the tray the moment the tray moved.
    """
    bx, by, bz = (e - t for e, t in zip(eye, target))  # camera +Z = eye - target
    norm = math.sqrt(bx * bx + by * by + bz * bz) or 1.0
    zx, zy, zz = bx / norm, by / norm, bz / norm
    # camera +X = up x +Z
    xx, xy, xz = up[1] * zz - up[2] * zy, up[2] * zx - up[0] * zz, up[0] * zy - up[1] * zx
    norm = math.sqrt(xx * xx + xy * xy + xz * xz)
    if norm < 1e-6:
        # STRAIGHT DOWN: the view direction is parallel to `up`, so `up x +Z` vanishes and the frame
        # is undefined. Left unhandled this produced a zero X axis and a non-unit quaternion that
        # Isaac Lab normalised to identity -- the camera still happened to look down, but its ROLL
        # was silently whatever identity gives rather than anything chosen. Fall back to world +X as
        # the reference so a top-down camera has a defined, repeatable image orientation.
        xx, xy, xz = 1.0 * zz - 0.0, 0.0 - 0.0 * zz, 0.0 * zy - 1.0 * zx  # (1,0,0) x +Z
        norm = math.sqrt(xx * xx + xy * xy + xz * xz) or 1.0
    norm = norm or 1.0
    xx, xy, xz = xx / norm, xy / norm, xz / norm
    yx, yy, yz = zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx  # +Y = +Z x +X
    trace = xx + yy + zz
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (yz - zy) / s, (zx - xz) / s, (xy - yx) / s)
    if xx > yy and xx > zz:
        s = math.sqrt(1.0 + xx - yy - zz) * 2.0
        return ((yz - zy) / s, 0.25 * s, (yx + xy) / s, (zx + xz) / s)
    if yy > zz:
        s = math.sqrt(1.0 + yy - xx - zz) * 2.0
        return ((zx - xz) / s, (yx + xy) / s, 0.25 * s, (zy + yz) / s)
    s = math.sqrt(1.0 + zz - xx - yy) * 2.0
    return ((xy - yx) / s, (zx + xz) / s, (zy + yz) / s, 0.25 * s)
