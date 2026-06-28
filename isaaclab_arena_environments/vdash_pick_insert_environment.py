# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""V-DASH pick→insert environment (brief §3).

Composes the Franka embodiment, a kinematic parametric socket of the requested clearance ``c``, the
Ø8 mm peg, a background table and a dome light into an :class:`IsaacLabArenaEnvironment` driven by
:class:`~isaaclab_arena.tasks.vdash_pick_insert_task.VDashPickInsertTask` (§3.4 predicates as
terminations + §3.5 per-episode handoff log) and the FORGE-grade physics callback
(:func:`~isaaclab_arena_environments.mdp.env_callbacks.vdash_assembly_env_cfg_callback`).

The clearance ``--clearance`` (mm) selects one of the registered ``vdash_socket_c{...}`` assets and is
injected into the ``inserted`` success test; ``--level`` selects the L0–L3 scene-diversity profile.
``--preinsert`` spawns the peg already seated in the bore — a fast integration check that the
``inserted → success`` termination fires and the logger writes a ``success`` record.
"""

from __future__ import annotations

import argparse
import os

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# canonical clearances we ship sockets for (mm)
_SUPPORTED_CLEARANCES_MM = (2.0, 1.0, 0.5, 0.25)
_DISTRACTOR_ASSETS = ("red_cube", "green_cube", "blue_block_basic_robolab")


class VDashPickInsertEnvironment(ExampleEnvironmentBase):

    name: str = "vdash_pick_insert"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.vdash_pick_insert_task import VDashPickInsertTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp
        from isaaclab_arena_environments.mdp import vdash_assets, vdash_randomization
        from isaaclab_arena_environments.mdp.vdash_config import load_task_cfg

        task_cfg = load_task_cfg()
        geom = task_cfg["geometry"]
        mouth_height = float(geom["mouth_height"])
        min_depth = float(task_cfg["inserted"]["min_depth"])

        # --- resolve clearance -> registered socket asset ----------------------------------------
        clearance_mm = _nearest_clearance(args_cli.clearance)
        socket_name = vdash_assets.VDASH_SOCKET_BY_CLEARANCE[clearance_mm]
        clearance_m = clearance_mm / 1000.0

        # --- assets ------------------------------------------------------------------------------
        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        socket = self.asset_registry.get_asset_by_name(socket_name)()
        peg = self.asset_registry.get_asset_by_name("vdash_peg")()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            # Start the arm hovering straight down over the workspace center (~16 cm above the grasp)
            # instead of the far/high default home -> shorter, more consistent pick demos for the VLA.
            # IK-solved for TCP (0.50, 0, 0.27) pointing -z -- the center of the L1 peg/socket spread
            # x in [0.40,0.60] (verified from data). Applied via the embodiment's init_franka_arm_pose
            # reset event (NOT init_state.joint_pos, which the event overrides).
            initial_joint_pose=[0.0, -0.0765, 0.0, -2.3076, 0.0, 2.2311, 0.785, 0.04, 0.04],
        )
        embodiment.scene_config.robot = mdp.FRANKA_PANDA_VDASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        teleop_device = (
            self.device_registry.get_device_by_name(args_cli.teleop_device)()
            if args_cli.teleop_device is not None
            else None
        )

        # --- poses (table top ~ z=0; socket mouth at z=mouth_height) ------------------------------
        background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))
        # floor under the table (table top ~ z=0, table ~1.05 m tall) — replaces the blank white room
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))
        socket.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        if args_cli.preinsert:
            # seat the peg tip on the bore axis (default: ~5 mm below the success depth threshold)
            peg_z = args_cli.preinsert_z if args_cli.preinsert_z is not None else (mouth_height - min_depth - 0.005)
            peg.set_initial_pose(Pose(position_xyz=(0.5, 0.0, peg_z), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        else:
            peg.set_initial_pose(Pose(position_xyz=(0.40, 0.0, 0.05), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        # --- scene-diversity profile (L0–L3) -----------------------------------------------------
        prof = vdash_randomization.profile(args_cli.level)
        assets = [background, socket, peg, light]
        # VDASH_NO_FLOOR: drop the floor (added 2026-06-19) to match the blank-white-room scene the
        # v3 VLA dataset was recorded in — isolates background domain-gap from model quality at eval.
        if not os.environ.get("VDASH_NO_FLOOR"):
            assets.append(ground_plane)
        assets += self._make_distractors(prof.num_distractors)

        scene = Scene(assets=assets)

        task = VDashPickInsertTask(
            socket=socket,
            peg=peg,
            background_scene=background,
            task_cfg=task_cfg,
            clearance_m=clearance_m,
            pose_range=prof.pose_range,
            min_separation=prof.min_separation,
            lighting_event=prof.lighting_event,
            texture_event=prof.texture_event,
            run_tag=f"{args_cli.run_tag}_c{clearance_mm}_{prof.level}",
            log_dir=args_cli.log_dir,
            # VDASH_EPISODE_S overrides the episode budget (default off) — used to give the tactile
            # CALIBRATE probe room for its touch sequence before the insert; headline path unchanged.
            episode_length_s=float(os.environ.get("VDASH_EPISODE_S") or task_cfg["termination"]["episode_length_s"]),
            overhead_cam=args_cli.enable_cameras,  # M8: top-down vision cam only when cameras are on
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.vdash_assembly_env_cfg_callback,
        )

    def _make_distractors(self, n: int) -> list:
        """Best-effort distractor objects for L3 (skipped if the asset isn't registered)."""
        from isaaclab_arena.utils.pose import Pose

        out = []
        for i in range(n):
            asset_name = _DISTRACTOR_ASSETS[i % len(_DISTRACTOR_ASSETS)]
            try:
                obj = self.asset_registry.get_asset_by_name(asset_name)()
            except Exception:
                continue
            # spread distractors around the workspace, clear of the socket
            x = 0.35 + 0.10 * (i % 3)
            y = 0.20 if i % 2 == 0 else -0.20
            obj.set_initial_pose(Pose(position_xyz=(x, y, 0.03), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
            out.append(obj)
        return out

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--clearance", type=float, default=1.0, help="socket radial clearance in mm")
        parser.add_argument("--level", type=str, default="L0", help="scene-diversity level L0-L3")
        parser.add_argument("--background", type=str, default="table")
        parser.add_argument("--embodiment", type=str, default="franka")
        parser.add_argument("--log_dir", type=str, default="logs/vdash")
        parser.add_argument("--run_tag", type=str, default="vdash")
        parser.add_argument(
            "--preinsert", action="store_true", help="spawn the peg pre-seated (inserted-path smoke test)"
        )
        parser.add_argument("--preinsert_z", type=float, default=None, help="override peg-tip z for --preinsert (m)")
        parser.add_argument("--teleop_device", type=str, default=None)


def _nearest_clearance(value_mm: float) -> float:
    """Snap a requested clearance (mm) to the nearest shipped socket clearance."""
    return min(_SUPPORTED_CLEARANCES_MM, key=lambda c: abs(c - value_mm))
