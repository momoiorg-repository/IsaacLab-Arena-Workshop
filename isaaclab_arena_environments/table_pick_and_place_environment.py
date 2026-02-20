# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class TablePickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "table_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference

        background = self.asset_registry.get_asset_by_name("galileo")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination_container = self.asset_registry.get_asset_by_name("red_container")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        
        # Goal object
        destination_location = ObjectReference(
            name="destination_location",
            prim_path="{ENV_REGEX_NS}/red_container",
            parent_asset=destination_container,
            object_type=ObjectType.RIGID,
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set poses
        # Placing object on the white table in Galileo room
        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.55, 0.0, 0.30),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        
        # Placing container on the white table in Galileo room
        destination_container.set_initial_pose(
            Pose(
                position_xyz=(0.68, 0.31, 0.23),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        scene = Scene(assets=[background, pick_up_object, destination_container, destination_location])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object, 
                destination_location, 
                background, 
                destination_object=destination_container
            ),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="dex_cube")
        parser.add_argument("--embodiment", type=str, default="franka")
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default="keyboard")
