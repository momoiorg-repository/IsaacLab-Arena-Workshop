# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import argparse

from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli


def main():
    parser = argparse.ArgumentParser()
    add_example_environments_cli_args(parser)
    # mock cli args
    sys.argv = ["dummy", "table_pick_and_place", "--embodiment", "crx5ia", "--object", "dex_cube"]
    args_cli = parser.parse_args()

    arena_env_builder = get_arena_builder_from_cli(args_cli)
    name, cfg = arena_env_builder.build_registered()

    print("=== SCENE ===")
    print([k for k in dir(cfg.scene) if not k.startswith("_")])
    if hasattr(cfg.scene, "red_container"):
        print("red_container pos:", cfg.scene.red_container.init_state.pos)

    print("=== EVENTS ===")
    print([k for k in dir(cfg.events) if not k.startswith("_")])
    if hasattr(cfg.events, "red_container"):
        print("red_container event func:", cfg.events.red_container.func.__name__)
        print("red_container event params:", cfg.events.red_container.params)


if __name__ == "__main__":
    main()
