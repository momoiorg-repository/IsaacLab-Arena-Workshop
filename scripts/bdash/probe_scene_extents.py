# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""World-space bounding boxes for everything the layout solver has to place things ON or AROUND.

The layout is being re-solved against reach and camera-occlusion requirements, and two of the
inputs are not in any config: the TABLE's actual footprint (a Nucleus prop, so its size is a
property of the USD, not of ``table_pose``) and where the robot base actually sits. Placing a tray
off the table edge is not a subtle failure -- the tray is a rigid body, so it simply falls -- and
guessing the size of a stock asset is exactly the kind of assumption that has cost this project
runs before.

    unset DISPLAY
    /isaac-sim/python.sh scripts/bdash/probe_scene_extents.py --num_envs 1 --seed 0 \\
        bdash_chuck_load --task none --variants all
"""

print("[extents] launching Isaac Sim (~40s)...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json

import omni.usd


def _aabb(path):
    """World AABB of a prim and its children, as ((min...), (max...)) in metres, or None."""
    box = omni.usd.get_context().compute_path_world_bounding_box(path)
    if box is None:
        return None
    lo, hi = box
    lo, hi = tuple(float(v) for v in lo), tuple(float(v) for v in hi)
    # An empty/invalid box comes back inverted; report it as absent rather than as a huge extent.
    return None if any(h < ll for ll, h in zip(lo, hi)) else (lo, hi)


def main() -> None:
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()

    stage = omni.usd.get_context().get_stage()
    root = "/World/envs/env_0"
    # The background is a stock prop, and its overall AABB is not enough to place a camera by: a lab
    # table carries structure ABOVE its work surface (measured, this one reaches z +0.715), which
    # both occludes a look-down view and is not where the tray can rest. So the background is
    # descended into, while the task's own assets are simple enough to take whole.
    prims = []
    for prim in stage.GetPrimAtPath(root).GetChildren():
        prims.append(prim)
        if not prim.GetName().startswith(("bdash_", "Robot")):
            prims.extend(prim.GetChildren())
            for child in prim.GetChildren():
                prims.extend(child.GetChildren())
    out = {}
    for prim in prims:
        path = str(prim.GetPath())
        box = _aabb(path)
        if box is None:
            continue
        lo, hi = box
        out[path[len(root) + 1 :]] = {
            "min": [round(v, 4) for v in lo],
            "max": [round(v, 4) for v in hi],
            "size": [round(h - ll, 4) for ll, h in zip(lo, hi)],
            "centre": [round((h + ll) / 2.0, 4) for ll, h in zip(lo, hi)],
        }

    origin = env.scene.env_origins[0].tolist()
    print(f"[extents] env_origin = {[round(v, 4) for v in origin]}", flush=True)
    print(f"{'prim':44} {'centre xy':>18} {'size xyz':>26} {'top z':>8}", flush=True)
    for name, box in sorted(out.items()):
        print(
            f"{name:44} ({box['centre'][0]:7.4f},{box['centre'][1]:8.4f}) "
            f"({box['size'][0]:7.4f},{box['size'][1]:7.4f},{box['size'][2]:7.4f}) {box['max'][2]:8.4f}",
            flush=True,
        )
    print("EXTENTS_JSON " + json.dumps({"env_origin": origin, "prims": out}), flush=True)
    print("PROBE_EXTENTS_DONE", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
