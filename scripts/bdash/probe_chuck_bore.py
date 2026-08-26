# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Where is the chuck's COOKED collision surface, as a function of radius from the bore axis?

The nominal geometry says a Ø40 bore runs the full 80 mm of the body, so a downward ray on the axis
should pass straight through. This asks PhysX instead of the mesh, because those have already been
shown to disagree once on this scene: the tray's ``convexDecomposition`` cook put the cavity floor
at 9.09 mm against a true 5.00 mm, which spawned every workpiece 3.9 mm inside the floor and cost
the whole of the GATE-1 upright shortfall (docs/progress/2026-08-14.md §12).

Motivation here: with ``--task full`` the W-C variant stops 5-13 mm ABOVE the chuck face with a
vertical reaction of 4.8-5.6 N and almost no lateral force, i.e. something flat is holding it up
where the nominal geometry says there is an open bore. W-A and W-B enter the same bore with ~0 N.

    /isaac-sim/python.sh scripts/bdash/probe_chuck_bore.py --num_envs 1 bdash_chuck_load \\
        --task none --variants all
"""

print("[borecheck] launching Isaac Sim (~40s)...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym


def main():
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()

    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom  # noqa: F401  (imported for the stage traversal below)

    stage = get_current_stage()
    chuck_prims = [p.GetPath().pathString for p in stage.Traverse() if "chuck_body" in p.GetPath().pathString.lower()]
    print(f"[borecheck] chuck prims: {chuck_prims[:6]}", flush=True)

    # Raycast straight down the bore axis and at increasing radius, from well above the face.
    import omni.physx
    from isaacsim.core.utils.prims import get_prim_at_path  # noqa: F401

    query = omni.physx.get_physx_scene_query_interface()
    cx, cy = 0.52, -0.16  # configs/bdash/chuck_load/task.yaml scene.chuck_pose
    start_z = 0.40
    print("\n  radius(mm)   hit_z(mm)   note")
    for r_mm in (0, 2, 5, 8, 10, 12, 15, 18, 19, 20, 21, 22, 25, 30, 40, 60):
        r = r_mm / 1000.0
        hit = query.raycast_closest((cx + r, cy, start_z), (0.0, 0.0, -1.0), 1.0)
        if hit["hit"]:
            z_mm = hit["position"][2] * 1000.0
            note = hit["rigidBody"].split("/")[-1]
            print(f"  {r_mm:8d}   {z_mm:9.4f}   {note}")
        else:
            print(f"  {r_mm:8d}         ---   (no hit: open through the bore)")

    print("\n  nominal: face at 80.0 mm, bore radius 20.0 mm (open through), body radius 80.0 mm")
    print("  jaws: inner radius 12.5 + jaw_open, z band 55.0 .. 80.0 mm")
    sim.stop()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
