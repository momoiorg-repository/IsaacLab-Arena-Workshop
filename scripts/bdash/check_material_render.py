# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Does the material write actually reach the RENDERED image?

THE POSE MUST BE HELD FIXED. Measuring rendered colour across env.reset() cannot answer this: the
parts are re-scattered every reset, so their shading changes with orientation whether or not the
material changed at all. Measured -- with randomisation switched OFF entirely, so every material
was constant, the across-reset spread was 27.7-54.4 against 44.6-50.7 with it on. The check could
not tell the two apart, and an earlier version of it "confirmed" a randomisation that was in fact
drawing the same colour every episode.

So: reset ONCE, then redraw the material in place, on a scene that has not moved. Any change in the
rendered pixels is then attributable to the material and to nothing else.

    unset DISPLAY
    /isaac-sim/python.sh scripts/bdash/check_material_render.py --enable_cameras --num_envs 1 \\
        --seed 0 --resets 5 bdash_chuck_load --task pick --variants all

Setting a USD shader input and seeing the config change proves nothing: the failure that matters is
a write that never reaches Hydra, which looks identical in every log and produces a dataset where
every part is the same colour. So this measures the TARGET'S OWN PIXELS, isolated with instance-id
segmentation, across resets.
"""

import os
import sys

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--resets", type=int, default=6)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import gymnasium as gym
    import numpy as np
    import torch

    builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = builder.build_registered()
    # left_cam, not right_cam: right_cam is the top-down view and the arm sits in its line of
    # sight, so two of the three parts render zero pixels there and the check cannot see them at all.
    for cam in ("left_cam",):
        cfg = getattr(env_cfg.scene, cam, None)
        if cfg is not None:
            cfg.data_types = list(cfg.data_types) + ["instance_id_segmentation_fast"]
            if hasattr(cfg, "colorize_instance_id_segmentation"):
                cfg.colorize_instance_id_segmentation = False

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    task = env.cfg.isaaclab_arena_env.task
    from isaaclab_arena_environments.mdp import bdash_chuck_materials as bcm
    from isaaclab_arena_environments.mdp.bdash_chuck_config import load_materials_cfg

    print(f"{'draw':>5} {'part':22} {'material drawn':30} {'rendered mean rgb':26} {'px':>6}", flush=True)
    rendered = []
    materials_cfg = load_materials_cfg()["workpiece_randomization"]
    with torch.inference_mode():
        env.reset()
        for _ in range(5):  # settle once; the scene then stays put for every draw below
            env.step(torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device))
            env.sim.step(render=True)
            env.scene.update(env.physics_dt)
        for i in range(args_cli.resets):
            # Redraw IN PLACE. No env.reset(), so nothing moves and nothing is re-scattered.
            # BDASH_PLAIN_MATERIALS skips the redraw entirely -- this is the NEGATIVE CONTROL, and
            # it has to skip THIS call, not merely the task's event config: the first attempt at a
            # control set that variable, which disabled the reset event while this direct call kept
            # randomising, so the "control" measured the randomised case and reported the same large
            # spread. With the redraw skipped the scene is completely static and the residual spread
            # is pure renderer noise, which is the only honest baseline for this measurement.
            if os.environ.get("BDASH_PLAIN_MATERIALS"):
                pass
            else:
                bcm.randomize_workpiece_appearance(
                    env,
                    torch.zeros(1, dtype=torch.long, device=env.device),
                    asset_names=list(task.workpiece_names),
                    cfg=materials_cfg,
                    seed=i,
                )
            for _ in range(2):
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)
            sensor = env.scene.sensors["left_cam"]
            rgb = np.asarray(sensor.data.output["rgb"][0].cpu(), dtype=np.float32)
            seg = np.asarray(sensor.data.output["instance_id_segmentation_fast"][0].cpu()).reshape(
                rgb.shape[0], rgb.shape[1]
            )
            info = sensor.data.info[0]["instance_id_segmentation_fast"]["idToLabels"]
            for k, name in enumerate(task.workpiece_names):
                leaf = env.scene[name].cfg.prim_path.rstrip("/").rsplit("/", 1)[-1]
                ids = [int(key) for key, val in info.items() if f"/{leaf}/" in str(val) or str(val).endswith(leaf)]
                mask = np.isin(seg, ids)
                px = int(mask.sum())
                mean = rgb[mask].mean(axis=0)[:3] if px else np.zeros(3)
                # An MDL draw has no scalar colour -- the material NAME is its identity. Report
                # whichever the draw actually carries so this check works for both paths.
                drawn = getattr(env, "bdash_appearance", None)
                entry = drawn[0][k] if drawn and drawn[0] else {}
                if "rgb" in entry:
                    label = "({:.3f},{:.3f},{:.3f})".format(*entry["rgb"])
                else:
                    label = entry.get("name", "-")[:28]
                rendered.append((i, k, tuple(float(c) for c in mean)))
                print(
                    f"{i:5d} {leaf:22} {label:30} ({mean[0]:6.1f},{mean[1]:6.1f},{mean[2]:6.1f})   {px:6d}",
                    flush=True,
                )
    # The verdict: does the SAME part change colour between resets?
    by_part: dict = {}
    for reset_i, part, mean in rendered:
        by_part.setdefault(part, []).append(mean)
    worst = 1e9
    for part, means in by_part.items():
        spread = max(sum(abs(a - b) for a, b in zip(m1, m2)) for i, m1 in enumerate(means) for m2 in means[i + 1 :])
        print(f"part {part}: max pairwise rendered-colour spread at FIXED POSE = {spread:.1f}", flush=True)
        worst = min(worst, spread)
    # Threshold from the MEASURED noise floor, not a guess. With the redraw skipped (the control,
    # BDASH_PLAIN_MATERIALS=1) the same scene at the same pose scores 2.0-4.0, so anything at or
    # under ~4 is renderer noise. 5.0 sits just above it.
    #
    # Note what this statistic can and cannot do: it compares MEAN colour, so two materials with a
    # similar base tone -- brushed aluminium against ash or pine, say -- separate by only ~6 even
    # though they are visibly different in grain. A texture-variance term would discriminate those
    # far better; mean colour is enough to answer "did the bind take effect at all", which is the
    # question this script exists for.
    print(f"MATERIAL_RENDER {'OK' if worst > 5.0 else 'NOT_REACHING_RENDER'} (min spread {worst:.1f})", flush=True)
    env.close()
except BaseException:  # noqa: BLE001
    import traceback

    traceback.print_exc(file=sys.stdout)
    print("MATERIAL_RENDER_FAIL", flush=True)
finally:
    simulation_app.close()
