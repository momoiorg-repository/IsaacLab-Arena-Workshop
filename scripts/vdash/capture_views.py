# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Capture Isaac Sim camera frames of the V-DASH pick→insert for the report.

Builds the ``vdash_pick_insert`` env (like ``policy_runner``), drives it with the ``vdash_scripted``
expert, and saves the Franka wrist/left/right RGB frames at a chosen step (default: mid-insertion).

Run inside the container (cameras require rendering, so do NOT pass --headless alone — pass
``--enable_cameras``):

    unset DISPLAY
    /isaac-sim/python.sh scripts/vdash/capture_views.py --policy_type vdash_scripted \
        --headless --enable_cameras --num_envs 1 --seed 0 --capture_step 470 \
        vdash_pick_insert --clearance 2.0 --level L0 --preinsert
"""

from __future__ import annotations

import os

import gymnasium as gym

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--policy_type", type=str, default="vdash_scripted")
    parser.add_argument("--capture_step", type=int, default=470, help="step at which to grab frames")
    parser.add_argument("--out_dir", type=str, default="docs/images/vdash")
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        import matplotlib.image as mpimg
        import numpy as np

        from isaaclab_arena.policy.vdash_scripted_policy import VDashScriptedPolicy, VDashScriptedPolicyArgs
        from isaaclab_arena.utils.random import set_seed
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        env_parser = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli = env_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        name, cfg = arena_builder.build_registered()
        env = gym.make(name, cfg=cfg)
        if args_cli.seed is not None:
            set_seed(args_cli.seed, env)

        policy = VDashScriptedPolicy(VDashScriptedPolicyArgs())
        obs, _ = env.reset()
        policy.reset()

        os.makedirs(args_cli.out_dir, exist_ok=True)
        scene = env.unwrapped.scene
        cam_keys = [k for k in ("wrist_cam", "left_cam", "right_cam") if k in scene.keys()]
        print(f"[capture] cameras in scene: {cam_keys}")

        def grab(tag):
            for k in cam_keys:
                out = scene[k].data.output
                rgb = out["rgb"] if "rgb" in out else next(iter(out.values()))
                img = rgb[0].detach().cpu().numpy()
                if img.dtype != np.uint8:
                    img = (img.clip(0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                img = img[..., :3]
                path = os.path.join(args_cli.out_dir, f"sim_{k}_{tag}.png")
                mpimg.imsave(path, img)
                print(f"[capture] wrote {path}  shape={img.shape}")

        import torch

        capture_at = sorted({args_cli.capture_step, 300, 520, 600, 640})
        for step in range(max(capture_at) + 2):
            with torch.inference_mode():
                action = policy.get_action(env, obs)
                obs, _, term, trunc, _ = env.step(action)
            if step in capture_at:
                grab(f"s{step}")
        print("[capture] DONE")


if __name__ == "__main__":
    main()
