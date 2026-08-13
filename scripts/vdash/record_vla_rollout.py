# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Record start-to-finish VLA inference rollouts of the V-DASH pick->insert as mp4.

Runs the fine-tuned VLA policy episode by episode (num_envs=1), grabs the wrist/left/right
camera frames every ``--stride`` sim steps, and writes one mp4 per episode named by outcome
(``vla_rollout_ep03_success.mp4``). Cameras require rendering:

    unset DISPLAY
    /isaac-sim/python.sh scripts/vdash/record_vla_rollout.py \\
        --policy_type isaaclab_arena.policy.vdash_vla_policy.VDashVLAPolicy \\
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/vdash_pick_insert_gr00t_closedloop_v6_recovery_config.yaml \\
        --headless --enable_cameras --num_envs 1 --seed 3 --episodes 8 \\
        vdash_pick_insert --clearance 2.0 --level L1
"""

from __future__ import annotations

import gymnasium as gym
import os

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=900, help="per-episode step cap")
    parser.add_argument("--stride", type=int, default=3, help="record every N sim steps")
    parser.add_argument("--fps", type=int, default=20, help="output video fps")
    parser.add_argument("--out_dir", type=str, default="results/vdash/rollout_videos")
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        import numpy as np
        import torch

        import imageio

        from isaaclab_arena.evaluation.policy_runner import get_policy_cls
        from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
        from isaaclab_arena.utils.random import set_seed
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        add_policy_runner_arguments(parser)
        env_parser = get_isaaclab_arena_environments_cli_parser(parser)
        policy_cls = get_policy_cls(parser.parse_known_args()[0].policy_type)
        env_parser = policy_cls.add_args_to_parser(env_parser)
        args_cli = env_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        name, cfg = arena_builder.build_registered()
        env = gym.make(name, cfg=cfg)
        if args_cli.seed is not None:
            set_seed(args_cli.seed, env)
        policy = policy_cls.from_args(args_cli)
        if hasattr(policy, "set_task_description"):
            policy.set_task_description(env.unwrapped.cfg.isaaclab_arena_env.task.get_task_description())

        scene = env.unwrapped.scene
        cam_keys = [k for k in ("wrist_cam", "left_cam", "right_cam") if k in scene.keys()]
        print(f"[rollout] cameras: {cam_keys}")
        os.makedirs(args_cli.out_dir, exist_ok=True)

        def frame() -> np.ndarray:
            views = []
            for k in cam_keys:
                out = scene[k].data.output
                rgb = out["rgb"] if "rgb" in out else next(iter(out.values()))
                img = rgb[0].detach().cpu().numpy()
                if img.dtype != np.uint8:
                    img = (img.clip(0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                views.append(img[..., :3])
            h = min(v.shape[0] for v in views)
            return np.hstack([v[:h] for v in views])

        for ep in range(args_cli.episodes):
            frames = []
            success = False
            with torch.inference_mode():  # keep resets inside: policy state tensors are inference tensors
                obs, _ = env.reset()
                policy.reset()
                for step in range(args_cli.max_steps):
                    action = policy.get_action(env, obs)
                    obs, _, term, trunc, _ = env.step(action)
                    if step % args_cli.stride == 0:
                        frames.append(frame())
                    fin = term | trunc
                    if bool(fin[0]):
                        success = bool(env.unwrapped.termination_manager.get_term("success")[0])
                        break
            tag = "success" if success else "fail"
            path = os.path.join(args_cli.out_dir, f"vla_rollout_ep{ep:02d}_{tag}.mp4")
            imageio.mimwrite(path, frames, fps=args_cli.fps, codec="libx264", quality=7)
            print(f"[rollout] ep{ep:02d}: {tag}  steps={step + 1}  frames={len(frames)} -> {path}", flush=True)
        print("[rollout] DONE")


if __name__ == "__main__":
    main()
