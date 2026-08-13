# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""De-risk the M8 overhead camera + classic-vision estimate against ground truth.

Builds the bdash env with the top-down camera (needs --enable_cameras), holds the peg static for a
few steps, runs :class:`ClassicVisionGrasp`, and prints the per-env error vs the true grasp point;
saves the env-0 RGB + blue mask to docs/images/bdash/ for a visual check.

    unset DISPLAY
    /isaac-sim/python.sh scripts/bdash/test_overhead_cam.py --enable_cameras --num_envs 4 --seed 0 \\
        bdash_pick_insert --clearance 2.0 --level L0
"""

from __future__ import annotations

import gymnasium as gym
import os
import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "images", "bdash")


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--hold_steps", type=int, default=8)
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from isaaclab.utils.math import quat_apply

        from isaaclab_arena.controllers.bdash_classic_vision import ClassicVisionGrasp, segment_blue
        from isaaclab_arena.utils.random import set_seed
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        env_parser = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli = env_parser.parse_args()
        arena_builder = get_arena_builder_from_cli(args_cli)
        name, cfg = arena_builder.build_registered()
        print("[cam] built cfg, calling gym.make...", flush=True)
        env = gym.make(name, cfg=cfg)
        print("[cam] gym.make done", flush=True)
        if args_cli.seed is not None:
            set_seed(args_cli.seed, env)

        task = env.unwrapped.cfg.isaaclab_arena_env.task
        names = task.names
        grip_offset = 0.060
        vision = ClassicVisionGrasp("overhead_cam")

        # explicit traceback so a post-build error isn't swallowed by SimulationAppContext
        try:
            obs, _ = env.reset()
            print("[cam] reset done", flush=True)
            n = env.unwrapped.num_envs
            adim = env.action_space.shape[-1] if getattr(env, "action_space", None) is not None else 8
            act = torch.zeros(n, adim, device=env.unwrapped.device)
            for _ in range(args_cli.hold_steps):
                with torch.inference_mode():
                    obs, _, _, _, _ = env.step(act)
            print("[cam] hold loop done", flush=True)

            est, valid = vision.estimate(env)
            print("[cam] estimate done", flush=True)
            # ground-truth grasp point: peg root + grip_offset along the peg axis
            peg = env.unwrapped.scene[names["peg"]].data
            off = torch.zeros_like(peg.root_pos_w)
            off[:, 2] = grip_offset
            gt = peg.root_pos_w + quat_apply(peg.root_quat_w, off)
            err = torch.norm((est - gt)[:, :2], dim=-1) * 1000.0  # lateral error (mm)
            zerr = (est[:, 2] - gt[:, 2]) * 1000.0
            for i in range(n):
                print(
                    f"[cam] env{i} valid={int(valid[i])} est=({est[i, 0]:.3f},{est[i, 1]:.3f},{est[i, 2]:.3f}) "
                    f"gt=({gt[i, 0]:.3f},{gt[i, 1]:.3f},{gt[i, 2]:.3f}) lat_err={err[i]:.1f}mm z_err={zerr[i]:.1f}mm",
                    flush=True,
                )

            # save env-0 rgb + mask
            os.makedirs(_OUT, exist_ok=True)
            rgb = env.unwrapped.scene["overhead_cam"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            if rgb.max() <= 1.5:
                rgb = (rgb * 255).astype(np.uint8)
            mask = segment_blue(env.unwrapped.scene["overhead_cam"].data.output["rgb"])[0].detach().cpu().numpy()
            fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
            ax[0].imshow(rgb.astype(np.uint8))
            ax[0].set_title("overhead rgb")
            ax[0].axis("off")
            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title(f"blue mask ({int(mask.sum())} px)")
            ax[1].axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(_OUT, "m8_overhead_check.png"), dpi=120)
            print(f"[cam] saved {os.path.join(_OUT, 'm8_overhead_check.png')}")
            print("OVERHEAD_CAM_OK")
        except Exception:
            import traceback

            print("[cam] EXCEPTION:", flush=True)
            traceback.print_exc()
        env.close()


if __name__ == "__main__":
    main()
