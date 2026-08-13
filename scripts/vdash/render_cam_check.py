# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Render the three VLA camera views at a fixed L1 reset — a quick check on framing / peg pixel size.

Reuses the proven record_vla_demos setup (scripted expert, single env, cameras). Resets seed-fixed so
two runs at different ``--focal`` render the SAME peg/socket pose for a clean A/B. Saves
``{out_prefix}_{wrist,left,right}.png`` to ``--out_dir``.

    unset DISPLAY
    /isaac-sim/python.sh scripts/vdash/render_cam_check.py --enable_cameras --num_envs 1 --seed 0 \\
        --focal 2.8 --out_prefix before vdash_pick_insert --clearance 2.0 --level L1
"""

import os

print("[camcheck] launching Isaac Sim (~90s)...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--focal", type=float, default=None, help="override side-cam focal_length (mm); None = leave as configured"
)
parser.add_argument("--out_dir", type=str, default="logs/vdash/camcheck")
parser.add_argument("--out_prefix", type=str, default="frame")
parser.add_argument(
    "--grab_step", type=int, default=8, help="step at which to grab frames (arm still high in APPROACH)"
)
parser.add_argument(
    "--n_resets", type=int, default=1, help="render this many consecutive L1 resets (spread check); suffix _r{i}"
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="vdash_scripted",
    help="policy driving the arm while framing (use zero_action for scenes without a scripted expert)",
)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from PIL import Image

from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.utils.random import set_seed


def _save(frame, path):
    t = frame[0] if frame.ndim == 4 else frame
    arr = t.detach().cpu().numpy()
    if arr.dtype != "uint8":
        arr = (arr * 255).clip(0, 255).astype("uint8") if arr.max() <= 1.0 else arr.clip(0, 255).astype("uint8")
    Image.fromarray(arr[..., :3]).save(path)


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    if args_cli.focal is not None:
        for cam in ("left_cam", "right_cam"):
            c = getattr(env_cfg.scene, cam, None)
            if c is not None and getattr(c, "spawn", None) is not None:
                c.spawn.focal_length = args_cli.focal
                print(f"[camcheck] {cam}.spawn.focal_length -> {args_cli.focal}", flush=True)

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if args_cli.seed is not None:
        set_seed(args_cli.seed, env)
    policy = get_policy_cls(args_cli.policy_type).from_args(args_cli)

    with torch.inference_mode():
        for i in range(args_cli.n_resets):
            env.reset()
            policy.reset()
            for _ in range(args_cli.grab_step):
                env.step(policy.get_action(env, None))
            cams = env.obs_buf["camera_obs"]
            sfx = "" if args_cli.n_resets == 1 else f"_r{i}"
            # Best-effort: only the vdash task exposes a "peg" name. Other scenes (e.g. bdash) are
            # framed against fixture poses instead, so a missing name must not abort the check.
            try:
                names = env.unwrapped.cfg.isaaclab_arena_env.task.names
                tracked = env.unwrapped.scene[names["peg"]].data.root_pos_w[0]
                print(f"[camcheck] reset {i}: peg=({tracked[0]:.3f},{tracked[1]:.3f})", flush=True)
            except (AttributeError, KeyError, TypeError):
                print(f"[camcheck] reset {i}", flush=True)
            for key, short in (("wrist_cam_rgb", "wrist"), ("left_cam_rgb", "left"), ("right_cam_rgb", "right")):
                if key in cams:
                    p = os.path.join(args_cli.out_dir, f"{args_cli.out_prefix}_{short}{sfx}.png")
                    _save(cams[key], p)
                    print(f"[camcheck] saved {p}", flush=True)
    print("CAMCHECK_DONE", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
