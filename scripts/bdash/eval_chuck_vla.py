# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One-shot S3 evaluation runner: a fine-tuned GR00T chuck model, scored live against the
pre-committed protocol (docs/rsj2026_s3_eval_protocol_ja.md).

Lean by design -- check_chuck_teacher.py demands a dozen teacher-only attributes (pick phases,
stall stats, grasp latches) that a VLA policy does not have, so this is a separate runner rather
than a stubbed shoehorn. Success is read straight from the scene with the SAME criteria as
scripts/bdash/audit_chuck_demos.py (axis upright, >=20 mm below the face, <=10 mm off axis), so a
"success" here cannot be the false-success the chuck_closed cut alone would admit.

Per-episode language is built from the drawn target's colour (instruction_for), i.e. the task
specification the planner would issue -- generating the sentence from the sampler's draw is not
perception privilege. `--cut lift` scores grasp+lift only (hierarchy upstream); `--cut load`
scores full loading incl. the seated audit.

Run inside the gn17 container (transformers 4.57.3 for N1.7):
  /isaac-sim/python.sh scripts/bdash/eval_chuck_vla.py --enable_cameras --num_envs 1 --headless \
    --vla_config isaaclab_arena_gr00t/policy/config/bdash_chuck_full247_closedloop_config.yaml \
    --cut load --episodes 20 --seed 17 --jsonl logs/bdash/s3_eval_full.jsonl \
    bdash_chuck_load --task full --variants all --embodiment franka_joint

TODO(held-out colours): report-only slice needs the appearance pool switched to the held-out set;
not wired yet -- the protocol keeps it non-gating.
"""

# ruff: noqa: E402
import argparse  # noqa: F401  (parser comes from the arena CLI)
import sys

print("[eval] launching Isaac Sim -- ~90s before the first episode...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--vla_config", required=True, help="Gr00t closedloop config yaml")
parser.add_argument("--cut", choices=["lift", "load"], required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=1600)
parser.add_argument("--jsonl", required=True)
parser.add_argument("--hier", action="store_true", help="run the VLA->classical hierarchy instead of the bare VLA")
parser.add_argument(
    "--dump_frames_dir",
    default=None,
    help="save a 3-view frame every 4 steps per episode (diagnosis/video; not used for scoring)",
)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import hashlib
import json
import os
import pathlib
import torch

VKEY = {"W-A": "bdash_workpiece_wa_0", "W-B": "bdash_workpiece_wb_0", "W-C": "bdash_workpiece_wc_0"}
HEIGHT = {"W-A": 0.09, "W-B": 0.10, "W-C": 0.08}
VARIANTS = ("W-A", "W-B", "W-C")


def _fingerprint() -> dict:
    cfg_dir = pathlib.Path(__file__).resolve().parents[2] / "configs/bdash/chuck_load"
    digest = hashlib.sha256()
    for f in sorted(cfg_dir.glob("*.yaml")):
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    return {"config_sha": digest.hexdigest()[:12]}


def _axis_z(q: torch.Tensor) -> float:
    x, y = float(q[1]), float(q[2])
    return 1.0 - 2.0 * (x * x + y * y)


def main() -> None:
    fingerprint = _fingerprint()
    print(f"[eval] scene/config sha={fingerprint['config_sha']}", flush=True)
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    from isaaclab_arena.utils.random import set_seed
    from isaaclab_arena_environments.mdp import bdash_chuck_language as bcl
    from isaaclab_arena_environments.mdp import bdash_chuck_predicates as vp
    from isaaclab_arena_gr00t.policy.gr00t_closedloop_policy import Gr00tClosedloopPolicy, Gr00tClosedloopPolicyArgs

    if args_cli.seed is not None:
        set_seed(args_cli.seed, env)
    if args_cli.hier:
        from isaaclab_arena.policy.bdash_chuck_hier_policy import BDashChuckHierPolicy

        policy = BDashChuckHierPolicy.from_args(args_cli)
    else:
        policy = Gr00tClosedloopPolicy(
            Gr00tClosedloopPolicyArgs(policy_config_yaml_path=args_cli.vla_config, policy_device="cuda:0", num_envs=1)
        )

    chuck = env.scene["bdash_chuck_body"]
    os.makedirs(os.path.dirname(args_cli.jsonl) or ".", exist_ok=True)
    out = open(args_cli.jsonl, "w")  # noqa: SIM115 -- flushed per episode, closed at the end
    passes = 0
    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            obs, _ = env.reset()
            policy.reset()
            tgt = int(env.bdash_target_idx[0])
            variant = VARIANTS[int(env.bdash_target_variant[0])]
            entry = env.bdash_appearance[0][tgt] if getattr(env, "bdash_appearance", None) else None
            language = bcl.instruction_for(entry, action="load")
            if args_cli.cut == "lift":
                language = language.replace("load it into the chuck", "lift it clear")
            policy.set_task_description(language)

            wp = env.scene[VKEY[variant]]
            z0 = float(wp.data.root_pos_w[0, 2])
            lifted_step = None
            loaded_step = None
            for step in range(args_cli.max_steps):
                action = policy.get_action(env, obs)
                obs, _, _, _, _ = env.step(action)
                if args_cli.dump_frames_dir and step % 4 == 0:
                    import numpy as np

                    from PIL import Image

                    cam = obs.get("camera_obs") if isinstance(obs, dict) else None
                    if cam is not None:
                        tiles = [
                            np.asarray(cam[k][0].detach().cpu())[..., :3] for k in sorted(cam) if k.endswith("_rgb")
                        ]
                        if tiles:
                            d = pathlib.Path(args_cli.dump_frames_dir) / f"ep{episode}"
                            d.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(np.concatenate(tiles, axis=1).astype("uint8")).save(d / f"f{step:05d}.png")
                if lifted_step is None:
                    q = env.scene["robot"].data.joint_pos[0]
                    width = float(q[-2] + q[-1])
                    if float(wp.data.root_pos_w[0, 2]) > z0 + 0.060 and 0.015 < width < 0.065:
                        lifted_step = step
                        if args_cli.cut == "lift":
                            break
                if args_cli.cut == "load" and bool(vp.chuck_closed(env)[0]):
                    loaded_step = step
                    break

            # Seated audit, live -- same window as audit_chuck_demos.py.
            p = wp.data.root_pos_w[0]
            qw = wp.data.root_quat_w[0]
            az = _axis_z(qw)
            ch = chuck.data.root_pos_w[0]
            face = float(ch[2]) + 0.08
            depth_mm = (face - (float(p[2]) - HEIGHT[variant] / 2 * az)) * 1000
            xy_mm = float(torch.hypot(p[0] - ch[0], p[1] - ch[1])) * 1000
            seated = az > 0.95 and depth_mm > 20 and xy_mm < 10
            ok = (lifted_step is not None) if args_cli.cut == "lift" else (loaded_step is not None and seated)
            passes += int(ok)
            rec = dict(
                episode=episode,
                variant=variant,
                language=language,
                cut=args_cli.cut,
                lifted_step=lifted_step,
                loaded_step=loaded_step,
                axis_z=round(az, 3),
                depth_mm=round(depth_mm, 1),
                xy_mm=round(xy_mm, 1),
                seated=bool(seated),
                ok=bool(ok),
                **fingerprint,
            )
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"[eval] ep{episode} {variant} ok={ok} lifted={lifted_step} loaded={loaded_step}", flush=True)
    out.close()
    print(f"S3_EVAL_DONE cut={args_cli.cut} pass={passes}/{args_cli.episodes}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Isaac Sim swallows exceptions on the way down; print before it can.
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
