# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure, per external camera, how often the target workpiece is actually visible during approach.

This is the acceptance test for the camera placement, and it exists because the failure it looks
for is invisible in a still frame. The arm approaches every grasp from directly above the part, so
a camera near the workspace centre sees the target fine at reset, fine in a screenshot, and NOT AT
ALL for the last half-second before the gripper closes -- which is the half-second the policy most
needs. Judging framing by eye on one rendered frame is what lets that through.

Two distinct failure modes are counted separately, because they have opposite fixes:

* OUT OF FRAME  -- the grasp point projects outside the image. Fix: move/zoom the camera.
* OCCLUDED      -- it projects inside the image but something (nearly always the arm) is in front.
                   Fix: move the camera off the arm's approach axis. This is the one an overhead
                   view cannot pass.

Occlusion is measured with ``instance_id_segmentation_fast`` rather than inferred: a pixel belongs
to the target or it does not. The segmentation output is added to the cameras HERE and never in the
recorded configuration, so the dataset is unaffected.

    unset DISPLAY
    /isaac-sim/python.sh scripts/bdash/audit_camera_visibility.py --enable_cameras --num_envs 1 \\
        --seed 0 --episodes 20 bdash_chuck_load --task pick --variants all
"""

import os

print("[camaudit] launching Isaac Sim (~90s)...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--policy_type", type=str, default="bdash_chuck_teacher")
parser.add_argument("--min_pixels", type=int, default=20, help="target pixels below which it is not usefully visible")
parser.add_argument("--pass_rate", type=float, default=0.95, help="acceptance bar for both criteria")
parser.add_argument("--jsonl", type=str, default="logs/bdash/cam_audit.jsonl")
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import collections
import gymnasium as gym
import json
import torch

from isaaclab_arena.controllers.scripted_pick import CLOSE
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.utils.random import set_seed

CAMERAS = ("right_cam", "left_cam", "wrist_cam")


def _target_pixels(camera, index, prim_leaf):
    """Pixels of ``camera``'s instance-id frame belonging to the prim whose path ends ``prim_leaf``.

    Matching on the LEAF rather than the full path, and rebuilding the mapping every frame, are
    both deliberate. The configured ``prim_path`` still carries the ``env_.*`` regex, so a prefix
    built from it matches nothing (measured: every camera reported 0% visibility, including the
    wrist camera, which is impossible). And the renderer assigns ids per frame with no promise of
    stability, so a cached mapping would silently start counting a different body.

    The frame arrives COLORIZED -- (H, W, 4) uint8 with RGBA tuples as the mapping keys -- unless
    the camera is configured otherwise, which ``main`` does. Both forms are handled so this cannot
    quietly return zero again if that flag stops taking effect.
    """
    seg = camera.data.output.get("instance_id_segmentation_fast")
    info = camera.data.info[index].get("instance_id_segmentation_fast") if camera.data.info else None
    if seg is None or not info:
        return None, 0
    mapping = info.get("idToLabels", info)
    keys = [k for k, v in mapping.items() if str(v).rstrip("/").endswith(prim_leaf) or f"/{prim_leaf}/" in str(v)]
    if os.environ.get("BDASH_SEG_DBG"):
        print(f"[segdbg] leaf={prim_leaf!r} seg={tuple(seg.shape)} {seg.dtype} matched={keys}", flush=True)
    frame = seg[index]
    if frame.ndim == 3 and frame.shape[-1] == 4:  # colorized RGBA
        total = int(frame.shape[0] * frame.shape[1])
        if not keys:
            return 0, total
        hit = torch.zeros(frame.shape[:2], dtype=torch.bool, device=frame.device)
        for key in keys:
            colour = torch.tensor([int(c) for c in key], dtype=frame.dtype, device=frame.device)
            hit |= (frame == colour).all(dim=-1)
        return int(hit.sum()), total
    flat = frame.reshape(-1)
    if not keys:
        return 0, int(flat.numel())
    want = torch.tensor([int(k) for k in keys], device=flat.device, dtype=flat.dtype)
    return int(torch.isin(flat, want).sum()), int(flat.numel())


def _contrast(camera, index, mask_ids):
    """Mean |RGB| difference between the target's pixels and the ring of pixels around them.

    Segmentation answers "is the target in the image"; it cannot answer "can it be TOLD APART from
    what it is lying on". With the workpiece appearance now sampled down to value 0.18 (black
    oxide) against a mid-value blue tray, a part can be fully unoccluded and still nearly
    invisible, and that failure would pass every other check in this script.
    """
    import numpy as np

    seg = camera.data.output.get("instance_id_segmentation_fast")
    rgb = camera.data.output.get("rgb")
    if seg is None or rgb is None or not mask_ids:
        return None
    frame = np.asarray(rgb[index].cpu(), dtype=np.float32)[..., :3]
    ids = np.asarray(seg[index].cpu()).reshape(frame.shape[0], frame.shape[1])
    mask = np.isin(ids, list(mask_ids))
    if not mask.any():
        return None
    # 5-pixel dilation minus the mask itself = the immediate surround. Done SEPARABLY (11 rolls per
    # axis, not 121 over the square): a square structuring element is the outer product of two
    # 1-D ones, and the naive version dominated the whole audit's runtime at ~3 min/episode.
    pad = mask.copy()
    for axis in (0, 1):
        grown = pad.copy()
        for shift in range(-5, 6):
            if shift:
                grown |= np.roll(pad, shift, axis=axis)
        pad = grown
    ring = pad & ~mask
    if not ring.any():
        return None
    return float(np.abs(frame[mask].mean(axis=0) - frame[ring].mean(axis=0)).sum())


def _in_frame(camera, index, point_w):
    """Is ``point_w`` inside the image, in front of the lens? Uses the camera's own intrinsics."""
    from isaaclab.utils.math import quat_apply_inverse

    pos = camera.data.pos_w[index]
    quat = camera.data.quat_w_ros[index]
    local = quat_apply_inverse(quat.unsqueeze(0), (point_w - pos).unsqueeze(0))[0]
    if float(local[2]) <= 1e-6:  # ROS convention: +Z is forward
        return False
    k = camera.data.intrinsic_matrices[index]
    u = float(k[0, 0]) * float(local[0]) / float(local[2]) + float(k[0, 2])
    v = float(k[1, 1]) * float(local[1]) / float(local[2]) + float(k[1, 2])
    return 0.0 <= u < camera.image_shape[1] and 0.0 <= v < camera.image_shape[0]


def main() -> None:
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    # Audit-only: the recorded configuration keeps ["rgb"], so nothing here reaches the dataset.
    for name in CAMERAS:
        cam_cfg = getattr(env_cfg.scene, name, None)
        if cam_cfg is not None and "instance_id_segmentation_fast" not in cam_cfg.data_types:
            cam_cfg.data_types = list(cam_cfg.data_types) + ["instance_id_segmentation_fast"]
            # Raw ids, not an RGBA palette: colorized output makes the mapping keys colour tuples
            # and turns a pixel count into a colour match, which is both slower and easy to get
            # silently wrong.
            if hasattr(cam_cfg, "colorize_instance_id_segmentation"):
                cam_cfg.colorize_instance_id_segmentation = False
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if args_cli.seed is not None:
        set_seed(args_cli.seed, env)

    task = env.cfg.isaaclab_arena_env.task
    policy = get_policy_cls(args_cli.policy_type).from_args(args_cli)
    os.makedirs(os.path.dirname(args_cli.jsonl) or ".", exist_ok=True)
    handle = open(args_cli.jsonl, "w")  # noqa: SIM115

    # Print where the cameras ACTUALLY are. The configured offset has to survive the embodiment,
    # the cfg callback and the scene build; a pose that silently did not take would make every
    # number below a measurement of the wrong camera.
    env.reset()
    for cam in CAMERAS:
        sensor = env.scene.sensors.get(cam)
        if sensor is not None:
            pos = [round(float(v), 4) for v in sensor.data.pos_w[0]]
            print(f"[camaudit] {cam:10} world pos {pos}", flush=True)

    totals = {name: collections.Counter() for name in CAMERAS}
    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            env.reset()
            policy.reset()
            target = int(env.bdash_target_idx[0].item())
            name = task.workpiece_names[target]
            # The LEAF prim name. `cfg.prim_path` still holds the `env_.*` regex at this point.
            leaf = env.scene[name].cfg.prim_path.rstrip("/").rsplit("/", 1)[-1]
            per_cam = {cam: collections.Counter() for cam in CAMERAS}
            # AT THE GRASP, not averaged over the approach. The arm only occludes for the last
            # handful of steps, so a rate over ~110 approach steps dilutes a total blackout at the
            # decisive moment down to a few percent -- measured, an overhead camera scored the same
            # 100% as a side camera on the averaged figures. This is the number that separates them.
            last_px = {cam: None for cam in CAMERAS}
            # The WORST moment of the approach, which is what an occlusion argument is really about.
            min_px = {cam: None for cam in CAMERAS}
            min_contrast: dict = {}

            for _ in range(args_cli.max_steps):
                action = policy.get_action(env, None)
                # APPROACH and DESCEND only -- criterion (a) is about the run-in to the grasp, and
                # after CLOSE the part is in the hand and its visibility means something else.
                approaching = policy.phase is not None and int(policy.phase[0]) < CLOSE
                if approaching:
                    point = env.scene[name].data.root_pos_w[0]
                    for cam in CAMERAS:
                        sensor = env.scene.sensors.get(cam)
                        if sensor is None:
                            continue
                        pixels, _ = _target_pixels(sensor, 0, leaf)
                        counter = per_cam[cam]
                        if cam != "wrist_cam":
                            seg = sensor.data.output.get("instance_id_segmentation_fast")
                            info = sensor.data.info[0].get("instance_id_segmentation_fast") if seg is not None else None
                            mapping = (info or {}).get("idToLabels", info) or {}
                            ids = [
                                int(key)
                                for key, val in mapping.items()
                                if f"/{leaf}/" in str(val) or str(val).rstrip("/").endswith(leaf)
                            ]
                            value = _contrast(sensor, 0, ids)
                            if value is not None:
                                counter["contrast_sum"] += value
                                counter["contrast_n"] += 1
                                low = min_contrast.get(cam)
                                min_contrast[cam] = value if low is None else min(low, value)
                        counter["steps"] += 1
                        counter["in_frame"] += int(_in_frame(sensor, 0, point))
                        if pixels is not None:
                            counter["seen"] += int(pixels > 0)
                            counter["useful"] += int(pixels >= args_cli.min_pixels)
                            counter["pixels"] += pixels
                            last_px[cam] = pixels
                            worst = min_px[cam]
                            min_px[cam] = pixels if worst is None else min(worst, pixels)
                _obs, _rew, terminated, _trunc, _info = env.step(action)
                # CAMERA REFRESH (critical, same fix as record_vla_demos.py): on this box a
                # standalone `sim.render()` -- what the env's decimation loop uses -- does NOT
                # refresh the RTX camera sensors. Without this pump every frame read below is the
                # frozen reset render, which is exactly what happened first time round: the target's
                # pixel count was constant for a whole episode and took only three distinct values
                # across ten, one per variant, i.e. it was measuring projected area and never
                # occlusion at all.
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)
                if bool(terminated[0]):
                    break

            record = {"episode": episode, "target": target, "variant": task.profiles[target]["variant"]}
            for cam in CAMERAS:
                counter = per_cam[cam]
                steps = max(1, counter["steps"])
                record[cam] = {
                    "steps": counter["steps"],
                    "in_frame": round(counter["in_frame"] / steps, 4),
                    "visible": round(counter["seen"] / steps, 4),
                    "useful": round(counter["useful"] / steps, 4),
                    "mean_px": round(counter["pixels"] / steps, 1),
                    "grasp_px": last_px[cam],
                    "min_px": min_px[cam],
                    "min_contrast": None if min_contrast.get(cam) is None else round(min_contrast[cam], 1),
                }
                if last_px[cam] is not None:
                    totals[cam]["grasp_episodes"] += 1
                    totals[cam]["grasp_visible"] += int(last_px[cam] >= args_cli.min_pixels)
                totals[cam].update(counter)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"[camaudit] ep {episode:3d} {record['variant']}  "
                + "  ".join(
                    f"{c.split('_')[0]}: frame {record[c]['in_frame']:.0%} vis {record[c]['visible']:.0%}"
                    for c in CAMERAS
                ),
                flush=True,
            )

    print()
    print(f"{'camera':12} {'steps':>7} {'in frame':>10} {'visible':>10} {'>=' + str(args_cli.min_pixels) + 'px':>10}")
    verdict = True
    for cam in CAMERAS:
        counter = totals[cam]
        steps = max(1, counter["steps"])
        rates = (counter["in_frame"] / steps, counter["seen"] / steps, counter["useful"] / steps)
        print(f"{cam:12} {counter['steps']:7d} " + " ".join(f"{r:9.1%}" for r in rates), flush=True)
        # The wrist cam is exempt: it is ON the hand, so "the arm occludes it" is meaningless and
        # the part leaves its frame by design during the reach.
        grasp_n = counter["grasp_episodes"]
        grasp_rate = counter["grasp_visible"] / grasp_n if grasp_n else 0.0
        line = f"{'':12} {'':7} at grasp: visible in {counter['grasp_visible']}/{grasp_n} episodes"
        if counter["contrast_n"]:
            line += f";  mean target-vs-surround contrast {counter['contrast_sum'] / counter['contrast_n']:.1f}"
        print(line, flush=True)
        if cam != "wrist_cam":
            verdict &= rates[0] >= args_cli.pass_rate and rates[2] >= args_cli.pass_rate
            verdict &= grasp_rate >= args_cli.pass_rate
    print(f"CAM_AUDIT {'PASS' if verdict else 'FAIL'} (bar {args_cli.pass_rate:.0%})", flush=True)
    handle.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
