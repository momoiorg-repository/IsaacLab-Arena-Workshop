# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Vectorized (num_envs > 1) recorder for the chuck LIFT slice -- the v2 learning-data campaign.

A separate script, NOT a --multi flag on record_vla_demos.py: that script's single-env loop is the
validated artifact behind full247 and stays frozen. This one exists because 1000 demos at 54/h is
an 18-hour night, and the tensor machinery (policy, controllers, predicates) was vectorized from
day one -- only the episode ORCHESTRATION was env-0-bound. Here every env runs its own episode:
per-env cut detection (the vectorized `lifted` predicate), per-env language from its own drawn
target, per-env partial reset + settle, per-env recovery kicks, per-env frame-freeze counters.

Scope: chuck family, cut = lifted, upright scene. This is deliberate -- the S3 post-mortem
(docs/progress/2026-08-26_s3_eval.md) concluded the hierarchy needs the lift slice only, and a
narrower recorder is a checkable recorder. Camera-render pump (the CRITICAL fix from the single-env
recorder) is per sim-step and refreshes every env's sensors at once, which is why multi-env pays.

Run (isaaclab_arena-latest container):
  BDASH_PICK_SLOW_STEP=0.02 /isaac-sim/python.sh scripts/bdash/record_vla_demos_vec.py \
    --enable_cameras --num_envs 5 --headless --policy_type bdash_chuck_teacher \
    --seed 21 --num_demos 500 --dataset_file datasets/bdash/lift_v2_clean.hdf5 \
    bdash_chuck_load --task full --variants all
"""

# ruff: noqa: E402
import sys

print("[vrec] launching Isaac Sim -- startup takes ~90s...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--dataset_file", required=True)
parser.add_argument("--num_demos", type=int, default=500)
parser.add_argument("--policy_type", type=str, default="bdash_chuck_teacher")
parser.add_argument("--max_steps_per_demo", type=int, default=900)
parser.add_argument(
    "--settle_steps", type=int, default=60, help="upright-only scene settles fast (was 120 for side-lying creep)"
)
parser.add_argument("--arm_init_std", type=float, default=None)
parser.add_argument("--perturb_frac", type=float, default=0.0)
parser.add_argument("--perturb_mag", type=float, default=0.15)
parser.add_argument("--perturb_window", type=int, nargs=2, default=(30, 120))
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
import os
import random
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class CameraObsRecorder(RecorderTerm):
    def record_pre_step(self):
        cam = self._env.obs_buf.get("camera_obs")
        return ("obs", cam) if cam is not None else (None, None)


@configclass
class CameraObsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CameraObsRecorder


@configclass
class VecRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_camera_obs: CameraObsRecorderCfg = CameraObsRecorderCfg()


def main():
    out_dir = os.path.dirname(args_cli.dataset_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    if args_cli.arm_init_std is not None and hasattr(env_cfg.events, "randomize_franka_joint_state"):
        env_cfg.events.randomize_franka_joint_state.params["std"] = args_cli.arm_init_std
        print(f"[vrec] arm start-pose std -> {args_cli.arm_init_std} rad", flush=True)
    for k in list(vars(env_cfg.terminations)):
        setattr(env_cfg.terminations, k, None)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = VecRecorderManagerCfg(
        dataset_export_dir_path=out_dir,
        dataset_filename=fname,
        dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
    )
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    from isaaclab_arena.evaluation.policy_runner import get_policy_cls
    from isaaclab_arena_environments.mdp import bdash_chuck_language as bcl
    from isaaclab_arena_environments.mdp import bdash_chuck_predicates as vp

    if args_cli.seed is not None:
        from isaaclab_arena.utils.random import set_seed

        set_seed(args_cli.seed, env)
    policy = get_policy_cls(args_cli.policy_type).from_args(args_cli)
    task = env.cfg.isaaclab_arena_env.task
    t = task.cfg
    grasped = t["grasped"]
    names = task.names
    lift_params = dict(
        workpiece_names=task.workpiece_names,
        tray_name=task.tray.name,
        robot_name=names["robot"],
        finger_sensors=task.finger_sensor_names,
        rim_height=float(t["geometry"]["tray_rim_height"]),
        height=float(t["lifted"]["height"]),
        speed_max=float(t["lifted"]["speed_max"]),
        tip_offset=tuple(t["geometry"]["tip_offset"]),
        width_min=float(grasped["width_min"]),
        width_max=float(grasped["width_max"]),
        grasp_force=float(grasped["grasp_force"]),
    )

    n = env.num_envs
    dev = env.device
    rng = random.Random(args_cli.seed or 0)
    jsonl = open(os.path.join(out_dir, f"{fname}_attempts.jsonl"), "w")  # noqa: SIM115

    recorded = 0
    attempts = 0
    kick_at = [-1] * n
    kicked = [False] * n
    frozen = torch.zeros(n, dtype=torch.long, device=dev)
    prev_wrist = None

    def plan_kick(i: int) -> None:
        if args_cli.perturb_frac > 0.0 and rng.random() < args_cli.perturb_frac:
            kick_at[i] = rng.randint(*args_cli.perturb_window)
        else:
            kick_at[i] = -1
        kicked[i] = False

    def kick(i: int) -> None:
        robot = env.scene[names["robot"]]
        q = robot.data.joint_pos.clone()
        v = torch.zeros_like(robot.data.joint_vel)
        delta = (torch.rand(6, device=q.device) * 2 - 1) * args_cli.perturb_mag
        q[i, :6] += delta
        lim = getattr(robot.data, "soft_joint_pos_limits", None)
        if lim is not None and lim.shape[-1] == 2:
            q[i] = torch.clamp(q[i], lim[i, :, 0], lim[i, :, 1])
        robot.write_joint_state_to_sim(q, v)

    def language_for(i: int) -> str | None:
        drawn = getattr(env, "bdash_appearance", None)
        tgt = int(env.bdash_target_idx[i])
        entry = drawn[i][tgt] if drawn and drawn[i] else None
        return bcl.instruction_for(entry, action="lift")

    with torch.inference_mode():
        # WAVE SEMANTICS. The first orchestration used continuous per-env resets and lost the
        # teacher 95% -> 24% to a family of partial-reset interactions (stale-latch sabotage,
        # policy-under-settle watchdog hits, and two more found the same afternoon). Waves keep
        # EVERY reset on the battle-tested full-reset path of the single-env recorder: all envs
        # start together, finished envs freeze holding their part, and the wave ends with one
        # env.reset() for everyone. Throughput loss vs continuous is the wave straggler (~20%),
        # which is a price worth paying for using zero untested code paths.
        while recorded < args_cli.num_demos and simulation_app.is_running():
            env.reset()
            for _ in range(args_cli.settle_steps):
                env.step(torch.zeros(n, env.action_space.shape[-1], device=dev))
            env.recorder_manager.reset()
            policy.reset()
            for i in range(n):
                plan_kick(i)
            done = torch.zeros(n, dtype=torch.bool, device=dev)
            exported_wave = torch.zeros(n, dtype=torch.bool, device=dev)
            frozen[:] = 0
            prev_wrist = None
            hold_action = torch.zeros(n, env.action_space.shape[-1], device=dev)
            hold_action[:, -1] = -1.0  # finished envs keep their grip closed while waiting
            for step in range(args_cli.max_steps_per_demo):
                action = policy.get_action(env, None)
                action = torch.where(done.unsqueeze(-1), hold_action, action)
                env.step(action)
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)
                env.obs_buf = env.observation_manager.compute()

                cam = (env.obs_buf.get("camera_obs") or {}).get("wrist_cam_rgb")
                if cam is not None:
                    cur = cam.detach().float().flatten(1)
                    if prev_wrist is not None and cur.shape == prev_wrist.shape:
                        frozen += ((cur - prev_wrist).abs().mean(dim=1) < 1e-6).long() * (~done).long()
                    prev_wrist = cur

                for i in range(n):
                    if not done[i] and not kicked[i] and kick_at[i] >= 0 and step >= kick_at[i]:
                        ph = policy._pick.phase if getattr(policy, "_pick", None) is not None else None
                        if ph is not None and int(ph[i]) <= 1:
                            kick(i)
                            kicked[i] = True

                lifted = vp.lifted(env, **lift_params) & ~done
                for i in torch.nonzero(lifted, as_tuple=False).flatten().tolist():
                    if recorded < args_cli.num_demos and int(frozen[i]) == 0:
                        env.recorder_manager.record_pre_reset([i], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [i], torch.tensor([[True]], dtype=torch.bool, device=dev)
                        )
                        env.recorder_manager.export_episodes([i])
                        exported_wave[i] = True
                        recorded += 1
                    done[i] = True
                if bool(done.all()):
                    break
            pk = getattr(policy, "_pick", None)
            for i in range(n):
                attempts += 1
                jsonl.write(
                    json.dumps(
                        dict(
                            attempt=attempts,
                            env_slot=i,
                            exported=bool(exported_wave[i]),
                            steps=int(step + 1),
                            kicked=bool(kicked[i]),
                            frozen_steps=int(frozen[i]),
                            language=language_for(i) if exported_wave[i] else None,
                            variant=("W-A", "W-B", "W-C")[int(env.bdash_target_variant[i])],
                            side_lying=bool(env.bdash_side_lying[i, int(env.bdash_target_idx[i])]),
                            pick_attempts=int(pk.attempts[i]) if pk is not None else None,
                            pick_empty_close=bool(pk.empty_close[i]) if pk is not None else None,
                            pick_gave_up=bool(pk.gave_up[i]) if pk is not None else None,
                            pick_phase_end=int(pk.phase[i]) if pk is not None else None,
                        )
                    )
                    + "\n"
                )
            jsonl.flush()
            print(f"[vrec] {recorded}/{args_cli.num_demos} demos ({attempts} attempts)", flush=True)

    jsonl.close()
    print(f"[vrec] DONE: {recorded} demos in {attempts} attempts -> {args_cli.dataset_file}", flush=True)
    # HARD EXIT. simulation_app.close() hangs on this box (measured: the clean1 slice sat 2h+ in
    # shutdown, starving the campaign chain). Every episode was already flushed by
    # export_episodes(), so skipping the graceful close loses nothing -- os._exit bypasses the
    # finally-block close on purpose.
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
