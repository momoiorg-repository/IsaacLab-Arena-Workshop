# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Record VLA pick→handoff demos from the scripted expert (brief §3 M9).

No teleop: the privileged scripted expert (``vdash_scripted``) drives the arm (brief §2.1 permits the
scripted expert for demo generation — it is the VLA's teacher), and each episode is recorded from
reset until the §3.4 ``handoff`` predicate first fires, then cut and exported as a successful demo.
The recorded trajectory (camera RGB + robot state + action per step) is the imitation data for the
VLA front-end (the insertion is the rule-based controller's job and is not recorded). Output is an
Isaac Lab HDF5 dataset; convert to LeRobot with ``convert_franka_to_lerobot.py``.

Needs ``--enable_cameras`` (wrist + external RGB). The non-tiled cameras render one env, so record
single-env (``--num_envs 1``). L1-centric per the brief.

    unset DISPLAY
    /isaac-sim/python.sh scripts/vdash/record_vla_demos.py --enable_cameras --num_envs 1 --seed 0 \\
        --num_demos 200 --dataset_file datasets/vdash/vla_pick_handoff.hdf5 \\
        vdash_pick_insert --clearance 2.0 --level L1

Recovery slice (covariate-shift fix): add ``--perturb_frac 1.0`` (optionally ``--perturb_mag``/
``--perturb_window``) to kick the arm off-target mid-approach so the recorded demo includes the
scripted expert recovering toward the peg. Off by default.
"""

import os
import sys

# HDF5 file locking hangs the recorder's dataset creation on the docker bind-mount, and the lock
# setting is read as the HDF5 library loads — too early for an in-process os.environ to take effect
# reliably. So re-exec the script once with the env var in the REAL process environment (guaranteed
# before HDF5 loads). The guard prevents an exec loop; no shell `export` needed.
if os.environ.get("HDF5_USE_FILE_LOCKING", "").upper() != "FALSE":
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# immediate heartbeat (visible even through a `| grep '[rec]'`) so the ~90s Isaac Sim startup
# doesn't look like a dead/blank terminal.
print("[rec] launching Isaac Sim — startup takes ~90s before the first demo...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--dataset_file", type=str, required=True, help="output HDF5 path")
parser.add_argument("--num_demos", type=int, default=50, help="successful demos to record")
parser.add_argument("--policy_type", type=str, default="vdash_scripted")
parser.add_argument("--max_steps_per_demo", type=int, default=None,
                    help="cap per demo (default: 600 for --until handoff, 1200 for --until inserted)")
parser.add_argument("--until", choices=["handoff", "inserted"], default="handoff",
                    help="cut point: 'handoff' (M9, pick->handoff) or 'inserted' (full pick->insert task)")
parser.add_argument("--language", type=str, default="Pick up the peg and move it over the socket.")
parser.add_argument("--arm_init_std", type=float, default=None,
                    help="override the arm start-pose randomization std (rad). Higher (e.g. 0.10-0.20) "
                         "= more diverse starts + recovery-from-perturbed-pose demos; default keeps 0.02.")
# --- recovery-data injection (covariate-shift fix): kick the arm off-target mid-approach, let the
# GT-driven scripted expert recover, and record the recovery. Off by default -> behavior unchanged. ---
parser.add_argument("--perturb_frac", type=float, default=0.0,
                    help="probability per episode of a one-shot mid-trajectory joint kick (recovery "
                         "data). 0 = off (default; identical to prior recording).")
parser.add_argument("--perturb_mag", type=float, default=0.15,
                    help="max |delta| (rad) of the kick, applied to arm joints 1-6 (wrist + fingers "
                         "left alone so the gripper stays roughly down and the expert re-targets in xy/z).")
parser.add_argument("--perturb_window", type=int, nargs=2, default=(3, 50), metavar=("MIN", "MAX"),
                    help="sample the kick step from this range; it fires at the first step >= MIN that "
                         "is still pre-grasp (pick phase <= DESCEND), so recovery covers the approach.")
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random

import gymnasium as gym
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.evaluation.policy_runner import get_policy_cls


class CameraObsRecorder(RecorderTerm):
    """Record the ``camera_obs`` observation group (wrist + external RGB) each step under ``obs`` —
    the default ActionState recorder only stores the ``policy`` (state) group, but the VLA needs the
    images. Stored as obs/{wrist_cam_rgb,left_cam_rgb,right_cam_rgb} (convert_franka_to_lerobot keys)."""

    def record_pre_step(self):
        cam = self._env.obs_buf.get("camera_obs", None)
        return ("obs", cam) if cam is not None else (None, None)


@configclass
class CameraObsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CameraObsRecorder


@configclass
class VLARecorderManagerCfg(ActionStateRecorderManagerCfg):
    """ActionState recorder + the camera-obs term (so demos carry RGB for the VLA)."""

    record_camera_obs: CameraObsRecorderCfg = CameraObsRecorderCfg()
from isaaclab_arena.controllers.scripted_pick import DESCEND
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.mdp import vdash_predicates as vp
from isaaclab_arena_environments.mdp.vdash_config import load_task_cfg


def _apply_kick(env, robot_name: str, mag: float) -> None:
    """One-shot recovery perturbation: teleport the arm off-target by a bounded random joint delta.

    Perturbs arm joints 1-6 (indices 0-5) only — wrist + fingers untouched, so the gripper stays
    roughly down and the displacement is mostly EE position. Velocities zeroed (a teleport). The
    scripted expert's action is a clamped P-controller toward the *absolute* grasp waypoint, so the
    next steps drive back to the peg — that recovery is what gets recorded. joint_pos (the GR00T
    state) updates immediately on write; the EE-frame sensor catches up after the next physics step.
    """
    robot = env.scene[robot_name]
    q = robot.data.joint_pos.clone()
    v = torch.zeros_like(robot.data.joint_vel)
    n_arm = min(6, q.shape[1])
    delta = (torch.rand((q.shape[0], n_arm), device=q.device) * 2 - 1) * mag
    q[:, :n_arm] += delta
    lim = getattr(robot.data, "soft_joint_pos_limits", None)
    if lim is not None and lim.shape[-1] == 2:  # keep the kicked config inside joint limits
        q = torch.clamp(q, lim[..., 0], lim[..., 1])
    robot.write_joint_state_to_sim(q, v)


def main():
    out_dir = os.path.dirname(args_cli.dataset_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    # wider arm start-pose randomization -> diverse approaches + recovery-from-perturbed-start data
    if args_cli.arm_init_std is not None and hasattr(env_cfg.events, "randomize_franka_joint_state"):
        env_cfg.events.randomize_franka_joint_state.params["std"] = args_cli.arm_init_std
        print(f"[rec] arm start-pose std -> {args_cli.arm_init_std} rad", flush=True)

    # run under our control: drop auto-terminations, record action+state to HDF5. Capture the
    # success (inserted) predicate params first — used as the cut point for --until inserted.
    insert_params = dict(env_cfg.terminations.success.params) if env_cfg.terminations.success else {}
    env_cfg.terminations.time_out = None
    env_cfg.terminations.success = None
    if hasattr(env_cfg.terminations, "insertion_failed"):
        env_cfg.terminations.insertion_failed = None
    if hasattr(env_cfg.terminations, "object_dropped"):
        env_cfg.terminations.object_dropped = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = VLARecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = out_dir
    env_cfg.recorders.dataset_filename = fname
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if args_cli.seed is not None:
        set_seed(args_cli.seed, env)

    policy = get_policy_cls(args_cli.policy_type).from_args(args_cli)

    # §3.4 handoff predicate params (from task.yaml, via the loaded task)
    t = load_task_cfg()
    names = t["scene_names"]
    task = env.cfg.isaaclab_arena_env.task
    names = {**names, "peg": task.names["peg"], "socket": task.names["socket"]}
    geom = t["geometry"]
    hp = dict(
        peg_name=names["peg"], socket_name=names["socket"], robot_name=names["robot"],
        peg_finger_sensor=names["peg_finger_sensor"], tip_offset=tuple(geom["tip_offset"]),
        mouth_height=float(geom["mouth_height"]), **t["handoff"], **t["grasped"],
    )

    # cut point: handoff (M9) or full task (inserted = the env's success predicate).
    if args_cli.until == "inserted":
        if not insert_params:
            raise RuntimeError("--until inserted needs the env's success (inserted) termination; none found")
        cut = lambda: bool(vp.inserted(env, **insert_params)[0].item())  # noqa: E731
    else:
        cut = lambda: bool(vp.handoff(env, **hp)[0].item())  # noqa: E731
    max_steps = args_cli.max_steps_per_demo or (1200 if args_cli.until == "inserted" else 600)

    print(f"[rec] environment ready — recording up to {args_cli.num_demos} demos "
          f"until '{args_cli.until}' (<= {max_steps} steps each, first one in ~10s)...", flush=True)
    recorded = 0
    attempts = 0
    perturbed_ok = 0  # successful demos that carried a recovery kick
    robot_name = names["robot"]
    while recorded < args_cli.num_demos and simulation_app.is_running():
        # whole iteration under inference_mode: policy/recorder state tensors are inference tensors,
        # so their in-place resets must not happen outside an inference_mode context.
        with torch.inference_mode():
            env.reset()
            env.recorder_manager.reset()
            policy.reset()
            attempts += 1
            reached = False
            # decide this episode's recovery kick: fire once at the first pre-grasp step >= kick_step.
            do_kick = args_cli.perturb_frac > 0.0 and random.random() < args_cli.perturb_frac
            kick_step = random.randint(*args_cli.perturb_window) if do_kick else -1
            kicked = False
            prevcam = None  # VDASH_REC_DBG: detect whether the live camera obs actually changes step-to-step
            for step in range(max_steps):
                if (do_kick and not kicked and step >= kick_step and policy._pick is not None
                        and int(policy._pick.phase[0]) <= DESCEND):
                    _apply_kick(env, robot_name, args_cli.perturb_mag)
                    kicked = True
                action = policy.get_action(env, None)
                env.step(action)
                # Camera render fix (CRITICAL): on this box a standalone ``sim.render()`` — what the
                # env's decimation loop uses — does NOT refresh the RTX camera sensors, so every
                # recorded frame was the frozen reset render (model trained blind -> mean-regression).
                # Only a render-coupled physics pump refreshes them. Pump one render-step, refresh the
                # sensors, and recompute obs so the frame the recorder captures on the next pre-step
                # reflects the current scene. Verified: without this all frames are byte-identical.
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)
                env.obs_buf = env.observation_manager.compute()
                if os.environ.get("VDASH_REC_DBG"):
                    cam = env.obs_buf["camera_obs"]["wrist_cam_rgb"][0].detach().float()
                    if prevcam is not None and step % 15 == 1:
                        print(f"[recdbg] step={step} wrist obs framediff={(cam - prevcam).abs().mean():.4f}",
                              flush=True)
                    prevcam = cam.clone()
                if cut():
                    reached = True
                    break
            if reached:
                perturbed_ok += int(kicked)
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                recorded = env.recorder_manager.exported_successful_episode_count
                tag = "OK  "
            else:
                tag = "MISS"
            rate = recorded / attempts if attempts else 0.0
            kick_tag = " +kick" if kicked else ""
            print(f"[rec] {tag} success={recorded}/{args_cli.num_demos}  attempts={attempts}  "
                  f"rate={rate:.0%}  (this={args_cli.until if reached else 'miss'}{kick_tag})", flush=True)

    rate = recorded / attempts if attempts else 0.0
    kick_summary = f" [{perturbed_ok} with recovery kick]" if args_cli.perturb_frac > 0.0 else ""
    print(f"[rec] DONE: {recorded} successful demos in {attempts} attempts (rate {rate:.0%}){kick_summary} "
          f"-> {args_cli.dataset_file}  (language: '{args_cli.language}')", flush=True)
    print("RECORD_VLA_DONE", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()  # always close, so a crash never leaves the app (and HDF5 lock) hung
