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
parser.add_argument("--max_steps_per_demo", type=int, default=600)
parser.add_argument("--language", type=str, default="Pick up the peg and move it over the socket.")
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

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
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.mdp import vdash_predicates as vp
from isaaclab_arena_environments.mdp.vdash_config import load_task_cfg


def main():
    out_dir = os.path.dirname(args_cli.dataset_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    # run under our control until handoff: drop all auto-terminations, record action+state to HDF5
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

    print(f"[rec] environment ready — recording up to {args_cli.num_demos} demos "
          f"(first one in ~10s)...", flush=True)
    recorded = 0
    attempts = 0
    while recorded < args_cli.num_demos and simulation_app.is_running():
        # whole iteration under inference_mode: policy/recorder state tensors are inference tensors,
        # so their in-place resets must not happen outside an inference_mode context.
        with torch.inference_mode():
            env.reset()
            env.recorder_manager.reset()
            policy.reset()
            attempts += 1
            reached = False
            for _ in range(args_cli.max_steps_per_demo):
                action = policy.get_action(env, None)
                env.step(action)
                if bool(vp.handoff(env, **hp)[0].item()):
                    reached = True
                    break
            if reached:
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
            print(f"[rec] {tag} success={recorded}/{args_cli.num_demos}  attempts={attempts}  "
                  f"rate={rate:.0%}  (this={'handoff' if reached else 'no-handoff'})", flush=True)

    rate = recorded / attempts if attempts else 0.0
    print(f"[rec] DONE: {recorded} successful demos in {attempts} attempts (rate {rate:.0%}) "
          f"-> {args_cli.dataset_file}  (language: '{args_cli.language}')", flush=True)
    print("RECORD_VLA_DONE", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()  # always close, so a crash never leaves the app (and HDF5 lock) hung
