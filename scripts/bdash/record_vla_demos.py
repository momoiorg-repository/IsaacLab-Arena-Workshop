# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Record VLA pick→handoff demos from the scripted expert (brief §3 M9).

No teleop: the privileged scripted expert (``bdash_scripted``) drives the arm (brief §2.1 permits the
scripted expert for demo generation — it is the VLA's teacher), and each episode is recorded from
reset until the §3.4 ``handoff`` predicate first fires, then cut and exported as a successful demo.
The recorded trajectory (camera RGB + robot state + action per step) is the imitation data for the
VLA front-end (the insertion is the rule-based controller's job and is not recorded). Output is an
Isaac Lab HDF5 dataset; convert to LeRobot with ``convert_franka_to_lerobot.py``.

Needs ``--enable_cameras`` (wrist + external RGB). The non-tiled cameras render one env, so record
single-env (``--num_envs 1``). L1-centric per the brief.

    unset DISPLAY
    /isaac-sim/python.sh scripts/bdash/record_vla_demos.py --enable_cameras --num_envs 1 --seed 0 \\
        --num_demos 200 --dataset_file datasets/bdash/vla_pick_handoff.hdf5 \\
        bdash_pick_insert --clearance 2.0 --level L1

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
parser.add_argument("--policy_type", type=str, default="bdash_scripted")
parser.add_argument(
    "--max_steps_per_demo",
    type=int,
    default=None,
    help="cap per demo (default: 600 for --until handoff, 1200 for --until inserted)",
)
parser.add_argument(
    "--family",
    choices=["peg", "chuck"],
    default="peg",
    help=(
        "which task family: 'peg' (bdash_pick_insert) or 'chuck' (bdash_chuck_load). Selects the "
        "predicate module, the task config and the episode cut point."
    ),
)
parser.add_argument(
    "--until",
    choices=["handoff", "inserted", "lifted", "placed", "loaded"],
    default=None,
    help=(
        "cut point. peg: 'handoff' (default, M9) or 'inserted' (full task). "
        "chuck: 'lifted' (default, grasp+lift only) or 'placed' (grasp -> transport -> stand it on the "
        "仮置き台 correct-end-down, released and settled)."
    ),
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=120,
    help=(
        "hold steps between reset and the start of recording, so any residual settling motion is "
        "NOT exported. 120 (2.0 s), not the 10 this used to be: workpieces ARE placed at their "
        "analytic resting pose, but a side-lying cylinder is in neutral equilibrium about its own "
        "axis and the stepped variants rest tilted on an edge, so they creep for about a second "
        "after the reset. Measured with check_chuck_scene.py: at 10 steps parts were still moving "
        "at up to 0.117 m/s with 17 mm of drift, at 60 one reset in eight still failed, at 120 all "
        "14 resets were at rest. Independent of the 2026-08-16 layout change -- the same drifts "
        "appear with the tray yaw at 0 -- so this was in every demo recorded before today."
    ),
)
parser.add_argument("--language", type=str, default=None, help="task instruction (defaults per --family)")
parser.add_argument(
    "--arm_init_std",
    type=float,
    default=None,
    help=(
        "override the arm start-pose randomization std (rad). Higher (e.g. 0.10-0.20) "
        "= more diverse starts + recovery-from-perturbed-pose demos; default keeps 0.02."
    ),
)
# --- recovery-data injection (covariate-shift fix): kick the arm off-target mid-approach, let the
# GT-driven scripted expert recover, and record the recovery. Off by default -> behavior unchanged. ---
parser.add_argument(
    "--perturb_frac",
    type=float,
    default=0.0,
    help=(
        "probability per episode of a one-shot mid-trajectory joint kick (recovery "
        "data). 0 = off (default; identical to prior recording)."
    ),
)
parser.add_argument(
    "--perturb_mag",
    type=float,
    default=0.15,
    help=(
        "max |delta| (rad) of the kick, applied to arm joints 1-6 (wrist + fingers "
        "left alone so the gripper stays roughly down and the expert re-targets in xy/z)."
    ),
)
parser.add_argument(
    "--perturb_window",
    type=int,
    nargs=2,
    default=(3, 50),
    metavar=("MIN", "MAX"),
    help=(
        "sample the kick step from this range; it fires at the first step >= MIN that "
        "is still pre-grasp (pick phase <= DESCEND), so recovery covers the approach."
    ),
)
parser.add_argument(
    "--jsonl",
    type=str,
    default=None,
    help=(
        "per-attempt sidecar log (chuck: target/variant/pose-class/steps). Defaults next to the HDF5. "
        "Needed because env_cfg.recorders is replaced wholesale below, which drops the task's own "
        "metric recorder terms -- without this there is no per-demo metadata to slice GATE 1 by."
    ),
)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

_DEFAULTS = {
    "peg": ("handoff", "Pick up the peg and move it over the socket."),
    "chuck": ("lifted", "Pick up a workpiece from the tray and lift it clear."),
}
#: Cut points whose instruction is chosen PER EPISODE from the target's drawn material, so the
#: dataset-level `--language` is a placeholder rather than the label. The per-episode sentence is in
#: the attempts sidecar under `language`, and the LeRobot conversion reads it from there.
_PER_EPISODE_LANGUAGE = {"placed", "loaded"}
if args_cli.until is None:
    args_cli.until = _DEFAULTS[args_cli.family][0]
if args_cli.language is None:
    args_cli.language = _DEFAULTS[args_cli.family][1]
if args_cli.until in _PER_EPISODE_LANGUAGE and "--language" not in sys.argv:
    # Say so loudly rather than writing the lifted-stage sentence onto pick-place demos. That is
    # what the first smoke run did: every episode carried "lift it clear" while the demo actually
    # ran to a part standing on the pad, and the conversion would have baked it in.
    args_cli.language = "PER_EPISODE"
if (args_cli.until in ("lifted", "placed", "loaded")) != (args_cli.family == "chuck"):
    parser.error(f"--until {args_cli.until!r} is not available for --family {args_cli.family!r}")

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

from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena_environments.mdp import bdash_chuck_language as bcl_top


class CameraObsRecorder(RecorderTerm):
    """Record the ``camera_obs`` observation group (wrist + external RGB) each step under ``obs`` —
    the default ActionState recorder only stores the ``policy`` (state) group, but the VLA needs the
    images. Stored as obs/{wrist_cam_rgb,left_cam_rgb,right_cam_rgb} (convert_franka_to_lerobot keys)."""

    def record_pre_step(self):
        cam = self._env.obs_buf.get("camera_obs")
        return ("obs", cam) if cam is not None else (None, None)


@configclass
class CameraObsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CameraObsRecorder


class ConditioningRecorder(RecorderTerm):
    """Record the episode's task id and the target workpiece's axis, under ``obs``.

    Both are **inert to the current converter**, which only reads the keys its YAML names -- and that
    is the point. Which channel the VLA is actually conditioned on (images alone, per-episode
    language, or an explicit state input) is a modelling question that only binds at *conversion*
    time, but it can only be answered from data that was recorded. Writing them now costs nothing
    and means the decision never forces a re-record.

    ``axis_cond`` is ``[cos 2phi, sin 2phi, side_lying, z_top, valid]``. The doubled angle is
    deliberate: a cylinder's direction lives on RP^1, and ``(cos 2phi, sin 2phi)`` is its continuous
    single-valued embedding, so the conditioning channel has no branch cut even though the wrist
    command necessarily does.
    """

    def record_pre_step(self):
        from isaaclab_arena_environments.mdp import bdash_chuck_language as bcl

        env = self._env
        target = getattr(env, "bdash_target_idx", None)
        if target is None:
            return (None, None)
        names = env.cfg.isaaclab_arena_env.task.workpiece_names
        rows = torch.arange(env.num_envs, device=target.device)
        quat = torch.stack([env.scene[n].data.root_quat_w for n in names], dim=1)[rows, target]

        local_z = torch.zeros(env.num_envs, 3, device=quat.device, dtype=quat.dtype)
        local_z[:, 2] = 1.0
        axis = quat_apply(quat, local_z)
        phi = torch.atan2(axis[:, 1], axis[:, 0])
        side = (axis[:, 2].abs() < 0.5).to(quat.dtype)
        z_top = torch.stack([env.scene[n].data.root_pos_w for n in names], dim=1)[rows, target][:, 2]
        cond = torch.stack([torch.cos(2 * phi), torch.sin(2 * phi), side, z_top, torch.ones_like(side)], dim=-1)
        # spec §9: which layer wrote this step's ee_pose_action (bdash_chuck_policy.ACTION_OWNERS).
        # Constant "pick" on the pick stage, which is exactly the claim worth having in the dataset:
        # nothing else touched the action the model is being trained to imitate. -1 = not yet set
        # (the settle steps, which are driven by zero actions and are not exported anyway).
        owner = getattr(env, "bdash_action_owner", None)
        owner = (
            torch.full((env.num_envs, 1), -1, dtype=torch.int32, device=quat.device)
            if owner is None
            else owner.reshape(-1, 1).to(torch.int32)
        )
        # The TARGET's drawn appearance, as [r, g, b, roughness, metallic]. Per step so it lines up
        # with every other obs key; constant within an episode by construction.
        drawn = getattr(env, "bdash_appearance", None)
        if drawn and drawn[0] is not None:
            rows = []
            for env_index in range(env.num_envs):
                entry = drawn[env_index][int(target[env_index])]
                if "rgb" in entry:
                    rows.append(
                        [*[float(c) for c in entry["rgb"]], float(entry["roughness"]), float(entry["metallic"])]
                    )
                else:
                    # MDL draw. There is no scalar colour to embed -- which is why this used to
                    # write five zeros -- so carry the material IDENTITY instead: index, family,
                    # and three zeros to keep the channel width fixed. The identity is what the
                    # per-episode instruction names, so it has to be in the data, and an index says
                    # exactly which material where an RGB triple only approximates it.
                    mat_i, fam_i = bcl.identity_of(entry)
                    rows.append([float(mat_i), float(fam_i), 0.0, 0.0, 0.0])
            appearance = torch.tensor(rows, dtype=cond.dtype, device=cond.device)
        else:
            appearance = torch.zeros(env.num_envs, 5, dtype=cond.dtype, device=cond.device)
        return (
            "obs",
            {
                "axis_cond": cond,
                "task_id": target.reshape(-1, 1).to(torch.int32),
                "action_owner": owner,
                "appearance": appearance,
            },
        )


@configclass
class ConditioningRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ConditioningRecorder


@configclass
class VLARecorderManagerCfg(ActionStateRecorderManagerCfg):
    """ActionState recorder + the camera-obs term (so demos carry RGB for the VLA)."""

    record_camera_obs: CameraObsRecorderCfg = CameraObsRecorderCfg()
    record_conditioning: ConditioningRecorderCfg = ConditioningRecorderCfg()


from isaaclab.utils.math import quat_apply

from isaaclab_arena.controllers.scripted_pick import DESCEND
from isaaclab_arena.utils.random import set_seed

if args_cli.family == "chuck":
    from isaaclab_arena_environments.mdp import bdash_chuck_predicates as vp
    from isaaclab_arena_environments.mdp.bdash_chuck_config import load_task_cfg
else:
    from isaaclab_arena_environments.mdp import bdash_peg_predicates as vp
    from isaaclab_arena_environments.mdp.bdash_peg_config import load_task_cfg


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
    if os.environ.get("BDASH_DBG_TERMS"):
        fields = {k: type(getattr(env_cfg.terminations, k)).__name__ for k in vars(env_cfg.terminations)}
        print(f"[dbg] terminations fields: {fields}", flush=True)
        print(f"[dbg] success repr: {getattr(env_cfg.terminations, 'success', '<absent>')!r}"[:400], flush=True)
    insert_params = dict(env_cfg.terminations.success.params) if env_cfg.terminations.success else {}
    env_cfg.terminations.time_out = None
    env_cfg.terminations.success = None
    if hasattr(env_cfg.terminations, "insertion_failed"):
        env_cfg.terminations.insertion_failed = None
    if hasattr(env_cfg.terminations, "object_dropped"):
        env_cfg.terminations.object_dropped = None
    # The load stage's force abort. Dropped for the same reason as the others: the recorder owns the
    # episode boundary, and an auto-reset in the middle of one truncates the demo it is writing. It
    # was not in this list because the load stage did not exist when the list was written.
    if hasattr(env_cfg.terminations, "load_failed"):
        env_cfg.terminations.load_failed = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = VLARecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = out_dir
    env_cfg.recorders.dataset_filename = fname
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if args_cli.seed is not None:
        set_seed(args_cli.seed, env)

    policy = get_policy_cls(args_cli.policy_type).from_args(args_cli)

    t = load_task_cfg()
    task = env.cfg.isaaclab_arena_env.task
    names = dict(t["scene_names"])

    # cut point: the predicate whose first firing ends the recorded demo.
    if args_cli.family == "chuck":
        # chuck: 'lifted' — the VLA owns grasp+lift only; everything after is classic control.
        grasped = t["grasped"]
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
        if args_cli.until == "loaded":
            # THE WHOLE TASK: grasp, transport, insert, and the chuck taking the part. The cut is
            # the env's own `loaded` success termination, evaluated with the env's own params --
            # captured above before the recorder nulls the terminations, exactly as the peg path
            # captures its `inserted` params.
            #
            # NOT rebuilt by hand, and that is a lesson, not a preference. The first version of this
            # block reconstructed the params from `task.profiles`, which does not carry
            # `force_seated` -- the task derives that from the chuck's stopper geometry when it
            # builds the termination -- so every variant came out False. W-C physically cannot reach
            # its nominal 72 mm by depth (the fingers stop at 65.1 mm); it can only seat by FORCE,
            # so the hand-built cut could never fire for it: measured, W-C 0/4 all at the step cap,
            # while the checker -- driving the same teacher through the env's own terminal -- passes
            # it at ~78%. Two spellings of one predicate had quietly diverged.
            # The cut is the env's OWN success -- `chuck_closed`, no parameters -- because that is
            # what the full stage's success actually is (spec §2.3: the commit is the jaw close;
            # `loaded` was demoted to the handover cue). Two previous spellings of this cut both
            # diverged from the env: a hand-built `loaded` param set missed the stopper-derived
            # `force_seated` (W-C 0/4, step-cap timeouts), and reusing `terminations.success.params`
            # captured chuck_closed's EMPTY dict and asserted out. Zero parameters, zero divergence.
            cut = lambda: bool(vp.chuck_closed(env)[0].item())  # noqa: E731
            default_max_steps = 1600
        elif args_cli.until == "placed":
            # PICK-PLACE demo (2026-08-20 ruling): the VLA's slice now runs to the part standing on
            # the 仮置き台 with the correct end down, not just clear of the tray.
            #
            # The cut is the TEACHER'S OWN FSM reaching DONE -- released, settled and withdrawn --
            # and not a predicate over the workpiece's pose. Two reasons. It is proprioceptive, so
            # the episode boundary does not depend on anything the policy will not have at
            # inference; and it is the same event in every episode, where a pose predicate would cut
            # early on a part that happened to settle fast and late on one that rocked, putting a
            # variable amount of "nothing happening" at the end of the demos.
            #
            # Place QUALITY is measured separately (CP1b) and is not a cut condition: a demo that
            # ends with the part fallen over is a demo of falling over, and it is filtered by the
            # quality bar rather than by never being cut.
            def cut():
                stats = policy.reerect_stats() if hasattr(policy, "reerect_stats") else {}
                return stats.get("reerect_phase") == "done" and not stats.get("reerect_gave_up")

            default_max_steps = 2200
        else:
            cut = lambda: bool(vp.lifted(env, **lift_params)[0].item())  # noqa: E731
            default_max_steps = 500
    else:
        # §3.4 handoff predicate params (from task.yaml, via the loaded task)
        names = {**names, "peg": task.names["peg"], "socket": task.names["socket"]}
        geom = t["geometry"]
        hp = dict(
            peg_name=names["peg"],
            socket_name=names["socket"],
            robot_name=names["robot"],
            peg_finger_sensor=names["peg_finger_sensor"],
            tip_offset=tuple(geom["tip_offset"]),
            mouth_height=float(geom["mouth_height"]),
            **t["handoff"],
            **t["grasped"],
        )
        if args_cli.until == "inserted":
            if not insert_params:
                raise RuntimeError("--until inserted needs the env's success (inserted) termination; none found")
            cut = lambda: bool(vp.inserted(env, **insert_params)[0].item())  # noqa: E731
            default_max_steps = 1200
        else:
            cut = lambda: bool(vp.handoff(env, **hp)[0].item())  # noqa: E731
            default_max_steps = 600
    max_steps = args_cli.max_steps_per_demo or default_max_steps

    jsonl_path = args_cli.jsonl or os.path.join(out_dir, f"{fname}_attempts.jsonl")
    jsonl = open(jsonl_path, "w")  # noqa: SIM115 — closed at the end of main()

    print(
        f"[rec] environment ready — recording up to {args_cli.num_demos} demos "
        f"until '{args_cli.until}' (<= {max_steps} steps each, first one in ~10s)...",
        flush=True,
    )
    recorded = 0
    attempts = 0
    perturbed_ok = 0  # successful demos that carried a recovery kick
    robot_name = names["robot"]
    while recorded < args_cli.num_demos and simulation_app.is_running():
        # whole iteration under inference_mode: policy/recorder state tensors are inference tensors,
        # so their in-place resets must not happen outside an inference_mode context.
        with torch.inference_mode():
            env.reset()
            # Settle BEFORE the recorder is reset, so any residual motion is not part of the demo.
            # Isaac Lab computes observations straight after the reset events with no physics in
            # between, so without this the first exported frames could show the scene still moving.
            for _ in range(args_cli.settle_steps):
                env.step(torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device))
            env.recorder_manager.reset()
            policy.reset()
            attempts += 1
            # Reset per attempt: a MISS never reaches the block that fills these in, and a stale
            # value from the previous attempt would label the next demo with the wrong material.
            episode_language = episode_material = None
            reached = False
            # decide this episode's recovery kick: fire once at the first pre-grasp step >= kick_step.
            do_kick = args_cli.perturb_frac > 0.0 and random.random() < args_cli.perturb_frac
            kick_step = random.randint(*args_cli.perturb_window) if do_kick else -1
            kicked = False
            # FRAME-FREEZE DETECTION, always on and per camera. This used to be a BDASH_REC_DBG
            # printout every 15 steps, which is not something 250 episodes can be inspected with.
            # The failure it exists to catch is silent and total -- a standalone `sim.render()` does
            # not refresh the RTX sensors on this box, so every recorded frame was the frozen reset
            # render and the model trained blind. `frozen` counts steps whose frame is BYTE-IDENTICAL
            # to the previous one, which is what that failure looks like; `mean` separates "the arm
            # happened to be still" from "the sensor is dead".
            prevcam: dict = {}
            diff_min: dict = {}
            diff_sum: dict = {}
            diff_frozen: dict = {}
            diff_n = 0
            for step in range(max_steps):
                if (
                    do_kick
                    and not kicked
                    and step >= kick_step
                    and policy._pick is not None
                    and int(policy._pick.phase[0]) <= DESCEND
                ):
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
                # Accumulated on the GPU and synced once per episode, so the always-on check costs
                # no per-step device round-trip.
                for key, value in (env.obs_buf.get("camera_obs") or {}).items():
                    cur = value[0].detach().float()
                    prev = prevcam.get(key)
                    if prev is not None:
                        d = (cur - prev).abs().mean()
                        diff_min[key] = d if key not in diff_min else torch.minimum(diff_min[key], d)
                        diff_sum[key] = d if key not in diff_sum else diff_sum[key] + d
                        zero = (d == 0).float()
                        diff_frozen[key] = zero if key not in diff_frozen else diff_frozen[key] + zero
                    prevcam[key] = cur
                if prevcam:
                    diff_n += 1
                if os.environ.get("BDASH_REC_DBG") and step % 15 == 1 and "wrist_cam_rgb" in diff_min:
                    print(
                        f"[recdbg] step={step} wrist framediff min={float(diff_min['wrist_cam_rgb']):.4f}", flush=True
                    )
                if cut():
                    reached = True
                    break
            if reached:
                # THE INSTRUCTION IS PER EPISODE, chosen from the material the target was actually
                # drawn with. A single constant sentence over a randomly chosen target is what this
                # replaces, and it made the task unlearnable rather than merely imprecise: the same
                # words asked for a different part every time.
                drawn_all = getattr(env, "bdash_appearance", None)
                tgt = int(getattr(env, "bdash_target_idx")[0]) if hasattr(env, "bdash_target_idx") else 0
                entry = drawn_all[0][tgt] if drawn_all and drawn_all[0] else None
                episode_language = bcl_top.instruction_for(
                    entry, action="load" if args_cli.until == "loaded" else "place"
                )
                episode_material = (entry or {}).get("name")
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
            # Per-attempt sidecar. GATE 1 has to be readable as a SPLIT (side-lying vs upright, and
            # by variant), not just an aggregate rate -- an overall 55% could be a healthy upright
            # slice with a dead side-lying one, and those imply opposite decisions.
            record = {
                "attempt": attempts,
                "exported": bool(reached),
                "steps": step + 1,
                "kicked": bool(kicked),
                # One sync per episode, not per step.
                "cam_frozen_steps": {k: int(v) for k, v in diff_frozen.items()},
                "cam_diff_mean": {k: round(float(v) / max(1, diff_n - 1), 6) for k, v in diff_sum.items()},
            }
            if args_cli.family == "chuck":
                target = int(env.bdash_target_idx[0].item())
                record.update(
                    target=target,
                    variant=task.profiles[target]["variant"],
                    side_lying=bool(env.bdash_side_lying[0, target].item()),
                    # The sentence this demo was actually labelled with, and the material it names.
                    # Recorded per attempt because the instruction now VARIES: checking that the
                    # sentence matches the part is otherwise impossible after the fact.
                    language=episode_language if reached else None,
                    target_material=episode_material if reached else None,
                )
                # WHAT THE DEMO LOOKED LIKE. Without this the dataset records only that appearance
                # was "randomised", and no failure can ever be attributed to a colour, a roughness
                # or a low-contrast draw -- nor can the sampled distribution be checked against the
                # configured one after the fact.
                drawn = (getattr(env, "bdash_appearance", None) or [None])[0]
                if drawn:
                    # An MDL draw has no rgb/roughness/metallic of its own -- the material name IS
                    # the record, and it is the thing a failure would have to be attributed to.
                    record["appearance"] = [
                        {
                            **({"name": a["name"]} if "name" in a else {}),
                            **({"family": a["family"]} if "family" in a else {}),
                            **({"mdl": a["mdl"]} if "mdl" in a else {}),
                            **(
                                {
                                    "rgb": [round(float(c), 4) for c in a["rgb"]],
                                    "roughness": round(float(a["roughness"]), 4),
                                    "metallic": round(float(a["metallic"]), 4),
                                }
                                if "rgb" in a
                                else {}
                            ),
                        }
                        for a in drawn
                    ]
                    record["target_appearance"] = record["appearance"][target]
            jsonl.write(json.dumps(record) + "\n")
            jsonl.flush()
            print(
                f"[rec] {tag} success={recorded}/{args_cli.num_demos}  attempts={attempts}  "
                f"rate={rate:.0%}  (this={args_cli.until if reached else 'miss'}{kick_tag})",
                flush=True,
            )

    rate = recorded / attempts if attempts else 0.0
    kick_summary = f" [{perturbed_ok} with recovery kick]" if args_cli.perturb_frac > 0.0 else ""
    print(
        f"[rec] DONE: {recorded} successful demos in {attempts} attempts (rate {rate:.0%}){kick_summary} "
        f"-> {args_cli.dataset_file}  (language: '{args_cli.language}')",
        flush=True,
    )
    print(f"[rec] per-attempt log -> {jsonl_path}", flush=True)
    jsonl.close()
    print("RECORD_VLA_DONE", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # PRINT IT. Isaac Sim installs its own excepthook and swallows the traceback otherwise: the
        # run just stops and `close()` below then hangs in the renderer, which from outside is
        # indistinguishable from a hang. Measured twice now -- once in the checker, once here, each
        # costing a py-spy session to even locate. The checker got this guard first; the recorder
        # kept the trap.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()  # always close, so a crash never leaves the app (and HDF5 lock) hung
