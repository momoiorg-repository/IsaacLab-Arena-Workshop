# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure the scripted chuck teacher's grasp+lift rate, split by variant and pose class.

THIS IS THE GATE BEFORE ANY DEMO IS RECORDED. ``record_vla_demos.py`` exports successes only, so a
weak teacher does not produce a bad dataset -- it produces a slow one, burning single-GPU
wall-clock on episodes that get thrown away. Run this first and read the split: the side-lying rate
is the number that decides whether the arbitrary-pose scope survives contact with the simulator.

Target: upright >= 90%, side-lying >= 70%.

Run WITHOUT cameras first (no RTX render, several times faster), then once with ``--enable_cameras``
to confirm the render pump changes nothing:

    unset DISPLAY
    /isaac-sim/python.sh scripts/bdash/check_chuck_teacher.py --num_envs 1 --seed 0 \\
        --episodes 30 --max_steps 400 bdash_chuck_load --task pick --variants all
"""

import collections
import json
import math
import os
import pathlib
import sys
import time
import yaml

print("[teachercheck] launching Isaac Sim (~40s)...", flush=True)

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--max_steps", type=int, default=400, help="give up on an episode after this many steps")
parser.add_argument("--policy_type", type=str, default="bdash_chuck_teacher")
parser.add_argument("--jsonl", type=str, default=None, help="write per-episode records here")
parser.add_argument(
    "--no_hand_clearance",
    action="store_true",
    help="spec §2.2/§3(d): measure the slice WITHOUT the §0-10 target hand-clearance guarantee",
)
parser.add_argument(
    "--force_yaw_zero",
    action="store_true",
    help="ABLATION ONLY: hold the §0-11 wrist yaw at zero. Never use for recording.",
)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_arena.controllers.ee_control import read_ee_pose
from isaaclab_arena.utils.random import set_seed


def _predicate_kwargs(task):
    """The same params the task injects into its terminations, so the harness scores identically."""
    grasped = task.cfg["grasped"]
    shared = {
        "robot_name": task.names["robot"],
        "finger_sensors": task.finger_sensor_names,
        "width_min": float(grasped["width_min"]),
        "width_max": float(grasped["width_max"]),
        "grasp_force": float(grasped["grasp_force"]),
    }
    lifted = dict(shared)
    lifted.update(
        workpiece_names=task.workpiece_names,
        tray_name=task.tray.name,
        rim_height=float(task.geom["tray_rim_height"]),
        height=float(task.cfg["lifted"]["height"]),
        speed_max=float(task.cfg["lifted"]["speed_max"]),
        tip_offset=tuple(task.geom["tip_offset"]),
    )
    return shared, lifted


def _scene_fingerprint() -> dict:
    """Identity of the SCENE and CONFIG this run measured, stamped into every record.

    §9: a measurement may only be compared against another taken on the same scene and the same
    config. This is not a style rule -- it cost a wrong root cause. `joint_margin_min` read
    0.147-0.222 after the re-erect and 0.684-0.790 in the baseline, zero overlap, and the obvious
    reading was that the manoeuvre was winding the arm up. The baseline was recorded before the
    robot's start pose was lowered on 2026-08-18, which moved joint 4 to 0.194 rad from its limit
    BY DESIGN. Two different robots, not two different manoeuvres; the control on the current scene
    reads 0.138-0.308 and still succeeds.

    A hash makes that mistake impossible to make silently: two runs with different fingerprints are
    not comparable and the number says so without anyone having to remember why.
    """
    import hashlib

    cfg_dir = pathlib.Path(__file__).resolve().parents[2] / "configs/bdash/chuck_load"
    digest = hashlib.sha256()
    files = sorted(cfg_dir.glob("*.yaml"))
    for f in files:
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    scene = yaml.safe_load((cfg_dir / "task.yaml").read_text())["scene"]
    return {
        "config_sha": digest.hexdigest()[:12],
        "config_files": [f.name for f in files],
        # The handful of numbers that decide whether two runs are even the same experiment.
        "initial_joint_pose": scene.get("initial_joint_pose"),
        "chuck_pose": scene.get("chuck_pose"),
        "tray_pose": scene.get("tray_pose"),
        "reerect_pose": scene.get("reerect_pose"),
    }


def main():
    # Deferred, not module-level: importing isaaclab_arena_environments.mdp (and the policy
    # registry, which pulls it in) before the environment is built deadlocks Isaac Sim inside
    # gym.make -- it hangs at 100% CPU with no traceback, only repeated `zenity: not found` from
    # the crash handler. check_chuck_scene.py keeps the same imports inside main() and is fine.
    fingerprint = _scene_fingerprint()
    print(f"[teachercheck] scene/config sha={fingerprint['config_sha']}  (§9: compare only like with like)", flush=True)
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    if args_cli.no_hand_clearance:
        # spec §2.2 / §3(d): the evaluation slice reported WITHOUT the §0-10 guarantee. Flipped here
        # rather than by editing task.yaml so the two slices are one CLI flag apart and the file
        # keeps the production value -- a run that has to edit config to be reproduced is not one.
        env_cfg.events.select_target.params["enforce_hand_clearance"] = False
        print("[teachercheck] §0-10 hand-clearance guarantee DISABLED for this run", flush=True)

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if args_cli.seed is not None:
        set_seed(args_cli.seed, env)

    # Imported only AFTER the environment exists. Pulling in the policy registry (or
    # isaaclab_arena_environments.mdp) beforehand deadlocks Isaac Sim inside gym.make: it spins at
    # 100% CPU, never reaches "Completed setting up the environment", and emits no traceback --
    # only repeated `zenity: not found` as the crash handler tries to raise a dialog.
    from isaaclab_arena.evaluation.policy_runner import get_policy_cls
    from isaaclab_arena_environments.mdp import bdash_chuck_predicates as cp
    from isaaclab_arena_environments.mdp import bdash_chuck_randomization as bcr

    policy = get_policy_cls(args_cli.policy_type).from_args(args_cli)
    if args_cli.force_yaw_zero:
        policy.force_yaw_zero = True
        print("[teachercheck] §0-11 wrist yaw HELD AT ZERO (ablation)", flush=True)

    task = env.cfg.isaaclab_arena_env.task
    grasp_kw, lift_kw = _predicate_kwargs(task)
    drop_z = float(task.background_scene.object_min_z)
    records = []
    # Opened BEFORE the loop and flushed per episode, not written once at the end. A GATE run is
    # 300+ episodes at ~9 s each, i.e. ~45 min; batching the write meant any wall-clock kill (the
    # `timeout` wrapper, an OOM, a stray Ctrl-C) threw away every completed episode. A killed run
    # now still leaves a usable prefix, which is enough to re-plan the episode budget.
    jsonl = None
    if args_cli.jsonl:
        os.makedirs(os.path.dirname(args_cli.jsonl) or ".", exist_ok=True)
        # noqa SIM115: the handle has to outlive this scope -- it is written once per episode across
        # the whole run and closed below. Wrapping the 60-line measurement loop in a `with` to
        # satisfy the check would bury the thing this script exists to do inside an I/O block.
        jsonl = open(args_cli.jsonl, "w")  # noqa: SIM115
    print(f"[teachercheck] env ready; policy={args_cli.policy_type}; running {args_cli.episodes} episodes", flush=True)

    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            env.reset()
            policy.reset()
            target = int(env.bdash_target_idx[0].item())
            variant = task.profiles[target]["variant"]
            side = bool(env.bdash_side_lying[0, target].item())

            # E1 (S2): per-episode injected axial grip error. BDASH_E1_ERRORS is a comma list of
            # mm values cycled by episode index, so P1/P2/P3 see the IDENTICAL error sequence --
            # the comparison is between policies, and the errors must not be a confound.
            injected_e = None
            injected_tilt = None
            _errs = os.environ.get("BDASH_E1_ERRORS", "")
            if _errs:
                injected_e = float(_errs.split(",")[episode % len(_errs.split(","))])
                os.environ["BDASH_GRASP_OFF_AXIAL_MM"] = str(injected_e)
            # Same mechanism for the R(g) sweep's second axis: an approach-tilt list, cycled by
            # episode, through the existing BDASH_GRASP_OFF_DEG test-bed knob in ScriptedPick.
            _tilts = os.environ.get("BDASH_E1_TILTS", "")
            if _tilts:
                injected_tilt = float(_tilts.split(",")[episode % len(_tilts.split(","))])
                os.environ["BDASH_GRASP_OFF_DEG"] = str(injected_tilt)
            grasped_step = lifted_step = None
            _t0 = time.monotonic()
            fired = None
            abort_snap = None
            pre_snap = None
            # Captured before any control runs, so displacement at the stall is measured against the
            # pose the sampler certified rather than against wherever the part has since been pushed.
            spawn_pos = torch.stack([env.scene[n].data.root_pos_w[0].clone() for n in task.workpiece_names])
            # WHEN the target goes over is the question four refuted hypotheses all failed to
            # answer: the stall snapshot is taken 60 no-progress steps in, by which time the part is
            # already down, so it cannot separate "the arm knocked it over" from "it was never
            # standing". `spawn_axis_z` is the pose the sampler actually produced (1.0 = upright)
            # and `tip_step` is the first step it drops out of upright, so the two are separable:
            # tip_step near 0 means the settle did it and the episode was mislabelled from the
            # start; tip_step during the descent means the arm did it.
            quality = {}
            spawn_axis_z = _target_axis_z(env, task, target, bcr)
            tip_step = None
            stall_snap = {}
            # spec §9: which layer wrote the action, counted per step. A per-step trace would be
            # 900 entries an episode; the count is what makes the failure mode legible -- "creep
            # owned 0 steps" or "hold owned 400" says which layer won without reading a trace.
            owner_steps: collections.Counter = collections.Counter()
            # Sampled every 10 steps rather than every step: a part that has left the tray does not
            # come back, so the coarse cadence loses nothing and saves ~9 GPU syncs per step.
            escape = {"escaped": {}, "escape_margin_mm": -999.0}
            for step in range(args_cli.max_steps):
                # `terminated` is the ONLY safe success signal here. ManagerBasedRLEnv auto-resets
                # inside step() the moment a termination fires, and `success` IS the lift predicate,
                # so re-evaluating cp.lifted after the step reads the NEXT episode and never sees it
                # -- which looks exactly like the part being lifted and then dropped. Nulling the
                # terminations instead is not an option: the task's SuccessRateMetric keys on a term
                # named `success` (record_vla_demos.py only gets away with it because it also
                # replaces the recorder manager wholesale).
                dropped_before = bool(cp.dropped(env, workpiece_names=task.workpiece_names, min_z=drop_z)[0])
                # Evaluated BEFORE the step and kept, because step() auto-resets on termination:
                # reading after it samples the next episode's spawn. Guarding on "not terminated"
                # instead -- which is what this did first -- silently drops the quality figures for
                # exactly the episodes that succeeded, leaving only failures in the log.
                quality = _load_quality(env, task, target, cp)
                if step % 10 == 0:
                    now = _tray_escape(env, task, spawn_pos, target)
                    if now["escape_margin_mm"] > escape["escape_margin_mm"]:
                        escape = now
                action = policy.get_action(env, None)
                owners = policy.action_owner_stats()
                if owners:
                    owner_steps[owners["action_owner"]] += 1
                    owner_steps["grip:" + owners["grip_owner"]] += 1
                # SNAPSHOT BEFORE THE STEP. `step()` auto-resets the instant a termination fires,
                # so anything about the SCENE read afterwards describes the next episode's fresh
                # spawn, not the failure. Read this way round, the first version of this diagnostic
                # reported the part "back in the tray at its spawn position under 103 N" for every
                # abort -- which was the new episode settling, and sent the investigation after a
                # throw that never happened. The script already carried a comment about this trap a
                # few lines below; it still caught me.
                pre_snap = {
                    "net": {
                        f"{task.profiles[k]['variant']}#{k}": round(float(v), 2)
                        for k, v in enumerate(cp._net_norms(env, tuple(task.names["workpiece_contact_sensors"]))[0])
                        if float(v) > 0.5
                    },
                    "finger": {
                        f"{task.profiles[k]['variant']}#{k}": round(float(v), 2)
                        for k, v in enumerate(cp._filtered_norms(env, task.finger_sensor_names)[0])
                        if float(v) > 0.5
                    },
                    "tcp": [round(float(v), 4) for v in read_ee_pose(env)[0][0]],
                    "target_pos": [
                        round(float(v), 4) for v in env.scene[task.workpiece_names[target]].data.root_pos_w[0]
                    ],
                    "target_from_tcp": round(
                        float(
                            (
                                env.scene[task.workpiece_names[target]].data.root_pos_w[0] - read_ee_pose(env)[0][0]
                            ).norm()
                        ),
                        4,
                    ),
                }
                if os.environ.get("BDASH_HEARTBEAT") and step % int(os.environ["BDASH_HEARTBEAT"]) == 0:
                    # Step rate AND where the arm is. An episode that stops producing output is
                    # either finished, hung, or -- the case that actually happens here -- crawling
                    # because the arm is pinned against a joint limit and the solver is grinding.
                    # Those three look identical from outside and need different responses.
                    robot = env.scene[task.names["robot"]]
                    q = robot.data.joint_pos[0, :7]
                    lo = robot.data.soft_joint_pos_limits[0, :7, 0]
                    hi = robot.data.soft_joint_pos_limits[0, :7, 1]
                    m = torch.minimum(q - lo, hi - q)
                    tcp_now = read_ee_pose(env)[0][0]
                    print(
                        f"[hb] step={step:5d} t={time.monotonic() - _t0:6.1f}s "
                        f"pick_phase={int(policy._pick.phase[0])} "
                        f"tcp=({float(tcp_now[0]):.3f},{float(tcp_now[1]):.3f},{float(tcp_now[2]):.3f}) "
                        f"joint_margin={float(m.min()):.3f}@j{int(m.argmin()) + 1}",
                        flush=True,
                    )
                _obs, _rew, terminated, _trunc, _info = env.step(action)
                if bool(terminated[0]) and fired is None:
                    # WHICH termination fired, not just that one did. `terminated` is an OR over four
                    # terms with completely different meanings -- success, a dropped part, an
                    # over-force abort and the time-out -- and "the episode ended" is uninformative
                    # until you know which one. The manager keeps its per-term buffers across the
                    # auto-reset, so this one IS safe to read afterwards.
                    tm = env.termination_manager
                    fired = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
                    abort_snap = pre_snap
                # A REJECTed part ends the cycle at the decision: the policy holds the part and
                # nothing further can happen. Cutting here records the true cycle time of a reject
                # instead of a step-cap timeout, and the JSONL carries the decision.
                if hasattr(policy, "gate_stats") and policy.gate_stats().get("gate_decision") == "reject":
                    break
                if grasped_step is None and bool(cp.grasped(env, **grasp_kw)[0]):
                    grasped_step = step
                # Guarded on `terminated` because step() auto-resets the env the instant a
                # termination fires -- reading the scene after that samples the NEXT episode.
                if not bool(terminated[0]):
                    if tip_step is None and _target_axis_z(env, task, target, bcr) < 0.95:
                        tip_step = step
                    if not stall_snap and policy.stall_stats().get("stall_tcp_z") is not None:
                        stall_snap = _stall_snapshot(env, task, target, cp, bcr, spawn_pos)
                if bool(terminated[0]):
                    # `terminated` covers success (lifted) and object_dropped; the part was checked
                    # for having fallen off the table on the step before, which separates them.
                    if not dropped_before:
                        lifted_step = step
                    break
                if os.environ.get("BDASH_DEBUG") and step % 20 == 0:
                    _explain_lifted(env, task, target, step, cp, grasp_kw)

            pos = env.scene[task.workpiece_names[target]].data.root_pos_w[0]
            # spec §4-5: what the watchdog saw, so a failure names its own mechanism in the log
            # rather than needing a re-run with BDASH_DEBUG to find out.
            robot = env.scene[task.names["robot"]]
            jp, lim = robot.data.joint_pos[0], robot.data.soft_joint_pos_limits[0]
            margin = torch.minimum(jp - lim[:, 0], lim[:, 1] - jp)[:7]
            tcp, _ = read_ee_pose(env)
            ik_residual = float((policy.last_grasp[0] - tcp[0]).norm()) if policy.last_grasp is not None else None
            record = {
                "episode": episode,
                "target": target,
                "variant": variant,
                "side_lying": side,
                "grasped_step": grasped_step,
                "lifted_step": lifted_step,
                **policy.stall_stats(),
                "owner_steps": dict(owner_steps),
                **(policy.touch_off_stats() if hasattr(policy, "touch_off_stats") else {}),
                **(policy.reerect_stats() if hasattr(policy, "reerect_stats") else {}),
                **(policy.gate_stats() if hasattr(policy, "gate_stats") else {}),
                "injected_e_mm": injected_e,
                "injected_tilt_deg": injected_tilt,
                "terminated_by": fired,
                **fingerprint,
                "abort_forces": abort_snap,
                **escape,
                "joint_margin_min": round(float(margin.min()), 4),
                # WHICH joint is tightest, not just how tight. The re-erect route ends with 0.15-0.22
                # rad of travel left against the upright path's 0.79 -- zero overlap -- and the fix
                # differs completely by joint: a wound WRIST is undone by spinning the free azimuth
                # (the part is a cylinder, so that rotation is free), a wound SHOULDER or ELBOW is
                # not, and would mean the approach itself has to change.
                "joint_margin_at": int(margin.argmin()) + 1,
                "ik_residual": None if ik_residual is None else round(ik_residual, 4),
                "clearance_fallback": bool(getattr(env, "bdash_clearance_fallback", torch.zeros(1))[0]),
                # `seated`'s three conjuncts. Only read when the episode did not terminate: on a
                # termination step() has already auto-reset and the scene is the NEXT episode.
                **({} if lifted_step is not None else _seat_terms(env, task, target, bcr)),
                **({} if lifted_step is not None else _fixture_contacts(env, task, target)),
                # spec §12.3: the MINIMUM-INSERTION rate and the QC-INCLUSIVE good-part rate, on
                # separate rows. Both read ground truth and neither is a termination -- QC is a
                # judgement made after the commit (§2.3), not a reason to stop moving. Taken from
                # the last pre-step evaluation so a terminated episode is measured at the pose it
                # terminated in, one physics step (16.7 ms) before the auto-reset.
                **quality,
                **_hand_on_neighbour(env, task, policy, target, bcr),
                **stall_snap,
                "spawn_axis_z": spawn_axis_z,
                "tip_step": tip_step,
                # Where the part was, so a failure can be attributed to reach rather than to pose.
                # The tray cavity spans x[0.305, 0.595]: its inner corner is only 0.314 m from the
                # Franka base, which is a heavily folded, poorly conditioned arm configuration.
                "x": round(float(pos[0]), 4),
                "y": round(float(pos[1]), 4),
                "reach_r": round(float((pos[0] ** 2 + pos[1] ** 2) ** 0.5), 4),
                "result": "OK" if lifted_step is not None else ("NO_LIFT" if grasped_step is not None else "NO_GRASP"),
            }
            records.append(record)
            if jsonl is not None:
                jsonl.write(json.dumps(record) + "\n")
                jsonl.flush()
            print(
                f"[teachercheck] ep={episode:3d} {variant} side={int(side)} "
                f"grasped={'-' if grasped_step is None else grasped_step:>4} "
                f"lifted={'-' if lifted_step is None else lifted_step:>4}  {record['result']}",
                flush=True,
            )

    if jsonl is not None:
        jsonl.close()
        print(f"[teachercheck] wrote {args_cli.jsonl}", flush=True)
    _report(records)
    env.close()


def _load_quality(env, task, target, cp):
    from isaaclab_arena_environments.mdp import bdash_chuck_assets

    """The two REPORTED load rates. Privileged on purpose; never wired into a termination.

    `min_insert`  -- `cp.seated`: axis, `min_insert_depth`, inside the bore. "did it go in at all"
    `qc_ok`       -- spec §2.3's good-part test: seat angle <= 0.5 deg AND protrusion within the
                     per-variant window. This is what decides scrap, and it is strictly harder than
                     `min_insert`: a part can be in the bore and still be a reject.
    """
    load = task._load_params()
    seated = bool(
        cp.seated(
            env,
            workpiece_names=task.workpiece_names,
            chuck_name=task.chuck_body.name,
            face_height=float(task.geom["chuck_face_height"]),
            seat_angle_rad=load["seat_angle_rad"],
            min_insert_depth=float(task.cfg["seated"]["min_insert_depth"]),
            # The bore is a through hole: without an upper bound a part that fell out of it reads
            # as seated (measured, W-B at 83 mm against an 80 mm bore).
            max_insert_depth=float(bdash_chuck_assets.chuck_geometry()["body_height"]),
            bore_clearance=float(task.cfg["chuck_load"]["bore_clearance"]),
            tip_offset=tuple(task.geom["tip_offset"]),
        )[0]
    )
    protrusion = float(
        cp.protrusion_m(
            env,
            workpiece_names=task.workpiece_names,
            chuck_name=task.chuck_body.name,
            face_height=float(task.geom["chuck_face_height"]),
            datum_offsets=load["datum_offsets"],
            tip_offset=tuple(task.geom["tip_offset"]),
        )[0]
    )
    chuck = env.scene[task.chuck_body.name].data.root_pos_w[0]
    tip_z = float(env.scene[task.workpiece_names[target]].data.root_pos_w[0, 2])
    depth = float(chuck[2]) + float(task.geom["chuck_face_height"]) - tip_z
    nominal = load["nominals"][target]
    tol = load["protrusion_tol"][target]
    axis_z = max(-1.0, min(1.0, _rotate_z(env.scene[task.workpiece_names[target]].data.root_quat_w[0])))
    angle_deg = math.degrees(math.acos(axis_z))
    return {
        "min_insert": seated,
        "protrusion": round(protrusion, 4),
        "protrusion_err": round(protrusion - nominal, 4),
        "protrusion_tol": round(tol, 4),
        # spec §2.3: seat angle <= 0.5 deg, protrusion within the family window, right way up
        "qc_ok": bool(seated and angle_deg <= 0.5 and abs(protrusion - nominal) <= tol),
        "qc_angle_deg": round(angle_deg, 3),
        "insert_depth": round(depth, 4),
    }


def _rotate_z(quat):
    w, x, y, z = (float(v) for v in quat)
    return 1.0 - 2.0 * (x * x + y * y)


def _fixture_contacts(env, task, target):
    """Which chuck fixture the target is touching, per filter, in newtons.

    A stalled insertion reports a vertical reaction but not its source. `force_matrix_w` on a
    sensor filtered against several prims has one column per prim, so this names the partner
    outright instead of leaving it to be inferred from geometry.
    """
    if not task.fixture_sensor_names:
        return {}
    fm = env.scene[task.fixture_sensor_names[target]].data.force_matrix_w  # (N, B, M, 3)
    mags = fm[0].reshape(-1, len(task.fixture_filter_names), 3).norm(dim=-1).sum(dim=0)
    return {f"F_{name}": round(float(v), 3) for name, v in zip(task.fixture_filter_names, mags)}


def _seat_terms(env, task, target, bcr):
    """`seated` broken into its three conjuncts, for the target, in readable units.

    ``cp.seated`` is ``angle AND depth AND lateral``; a bool alone cannot say which one is short.
    Only meaningful when the episode did NOT terminate -- ``env.step()`` auto-resets on
    termination, so anything read after that samples the next episode's spawn.
    """
    data = env.scene[task.workpiece_names[target]].data
    axis = bcr._rotate_z_axis(data.root_quat_w[0])
    tip = data.root_pos_w[0]  # tip_offset is (0,0,0): the USD origin IS the leading end face
    chuck = env.scene[task.chuck_body.name].data.root_pos_w[0]
    face_z = float(chuck[2]) + float(task.geom["chuck_face_height"])
    return {
        "seat_angle_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, axis[2])))), 3),
        "seat_depth": round(face_z - float(tip[2]), 4),
        "seat_lateral": round(float(((tip[0] - chuck[0]) ** 2 + (tip[1] - chuck[1]) ** 2) ** 0.5), 4),
    }


def _tray_escape(env, task, spawn_pos, target):
    """Workpieces that have left the tray cavity, by axis-aligned footprint (spec §2.2).

    Overflow is a BUG, not a count to tune down: a part outside the tray is out of the picking
    workspace, and every frame it appears in is a frame the VLA is trained on with an unreachable
    object in it. This measures it instead of assuming it either way.

    It is deliberately not measured at spawn: ``sample_tray_layout`` provably cannot overflow --
    it bounds both axis endpoints of the capsule, so half-length and pose are already in the
    margin, verified at 3.008 mm worst over 9,000 placements against a 3.0 mm ``wall_inset_body``.
    What can put a part outside is PHYSICS: the tray lip is 7 mm against a side-lying Ø25 shaft
    whose centre sits at 12.5 mm, so ~0.4 m/s toward a wall is enough to climb out. Hence the
    displacement from spawn is reported alongside -- a part that escaped without moving would mean
    the sampler claim is wrong after all, and this measurement should catch that rather than
    presuppose it.
    """
    from isaaclab_arena_environments.mdp import bdash_chuck_randomization as bcr

    frame = bcr.tray_frame(task.cfg)
    tray = task.assets_cfg["tray"]
    wall = float(tray["wall"])
    half_x = (float(tray["outer"][0]) - 2.0 * wall) / 2.0
    half_y = (float(tray["outer"][1]) - 2.0 * wall) / 2.0
    # IN THE TRAY'S OWN FRAME. The tray is yawed, so comparing against world-axis half-extents
    # reports parts that never moved as having escaped -- measured, 22 of 30 episodes, almost all
    # with `moved_mm` 0.0. That contradiction is what the moved-from-spawn figure is here to expose,
    # and it convicted this check rather than the sampler.
    cos_y, sin_y = math.cos(-frame[2]), math.sin(-frame[2])
    origin = env.scene.env_origins[0, 0:3]
    # -1e9, not 0.0: seeded at zero the "worst" figure can only ever report a breach, so a clean
    # run prints +0.0 mm and says nothing about how close anything came.
    escaped, worst = {}, -1e9
    for k, name in enumerate(task.workpiece_names):
        # The TARGET is lifted out of the tray on purpose; that is the task, not an escape.
        if k == target:
            continue
        pos = env.scene[name].data.root_pos_w[0] - origin
        dx, dy = float(pos[0]) - frame[0], float(pos[1]) - frame[1]
        over_x = abs(dx * cos_y - dy * sin_y) - half_x
        over_y = abs(dx * sin_y + dy * cos_y) - half_y
        over = max(over_x, over_y)
        worst = max(worst, over)
        if over > 0.0:
            escaped[f"{task.profiles[k]['variant']}#{k}"] = {
                "over_mm": round(over * 1000.0, 1),
                "moved_mm": round(float((env.scene[name].data.root_pos_w[0] - spawn_pos[k]).norm()) * 1000.0, 1),
            }
    return {"escaped": escaped, "escape_margin_mm": round(worst * 1000.0, 1)}


def _target_axis_z(env, task, target, bcr):
    """Vertical component of the target's own axis: 1.0 standing, ~0 lying, <0 inverted."""
    return round(bcr._rotate_z_axis(env.scene[task.workpiece_names[target]].data.root_quat_w[0])[2], 4)


def _stall_snapshot(env, task, target, cp, bcr, spawn_pos):
    """State at the FIRST stall: what the fingers are touching, and whether the target still stands.

    The watchdog counters say a phase stopped converging; they never say why, which is what left
    every DESCEND stall unattributed. The leading hypothesis is that the descent knocks the target
    over and the teacher then chases a part that is no longer where -- or in the pose -- it was
    certified in. ``stall_target_axis_z`` (1.0 upright, ~0 lying) and ``stall_target_moved`` test
    that directly, at the moment of the stall rather than at the end of the episode.

    NOTE the sensors are workpiece<->FINGER only (``bdash_chuck_predicates._filtered_norms``): the
    hand BODY resting on a neighbour registers nothing at all. So "no contact at the stall" is
    itself a result -- it rules the fingers out and points at the hand body or a non-contact cause.
    """
    forces = cp._filtered_norms(env, task.finger_sensor_names)[0]
    data = env.scene[task.workpiece_names[target]].data
    axis = bcr._rotate_z_axis(data.root_quat_w[0])
    return {
        "stall_contacts": {
            f"{task.profiles[k]['variant']}#{k}": round(float(v), 2) for k, v in enumerate(forces) if float(v) > 0.5
        },
        "stall_target_force": round(float(forces[target]), 2),
        "stall_target_axis_z": round(axis[2], 4),
        "stall_target_moved": round(float((data.root_pos_w[0] - spawn_pos[target]).norm()), 4),
    }


def _hand_on_neighbour(env, task, policy, target, bcr):
    """Re-check the §0-10 geometry at the END of the episode, against the yaw actually commanded.

    Turns "the watchdog gave up" into a falsifiable claim. ``predicted_stall_z = grasp_z + worst``
    is the TCP height at which the hand's underside first touches the most obstructing neighbour --
    the same zero-free-parameter line that reproduced the pre-guarantee stalls to +-2 mm. If the
    measured ``stall_tcp_z`` sits on it, the hand-on-neighbour mechanism survived the guarantee
    (i.e. the guarantee has a hole, most likely because the sampler certified the target against a
    different yaw or a pre-settling layout). If it does not, the residual failures are something new
    and the §0-10 model is not the thing to keep tightening.
    """
    if policy.last_grasp is None:
        return {}
    hand = task.cfg["target"]["hand"]
    clear_z = float(hand["clear_z"])
    placed = []
    for k, name in enumerate(task.workpiece_names):
        data = env.scene[name].data
        pos = data.root_pos_w[0]
        axis = bcr._rotate_z_axis(data.root_quat_w[0])
        side = bool(env.bdash_side_lying[0, k])
        top = bcr.piece_top_z(task.profiles[k], float(pos[2]), axis, side)
        placed.append(((float(pos[0]), float(pos[1])), top, float(task.profiles[k]["max_radius"])))

    gx, gy, gz = (float(v) for v in policy.last_grasp[0])
    axis_t = tuple(float(v) for v in policy.last_axis[0])
    side_t = bool(policy.last_side_lying[0])
    yaw = 0.0 if policy.last_yaw is None else float(policy.last_yaw[0])
    close_dir = bcr.closing_direction(axis_t, side_t, 0.0 if side_t else yaw)
    neighbours = [p for j, p in enumerate(placed) if j != target]
    clear, worst = bcr.hand_clearance(
        (gx, gy),
        gz,
        close_dir,
        neighbours,
        half_span=float(hand["half_span"]),
        half_width=float(hand["half_width"]),
        clear_z=clear_z,
    )
    return {
        "clear_at_end": bool(clear),
        "worst_overlap": round(worst, 4),
        # NOT the hand underside: the TCP height at which the underside meets the blocking top.
        "predicted_stall_z": None if clear else round(gz + worst, 4),
    }


def _explain_lifted(env, task, target, step, cp, grasp_kw):
    """Break `lifted` into its terms, so a failure names which one is false rather than just 'no'."""
    name = task.workpiece_names[target]
    width = float(cp.gripper_width(env, task.names["robot"])[0])
    forces = cp._filtered_norms(env, task.finger_sensor_names)[0]
    force = float(forces[target])
    # Which workpiece are the fingers actually touching? A DESCEND stall with force on a NON-target
    # part means the gripper is pressing into a neighbour -- i.e. the straight-line descent is
    # colliding, which no amount of controller gain will fix.
    others = [(task.profiles[k]["variant"], float(f)) for k, f in enumerate(forces) if k != target and f > 0.5]
    if others:
        print(f"[contact] step={step:3d} TARGET={force:.2f}N  NEIGHBOURS={others}", flush=True)
    part_z = float(env.scene[name].data.root_pos_w[0, 2])
    speed = float(env.scene[name].data.root_lin_vel_w[0].norm())
    rim_z = float(env.scene[task.tray.name].data.root_pos_w[0, 2]) + float(task.geom["tray_rim_height"])
    clear = part_z - rim_z
    g = grasp_kw
    print(
        f"[lifted?] step={step:3d} width={width:.4f} in[{g['width_min']},{g['width_max']}]="
        f"{g['width_min'] < width < g['width_max']}  force={force:6.2f}>{g['grasp_force']}="
        f"{force > g['grasp_force']}  clear={clear:+.4f}>{task.cfg['lifted']['height']}="
        f"{clear > task.cfg['lifted']['height']}  speed={speed:.3f}<{task.cfg['lifted']['speed_max']}="
        f"{speed < task.cfg['lifted']['speed_max']}",
        flush=True,
    )


def _report(records):
    buckets = collections.defaultdict(list)
    for record in records:
        buckets[(record["variant"], record["side_lying"])].append(record)
        buckets[("ALL", record["side_lying"])].append(record)

    # WHICH COLUMN IS THE SUCCESS RATE depends on the stage, and getting it wrong is not cosmetic.
    #
    # At stage="pick" the terminal IS the lift, so `terminated` and "the teacher did its job" are
    # the same event and one column suffices.
    #
    # At stage="full" they are NOT. The terminal is `loaded`, which is PROPRIOCEPTIVE by design
    # (depth from the TCP height and the commanded station; no workpiece pose -- see the predicate's
    # docstring for why). Proprioception cannot see the part's ORIENTATION, so a part held sideways
    # and pressed onto the chuck face satisfies it: `depth` only ever asked how low the wrist went.
    # Measured on 20 side-lying episodes: `loaded` fired in 20 of them and `seated` in ZERO -- a
    # printed "100%" against a bore with nothing in it.
    #
    # So at full stage the headline is `min_insert` (= `seated`: axis aligned, deep enough, inside
    # the bore). It reads ground truth, which is exactly why it is a REPORTING signal and never a
    # termination (spec: no privileged information in terminations, and the teacher must not be
    # able to steer by it).
    full = any(r.get("min_insert") is not None for r in records)
    key = "min_insert" if full else "lifted_step"

    def _ok(r):
        return bool(r["min_insert"]) if full else (r["lifted_step"] is not None)

    head = "insert" if full else "lift"
    print(f"\n  variant  pose        n   grasp  {head:>6s}" + ("     end     qc" if full else ""))
    print("  " + "-" * (40 + (16 if full else 0)))
    for variant in ("W-A", "W-B", "W-C", "ALL"):
        for side in (False, True):
            rows = buckets.get((variant, side))
            if not rows:
                continue
            n = len(rows)
            grasp_rate = sum(r["grasped_step"] is not None for r in rows) / n
            rate = sum(_ok(r) for r in rows) / n
            pose = "side-lying" if side else "upright   "
            line = f"  {variant:7s}  {pose}  {n:3d}  {grasp_rate:6.0%}  {rate:6.0%}"
            if full:
                # `end` is the terminal that actually stopped the episode. Printed BESIDE the real
                # rate, never instead of it: the gap between the two columns is the measurement of
                # how often the teacher believed it was done and was wrong.
                ended = sum(r["lifted_step"] is not None for r in rows) / n
                qc = sum(bool(r.get("qc_ok")) for r in rows) / n
                line += f"  {ended:6.0%} {qc:6.0%}"
            print(line)

    def _rate(side):
        rows = buckets.get(("ALL", side), [])
        return (sum(_ok(r) for r in rows) / len(rows)) if rows else 0.0

    upright, lying = _rate(False), _rate(True)
    print(f"\n  GATE ({key}): upright {upright:.0%} (need >=90%)   side-lying {lying:.0%} (need >=70%)")
    verdict = "PASS" if upright >= 0.90 and lying >= 0.70 else "BELOW TARGET"
    print(f"CHUCK_TEACHER_{'OK' if verdict == 'PASS' else 'BELOW_TARGET'}  upright={upright:.3f} side={lying:.3f}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # PRINT IT. Isaac Sim installs its own excepthook and the traceback otherwise never reaches
        # the log: the run simply stops mid-episode and the app begins shutting down, which looks
        # exactly like a hang from outside. Measured -- a run that died at step ~700 left no trace
        # of why, and the only clue was a py-spy dump showing the process already inside
        # `simulation_app.close()`.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
