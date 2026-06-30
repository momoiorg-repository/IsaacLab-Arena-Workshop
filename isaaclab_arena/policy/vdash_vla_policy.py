# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""V-DASH VLA + rule-based insertion switcher policy (the brief's M6 hierarchy for VLA eval).

The fine-tuned GR00T VLA is the **front end**: it drives pick -> handoff from camera RGB + state,
emitting 8-dof joint-position targets (it was trained on ``processed_actions``). Once the §3.4
``handoff`` predicate fires for an env, that env **latches** into the rule-based
:class:`~isaaclab_arena.controllers.insertion_controller.InsertionController` (force-limited press +
spiral search), exactly as ``vdash_scripted`` does — only the pick stage is the VLA instead of
``ScriptedPick``.

Action-space bridge: the eval env is the ``franka_joint`` embodiment (8-dof joint-position action),
so the VLA output applies natively. The InsertionController emits a 7-vector relative-IK action
(``[dx,dy,dz,ax,ay,az,grip]``); we convert it to joint targets with the same DLS IK the recording
env uses (:class:`FrankaIKJointRecordingAction`): feed the scaled pose delta through a
:class:`DifferentialIKController` to get arm joint targets, and map the binary gripper to the finger
target. Result is the 8-dof ``[7 arm + 1 finger]`` vector the ``franka_joint`` action term expects.

Run single-env (non-tiled cameras render one env):

    policy_runner.py --policy_type isaaclab_arena.policy.vdash_vla_policy.VDashVLAPolicy \\
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/vdash_pick_insert_gr00t_closedloop_config.yaml \\
        --num_envs 1 --enable_cameras --headless --num_episodes 20 \\
        vdash_pick_insert --embodiment franka_joint --clearance 2.0 --level L1
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import math
import os
import torch
from dataclasses import dataclass

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, subtract_frame_transforms

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.gr00t_closedloop_policy import Gr00tClosedloopPolicy, Gr00tClosedloopPolicyArgs


@dataclass
class VDashVLAPolicyArgs:
    policy_config_yaml_path: str = ""
    policy_device: str = "cuda"
    num_envs: int = 1

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> VDashVLAPolicyArgs:
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=getattr(args, "policy_device", "cuda"),
            num_envs=getattr(args, "num_envs", 1),
        )


@register_policy
class VDashVLAPolicy(PolicyBase):
    """GR00T VLA pick->handoff, then rule-based insertion (run in the ``franka_joint`` env)."""

    name = "vdash_vla"
    config_class = VDashVLAPolicyArgs

    # franka_joint action term body / offset (must match FrankaActionsCfg / franka_joint env)
    _EE_BODY = "panda_hand"
    _EE_OFFSET = (0.0, 0.0, 0.107)
    _ARM_JOINTS = "panda_joint.*"
    _FINGER1 = "panda_finger_joint1"
    _IK_SCALE = 0.5  # FrankaActionsCfg arm_action scale

    def __init__(self, config: VDashVLAPolicyArgs):
        super().__init__(config)
        self._vla = Gr00tClosedloopPolicy(
            Gr00tClosedloopPolicyArgs(
                policy_config_yaml_path=config.policy_config_yaml_path,
                policy_device=config.policy_device,
                num_envs=config.num_envs,
            )
        )
        self._ins = None
        self._pick = None  # set only in the scripted-pick diagnostic mode
        self._ik = None
        self._in_insertion: torch.Tensor | None = None
        self._hp: dict | None = None
        self._ids = None  # (arm_joint_ids, finger1_id, ee_body_idx, jacobi_idx, ee_offset_pos, ee_offset_quat)
        self._gate = None  # precision-budget handoff gate (VDASH_BUDGET_GATE); set in _ensure

    # ----------------------------------------------------------------- setup
    def _ensure(self, env):
        if self._ins is not None:
            return
        import isaaclab_arena_environments.mdp.vdash_predicates as _vp  # noqa: F401  (predicate module)
        from isaaclab_arena.controllers.insertion_controller import InsertionController
        from isaaclab_arena_environments.mdp.vdash_config import load_controllers_cfg, load_task_cfg

        u = env.unwrapped
        task = u.cfg.isaaclab_arena_env.task
        names = task.names
        mouth_height = float(task.geom["mouth_height"])
        cc = load_controllers_cfg()
        self._ins = InsertionController(
            names, mouth_height, cc["ee_control"], cc["insertion"], grip_offset=float(cc["pick"]["grip_offset"])
        )

        # §3.4 handoff predicate params (mirrors record_vla_demos.py)
        t = load_task_cfg()
        snames = {**t["scene_names"], "peg": names["peg"], "socket": names["socket"]}
        geom = t["geometry"]
        self._hp = dict(
            peg_name=snames["peg"],
            socket_name=snames["socket"],
            robot_name=snames["robot"],
            peg_finger_sensor=snames["peg_finger_sensor"],
            tip_offset=tuple(geom["tip_offset"]),
            mouth_height=mouth_height,
            **t["handoff"],
            **t["grasped"],
        )

        n, dev = u.num_envs, u.device
        self._in_insertion = torch.zeros(n, dtype=torch.bool, device=dev)
        from isaaclab_arena.controllers.budget_gate import BudgetGate

        _g = BudgetGate()
        self._gate = _g if _g.enabled else None

        # IK bridge: same controller config as the recording env's FrankaIKJointRecordingAction.
        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            num_envs=n,
            device=dev,
        )
        robot = u.scene["robot"]
        arm_ids, _ = robot.find_joints(self._ARM_JOINTS, preserve_order=True)
        finger1_ids, _ = robot.find_joints(self._FINGER1)
        body_ids, _ = robot.find_bodies(self._EE_BODY)
        ee_body_idx = body_ids[0]
        # fixed-base articulation: jacobian body index is body_idx - 1
        jacobi_idx = ee_body_idx - 1
        off_pos = torch.tensor(self._EE_OFFSET, device=dev).repeat(n, 1)
        off_quat = torch.zeros(n, 4, device=dev)
        off_quat[:, 0] = 1.0
        self._ids = (torch.tensor(arm_ids, device=dev), finger1_ids[0], ee_body_idx, jacobi_idx, off_pos, off_quat)

        # diagnostic: replace the VLA pick with the scripted expert pick to test the IK->joint bridge
        # in isolation (reliable grasp -> if insertion still fails, the bridge is the bug).
        if os.environ.get("VDASH_VLA_SCRIPTED_PICK"):
            from isaaclab_arena.controllers.scripted_pick import ScriptedPick

            self._pick = ScriptedPick(names, cc["ee_control"], cc["pick"])

    # ------------------------------------------------------------------ frame helpers (mirror IK action term)
    def _ee_frame_pose(self, env):
        u = env.unwrapped
        robot = u.scene["robot"]
        _, _, ee_body_idx, _, off_pos, off_quat = self._ids
        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # apply the tool offset (panda_hand -> EE control frame)
        ee_pos_b, ee_quat_b = combine_frame_transforms(ee_pos_b, ee_quat_b, off_pos, off_quat)
        return ee_pos_b, ee_quat_b

    def _ee_frame_jacobian(self, env):
        u = env.unwrapped
        robot = u.scene["robot"]
        arm_ids, _, _, jacobi_idx, off_pos, _ = self._ids
        jac = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, :]
        jac = jac[:, :, arm_ids]
        # shift the linear part for the tool offset: J_v += [omega x offset] == -skew(offset) @ J_w
        _, ee_quat_b = self._ee_frame_pose(env)
        off_w = torch.bmm(matrix_from_quat(ee_quat_b), off_pos.unsqueeze(-1)).squeeze(-1)
        skew = torch.zeros(jac.shape[0], 3, 3, device=jac.device)
        skew[:, 0, 1] = -off_w[:, 2]
        skew[:, 0, 2] = off_w[:, 1]
        skew[:, 1, 0] = off_w[:, 2]
        skew[:, 1, 2] = -off_w[:, 0]
        skew[:, 2, 0] = -off_w[:, 1]
        skew[:, 2, 1] = off_w[:, 0]
        jac[:, 0:3, :] += torch.bmm(skew, jac[:, 3:6, :])
        return jac

    def _ik_to_joint(self, env, ik_action: torch.Tensor) -> torch.Tensor:
        """Convert the InsertionController's (N,7) relative-IK action to the (N,8) joint-position action."""
        u = env.unwrapped
        robot = u.scene["robot"]
        arm_ids, finger1_id, _, _, _, _ = self._ids
        pose_cmd = ik_action[:, :6] * self._IK_SCALE  # env applies scale before the controller
        ee_pos_b, ee_quat_b = self._ee_frame_pose(env)
        jac = self._ee_frame_jacobian(env)
        joint_pos = robot.data.joint_pos[:, arm_ids]
        self._ik.set_command(pose_cmd, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        arm_des = self._ik.compute(ee_pos_b, ee_quat_b, jac, joint_pos)
        # gripper: binary IK action -> finger target (open 0.04 / closed 0.0)
        grip = torch.where(ik_action[:, 6] >= 0.0, 0.04, 0.0).unsqueeze(-1)
        return torch.cat([arm_des, grip], dim=-1)

    # ------------------------------------------------------------------ API
    def set_task_description(self, task_description: str | None) -> str:
        return self._vla.set_task_description(task_description)

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        self._ensure(env)
        import isaaclab_arena_environments.mdp.vdash_predicates as vp

        uenv = env.unwrapped  # predicate / controller / frame helpers need the base env
        if self._pick is not None:
            # diagnostic: scripted pick (IK) -> bridge; latch insertion on the pick's own finished flag
            pick_ik, pick_finished = self._pick.step(uenv)
            self._in_insertion = self._in_insertion | pick_finished
            pick_act = self._ik_to_joint(uenv, pick_ik)
        else:
            pick_act = self._vla.get_action(env, observation)  # (N, 8) joint targets (uses obs, not env)
            fired = vp.handoff(uenv, **self._hp)
            if self._gate is not None:  # precision-budget gate on the VLA->rule handoff
                newly = fired & ~self._in_insertion
                if bool(newly.any()):
                    go, tilt_e, e_mm, dz = self._gate.evaluate(uenv, self._hp["peg_finger_sensor"])
                    for i in newly.nonzero(as_tuple=True)[0].tolist():
                        print(
                            f"[budgetgate] pol=vla ep={getattr(self, '_dbg_ep', 0)} env={i} "
                            f"tilt_est={float(tilt_e[i]):.1f}deg e_est={float(e_mm[i]):.1f}mm "
                            f"budget={self._gate.budget_mm:.1f}mm decision={'GO' if bool(go[i]) else 'NOGO'} "
                            f"dz={float(dz[i]):.2f}N",
                            flush=True,
                        )
                    if self._gate.active and bool((newly & ~go).any()):
                        self._ins._decant = True  # correct-on-no-go: de-cant straightens the cocked grasp
            self._in_insertion = self._in_insertion | fired  # latch on §3.4 handoff
        if os.environ.get("VDASH_VLA_DEBUG"):  # TEMP: where is the gripper vs the peg, and does it close?
            self._dbg_step = getattr(self, "_dbg_step", 0) + 1
            if self._dbg_step % 10 == 1:
                robot = uenv.scene["robot"]
                jp = robot.data.joint_pos[0, :7]
                pa = pick_act[0, :7]
                d = (pa - jp).abs()
                ee = robot.data.body_state_w[0, self._ids[2], :3]
                peg = uenv.scene[self._hp["peg_name"]].data.root_pos_w[0, :3]
                dxy = float(((ee[:2] - peg[:2]) ** 2).sum() ** 0.5)  # horizontal gripper->peg offset
                dz = float(ee[2] - peg[2])  # vertical (EE above peg)
                grip = float(robot.data.joint_pos[0, self._ids[1]])  # finger gap: open~0.04 closed~0.0
                # track closest horizontal approach this episode (is it ever grasp-aligned?)
                self._dbg_min_dxy = min(getattr(self, "_dbg_min_dxy", 1e9), dxy)
                ep = getattr(self, "_dbg_ep", 0)
                print(
                    f"[vladbg] ep={ep} step={self._dbg_step} insert={bool(self._in_insertion[0])} "
                    f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) peg=({peg[0]:.3f},{peg[1]:.3f},{peg[2]:.3f}) "
                    f"dxy={dxy:.3f} dz={dz:.3f} minDxy={self._dbg_min_dxy:.3f} grip={grip:.3f} "
                    f"|tgt-cur|max={d.max():.4f}",
                    flush=True,
                )
        if os.environ.get("VDASH_GRASP_DIAG"):  # grasp-quality diag: how does the peg sit IN the gripper?
            from isaaclab.utils.math import quat_apply

            from isaaclab_arena.controllers.ee_control import read_ee_pose

            robot = uenv.scene["robot"]
            pegd = uenv.scene[self._hp["peg_name"]].data
            ee_pos_b, ee_quat_b = read_ee_pose(uenv)  # ee_frame (the SAME source tip_estimate uses)
            ee_pos = ee_pos_b[0]
            ee_quat = ee_quat_b[0]
            zloc = torch.tensor([0.0, 0.0, 1.0], device=ee_quat.device)
            peg_axis = quat_apply(pegd.root_quat_w[0], zloc)  # peg long axis in world
            ee_axis = quat_apply(ee_quat, zloc)  # gripper axis in world
            # in-gripper tilt = angle between peg axis and gripper axis (0 = nominal straight grasp).
            # The controller's tip_estimate assumes these are colinear; misalignment => wrong tip target.
            cosang = float(torch.dot(peg_axis, ee_axis).abs().clamp(max=1.0))
            ingrip_tilt = math.degrees(math.acos(cosang))
            # lateral offset of the peg from the gripper axis (perpendicular component of peg-EE), mm
            rel = pegd.root_pos_w[0, :3] - ee_pos
            along = torch.dot(rel, ee_axis) * ee_axis
            lat_mm = float(torch.norm(rel - along)) * 1000.0
            peg_speed = float(torch.norm(pegd.root_lin_vel_w[0])) if hasattr(pegd, "root_lin_vel_w") else -1.0
            grip = float(robot.data.joint_pos[0, self._ids[1]])
            ep = getattr(self, "_dbg_ep", 0)
            latched = bool(self._in_insertion[0])
            if not getattr(self, "_gd_handoff_logged", False) and latched:
                self._gd_handoff_logged = True
                print(
                    f"[graspdiag] pol=vla ep={ep} HANDOFF ingrip_tilt={ingrip_tilt:.1f}deg "
                    f"lat={lat_mm:.1f}mm peg_speed={peg_speed:.3f} grip={grip:.3f}",
                    flush=True,
                )
            elif getattr(self, "_dbg_step", 0) % 20 == 1 and grip < 0.02:  # grasped, periodic
                print(
                    f"[graspdiag] pol=vla ep={ep} step={getattr(self,'_dbg_step',0)} {'INS' if latched else 'pick'} "
                    f"ingrip_tilt={ingrip_tilt:.1f}deg lat={lat_mm:.1f}mm peg_speed={peg_speed:.3f}",
                    flush=True,
                )
        if os.environ.get("VDASH_VLA_FORCE_INSERT"):  # debug: exercise the IK->joint bridge from step 0
            self._in_insertion[:] = True
        ins_ik = self._ins.step(uenv, active=self._in_insertion)  # (N, 7) relative-IK
        if os.environ.get("VDASH_BACKSWITCH") and getattr(self._ins, "gave_up", None) is not None:
            self._in_insertion = self._in_insertion & ~self._ins.gave_up  # back-switch: return to front-end
        ins_act = self._ik_to_joint(uenv, ins_ik)  # (N, 8) joint targets
        return torch.where(self._in_insertion.unsqueeze(-1), ins_act, pick_act)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._vla.reset(env_ids)
        if self._pick is not None:
            self._pick.reset(env_ids)
        if self._ins is not None:
            self._ins.reset(env_ids)
            idx = slice(None) if env_ids is None else env_ids
            self._in_insertion[idx] = False
        if os.environ.get("VDASH_VLA_DEBUG") or os.environ.get("VDASH_GRASP_DIAG"):  # per-episode segmentation
            self._dbg_ep = getattr(self, "_dbg_ep", 0) + 1
            self._dbg_step = 0
            self._dbg_min_dxy = 1e9
            self._gd_handoff_logged = False

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("VDash VLA Policy", "GR00T VLA + rule-based insertion switcher")
        group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="GR00T closed-loop config YAML (the VLA front end)",
        )
        group.add_argument("--policy_device", type=str, default="cuda")
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> VDashVLAPolicy:
        return VDashVLAPolicy(VDashVLAPolicyArgs.from_cli_args(args))
