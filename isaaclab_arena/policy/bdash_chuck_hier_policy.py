# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The B-DASH hierarchy at run time: GR00T VLA picks, the classical P3 stack does the rest.

Phase A drives the ``franka_joint`` env with the fine-tuned GR00T policy (language-conditioned
pick). The moment the KINEMATIC lift condition holds -- finger width inside the grasp band and the
TCP above the lift height for a few consecutive steps, no object state read -- ownership moves to
the P3 teacher stack: transport, touch-off, gate, refix/insert. The VLA's grasp height is arbitrary
and unknown, which is exactly the quantity touch-off measures, so the classical side needs NO
information about what the VLA did: the boundary contract (measure -> derived window -> 3-way) is
the interface. That is the thesis of the talk, running.

The teacher outputs the IK embodiment's 7-dof action; the env is ``franka_joint`` (8-dof joint
targets), bridged with the same DLS-IK conversion `bdash_vla_policy.py` validated 4/4 on the peg
task (constants and frame helpers ported verbatim from there).

``BDASH_HIER_FORCE_SWITCH=<step>`` forces the handover at a fixed step -- a mechanics smoke for the
switch+bridge with a model that cannot yet lift (the full247 model approaches but never grasps).
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import os
import torch
from dataclasses import dataclass, field

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, subtract_frame_transforms

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.gr00t_closedloop_policy import Gr00tClosedloopPolicy, Gr00tClosedloopPolicyArgs


@dataclass
class BDashChuckHierArgs:
    policy_config_yaml_path: str = field(default="")
    policy_device: str = field(default="cuda:0")
    num_envs: int = field(default=1)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> BDashChuckHierArgs:
        return cls(
            policy_config_yaml_path=getattr(args, "vla_config", ""),
            num_envs=getattr(args, "num_envs", 1),
        )


@register_policy
class BDashChuckHierPolicy(PolicyBase):

    name = "bdash_chuck_hier"
    config_class = BDashChuckHierArgs

    # franka_joint action term body / offset -- MUST match bdash_vla_policy.py (validated bridge)
    _EE_BODY = "panda_hand"
    _EE_OFFSET = (0.0, 0.0, 0.107)
    _ARM_JOINTS = "panda_joint.*"
    _FINGER1 = "panda_finger_joint1"
    _IK_SCALE = 0.5
    _LIFT_TCP_Z = 0.15  # m; kinematic lift height for the handover trigger
    _WIDTH_BAND = (0.015, 0.065)  # m; holding-something finger width (all variants)
    _STABLE_STEPS = 5

    def __init__(self, config: BDashChuckHierArgs):
        super().__init__(config)
        # The classical stack runs gate mode P3 -- measurement decides. Set BEFORE teacher _ensure.
        os.environ.setdefault("BDASH_POLICY_MODE", "P3")
        from isaaclab_arena.policy.bdash_chuck_policy import BDashChuckTeacherArgs, BDashChuckTeacherPolicy

        self._vla = Gr00tClosedloopPolicy(
            Gr00tClosedloopPolicyArgs(
                policy_config_yaml_path=config.policy_config_yaml_path,
                policy_device=config.policy_device,
                num_envs=config.num_envs,
            )
        )
        self._teacher = BDashChuckTeacherPolicy(BDashChuckTeacherArgs())
        self._ik = None
        self._ids = None
        self._switched = False
        self._teacher_inited = False
        self._stable = 0
        self._step = 0
        self._force_switch = int(os.environ.get("BDASH_HIER_FORCE_SWITCH", "0") or "0")

    # ---------------- bridge (ported verbatim from bdash_vla_policy.py) ----------------
    def _ensure_bridge(self, env) -> None:
        if self._ik is not None:
            return
        u = env.unwrapped
        n, dev = u.num_envs, u.device
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
        jacobi_idx = ee_body_idx - 1
        off_pos = torch.tensor(self._EE_OFFSET, device=dev).repeat(n, 1)
        off_quat = torch.zeros(n, 4, device=dev)
        off_quat[:, 0] = 1.0
        self._ids = (torch.tensor(arm_ids, device=dev), finger1_ids[0], ee_body_idx, jacobi_idx, off_pos, off_quat)

    def _ee_frame_pose(self, env):
        u = env.unwrapped
        robot = u.scene["robot"]
        _, _, ee_body_idx, _, off_pos, off_quat = self._ids
        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_pos_b, ee_quat_b = combine_frame_transforms(ee_pos_b, ee_quat_b, off_pos, off_quat)
        return ee_pos_b, ee_quat_b

    def _ee_frame_jacobian(self, env):
        u = env.unwrapped
        robot = u.scene["robot"]
        arm_ids, _, _, jacobi_idx, off_pos, _ = self._ids
        jac = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, :]
        jac = jac[:, :, arm_ids]
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
        u = env.unwrapped
        robot = u.scene["robot"]
        arm_ids, finger1_id, _, _, _, _ = self._ids
        pose_cmd = ik_action[:, :6] * self._IK_SCALE
        ee_pos_b, ee_quat_b = self._ee_frame_pose(env)
        jac = self._ee_frame_jacobian(env)
        joint_pos = robot.data.joint_pos[:, arm_ids]
        self._ik.set_command(pose_cmd, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        arm_des = self._ik.compute(ee_pos_b, ee_quat_b, jac, joint_pos)
        grip = torch.where(ik_action[:, 6] >= 0.0, 0.04, 0.0).unsqueeze(-1)
        return torch.cat([arm_des, grip], dim=-1)

    # ---------------- hierarchy ----------------
    def set_task_description(self, text: str | None) -> str:
        return self._vla.set_task_description(text)

    def _lift_trigger(self, env) -> bool:
        """Kinematics only: holding-width fingers, TCP above the lift height, stable a few steps.

        "Stable" includes the WIDTH itself: a gripper closing on air sweeps through the holding
        band (measured: false handover at step 76 of the first smoke), while a gripper holding a
        part sits still at the part's width. Requiring the width to move less than 1 mm across the
        window rejects the sweep without reading any object state."""
        u = env.unwrapped
        robot = u.scene["robot"]
        q = robot.data.joint_pos[0]
        width = float(q[-2] + q[-1])
        _, _, ee_body_idx, _, _, _ = self._ids
        tcp_z = float(robot.data.body_state_w[0, ee_body_idx, 2])
        held = self._WIDTH_BAND[0] < width < self._WIDTH_BAND[1] and tcp_z > self._LIFT_TCP_Z
        w_hist = getattr(self, "_w_hist", [])
        w_hist = (w_hist + [width])[-self._STABLE_STEPS :]
        self._w_hist = w_hist
        width_still = len(w_hist) == self._STABLE_STEPS and (max(w_hist) - min(w_hist)) < 0.001
        self._stable = self._stable + 1 if (held and width_still) else 0
        return self._stable >= self._STABLE_STEPS

    def _hand_over(self, env) -> None:
        """Initialize the teacher mid-episode as 'pick DONE, part in hand'."""
        from isaaclab_arena.controllers.scripted_pick import DONE

        self._teacher.get_action(env, None)  # one call builds all lazy state (pick, controllers)
        self._teacher._pick.phase[:] = DONE
        self._teacher_inited = True
        print(f"[hier] HANDOVER at step {self._step}: VLA -> classical P3 stack", flush=True)

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        self._ensure_bridge(env)
        self._step += 1
        if not self._switched:
            force = self._force_switch and self._step >= self._force_switch
            if force or self._lift_trigger(env):
                self._switched = True
                self._hand_over(env)
            else:
                return self._vla.get_action(env, observation)
        ik_action = self._teacher.get_action(env, None)
        return self._ik_to_joint(env, ik_action)

    def reset(self, env_ids: torch.Tensor | None = None):
        self._vla.reset(env_ids)
        if self._teacher_inited:
            self._teacher.reset()
        self._switched = False
        self._teacher_inited = False
        self._stable = 0
        self._step = 0
        self._w_hist = []

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> BDashChuckHierPolicy:
        return BDashChuckHierPolicy(BDashChuckHierArgs.from_cli_args(args))
