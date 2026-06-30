# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""V-DASH scripted expert policy: scripted pick → rule-based insertion (brief §3.3).

Sequences :class:`~isaaclab_arena.controllers.scripted_pick.ScriptedPick` and
:class:`~isaaclab_arena.controllers.insertion_controller.InsertionController`. Each env latches into
the insertion controller once the pick reports ``finished`` (the VLA→rule *handoff*); before that the
pick controller drives. This is the rule-based arm of the V-DASH hierarchy used to measure the
clearance / scene-diversity success curves (M5) and, with the switcher, the ablations (M6).

Registered as policy ``vdash_scripted`` (use with ``policy_runner.py --policy_type vdash_scripted``).
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import os
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.controllers.insertion_controller import InsertionController
from isaaclab_arena.controllers.scripted_pick import ScriptedPick
from isaaclab_arena.policy.policy_base import PolicyBase


@dataclass
class VDashScriptedPolicyArgs:
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> VDashScriptedPolicyArgs:
        return cls()


@register_policy
class VDashScriptedPolicy(PolicyBase):

    name = "vdash_scripted"
    config_class = VDashScriptedPolicyArgs

    def __init__(self, config: VDashScriptedPolicyArgs):
        super().__init__(config)
        self._pick: ScriptedPick | None = None
        self._ins: InsertionController | None = None
        self._in_insertion: torch.Tensor | None = None
        self._m7_offset: torch.Tensor | None = None  # (N,2) M7 lateral offset (m)
        self._m7_tilt: torch.Tensor | None = None  # (N,2) M7 lean = θ·(cos az, sin az)
        self._m7_recover: bool = False  # release the hold at PRESS entry (recovery basin)
        self._debug = bool(os.environ.get("VDASH_DEBUG"))
        self._step = 0
        self._gate = None  # precision-budget handoff gate (VDASH_BUDGET_GATE); set in _ensure

    def set_m7(self, offset_xy: torch.Tensor, tilt_vec: torch.Tensor, recover: bool = False) -> None:
        """E4/M7 harness: hold the insertion at a per-env offset (m) + lean (rad·dir); applied each step.
        recover=True → deliver the perturbed handoff then release the hold at PRESS (recovery basin)."""
        self._m7_offset = offset_xy
        self._m7_tilt = tilt_vec
        self._m7_recover = recover

    # ------------------------------------------------------------------ setup
    def _ensure(self, env):
        if self._pick is not None:
            return
        from isaaclab_arena_environments.mdp.vdash_config import load_controllers_cfg

        task = env.unwrapped.cfg.isaaclab_arena_env.task
        names = task.names
        mouth_height = float(task.geom["mouth_height"])
        cc = load_controllers_cfg()
        self._pick = ScriptedPick(names, cc["ee_control"], cc["pick"])
        self._ins = InsertionController(
            names, mouth_height, cc["ee_control"], cc["insertion"], grip_offset=float(cc["pick"]["grip_offset"])
        )
        n, dev = env.unwrapped.num_envs, env.unwrapped.device
        self._in_insertion = torch.zeros(n, dtype=torch.bool, device=dev)
        from isaaclab_arena.controllers.budget_gate import BudgetGate

        _g = BudgetGate()
        self._gate = _g if _g.enabled else None

    # ------------------------------------------------------------------ act
    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        self._ensure(env)
        if self._m7_offset is not None:
            self._ins.set_m7(self._m7_offset, self._m7_tilt, recover=self._m7_recover)
        pick_action, pick_finished = self._pick.step(env)
        # latch into insertion once the pick reports finished (the handoff)
        if self._gate is not None:  # precision-budget gate: score the handoff against the back-end tolerance
            newly = pick_finished & ~self._in_insertion
            if bool(newly.any()):
                task = env.unwrapped.cfg.isaaclab_arena_env.task
                go, tilt_e, e_mm, dz = self._gate.evaluate(env, task.names["peg_finger_sensor"])
                for i in newly.nonzero(as_tuple=True)[0].tolist():
                    print(
                        f"[budgetgate] pol=scripted ep={getattr(self, '_gd_ep', 0)} env={i} "
                        f"tilt_est={float(tilt_e[i]):.1f}deg e_est={float(e_mm[i]):.1f}mm "
                        f"budget={self._gate.budget_mm:.1f}mm decision={'GO' if bool(go[i]) else 'NOGO'} "
                        f"dz={float(dz[i]):.2f}N",
                        flush=True,
                    )
                if self._gate.active and bool((newly & ~go).any()):
                    self._ins._decant = True  # correct-on-no-go: de-cant straightens the cocked grasp in place
        self._in_insertion = self._in_insertion | pick_finished
        if os.environ.get("VDASH_GRASP_DIAG"):  # grasp-quality diag (mirror of vdash_vla_policy)
            import math

            from isaaclab.utils.math import quat_apply

            from isaaclab_arena.controllers.ee_control import read_ee_pose

            task = env.unwrapped.cfg.isaaclab_arena_env.task
            pegd = env.unwrapped.scene[task.names["peg"]].data
            robot = env.unwrapped.scene[task.names["robot"]]
            ep, eq = read_ee_pose(env)
            ee_pos, ee_quat = ep[0], eq[0]
            zloc = torch.tensor([0.0, 0.0, 1.0], device=ee_quat.device)
            peg_axis = quat_apply(pegd.root_quat_w[0], zloc)
            ee_axis = quat_apply(ee_quat, zloc)
            ingrip_tilt = math.degrees(math.acos(float(torch.dot(peg_axis, ee_axis).abs().clamp(max=1.0))))
            rel = pegd.root_pos_w[0, :3] - ee_pos
            lat_mm = float(torch.norm(rel - torch.dot(rel, ee_axis) * ee_axis)) * 1000.0
            peg_speed = float(torch.norm(pegd.root_lin_vel_w[0]))
            grip = float(robot.data.joint_pos[0, -2:].abs().sum())
            if not getattr(self, "_gd_handoff_logged", False) and bool(self._in_insertion[0]):
                self._gd_handoff_logged = True
                print(
                    f"[graspdiag] pol=scripted ep={getattr(self, '_gd_ep', 0)} HANDOFF "
                    f"ingrip_tilt={ingrip_tilt:.1f}deg lat={lat_mm:.1f}mm peg_speed={peg_speed:.3f} "
                    f"grip={grip:.3f}",
                    flush=True,
                )
                # tilt-observability probe: is the in-gripper cock readable from the per-finger contact
                # forces (§2.1-legal: gripper's own force sensing, not peg pose)? Log per-finger nets.
                if os.environ.get("VDASH_FINGER_DIAG"):
                    try:
                        fm = env.unwrapped.scene[task.names["peg_finger_sensor"]].data.force_matrix_w[0]  # (B,M,3)
                        ff = fm.reshape(-1, fm.shape[-1]) if fm.dim() == 2 else fm.sum(dim=0)  # (M,3) per-finger net
                        f0, f1 = ff[0], ff[1]
                        print(
                            "[fingerdiag]"
                            f" gt_tilt={ingrip_tilt:.1f} gt_lat={lat_mm:.1f} f0=({f0[0]:.2f},{f0[1]:.2f},{f0[2]:.2f})"
                            f" f1=({f1[0]:.2f},{f1[1]:.2f},{f1[2]:.2f})"
                            f" d_xy=({float(f0[0] - f1[0]):.2f},{float(f0[1] - f1[1]):.2f})"
                            f" dz={float(f0[2] - f1[2]):.2f}",
                            flush=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"[fingerdiag] err {e}", flush=True)
        ins_action = self._ins.step(env, active=self._in_insertion)
        if os.environ.get("VDASH_BACKSWITCH") and getattr(self._ins, "gave_up", None) is not None:
            self._in_insertion = self._in_insertion & ~self._ins.gave_up  # back-switch: return to front-end
        action = torch.where(self._in_insertion.unsqueeze(-1), ins_action, pick_action)

        if self._debug and (self._step % 15 == 0):
            self._dbg(env)
        self._step += 1
        return action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._pick is None:
            return
        self._pick.reset(env_ids)
        self._ins.reset(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        self._in_insertion[idx] = False
        self._step = 0
        if os.environ.get("VDASH_GRASP_DIAG"):
            self._gd_ep = getattr(self, "_gd_ep", 0) + 1
            self._gd_handoff_logged = False

    # ------------------------------------------------------------------ debug
    def _dbg(self, env):
        task = env.unwrapped.cfg.isaaclab_arena_env.task
        peg = env.unwrapped.scene[task.names["peg"]].data
        sock = env.unwrapped.scene[task.names["socket"]].data
        import math

        from isaaclab.utils.math import quat_apply

        from isaaclab_arena.controllers.ee_control import read_ee_pose

        tip = peg.root_pos_w[0]
        mouth_z = sock.root_pos_w[0, 2] + float(task.geom["mouth_height"])
        depth = (mouth_z - tip[2]).item()
        lateral = torch.norm(tip[:2] - sock.root_pos_w[0, :2]).item()
        f = torch.norm(env.unwrapped.scene[task.names["peg_sensor"]].data.net_forces_w[0]).item()
        ee = read_ee_pose(env)[0][0]
        gw = (env.unwrapped.scene[task.names["robot"]].data.joint_pos[0, -2:].abs().sum()).item()
        # peg tilt = angle between the peg axis (local +z) and world +z
        zloc = torch.tensor([0.0, 0.0, 1.0], device=peg.root_quat_w.device)
        axis_w = quat_apply(peg.root_quat_w[0], zloc)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(axis_w[2].item())))))
        print(
            f"[VDASH_DEBUG] step={self._step} pick_ph={int(self._pick.phase[0])} "
            f"ins_ph={int(self._ins.phase[0])} in_ins={int(self._in_insertion[0])} "
            f"ee=({ee[0].item():.3f},{ee[1].item():.3f},{ee[2].item():.3f}) "
            f"tip_z={tip[2].item():.4f} depth={depth * 1000:.1f}mm lat={lateral * 1000:.2f}mm "
            f"tilt={tilt:.1f}deg grip={gw * 1000:.1f}mm f={f:.1f}N",
            flush=True,
        )

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> VDashScriptedPolicy:
        return VDashScriptedPolicy(VDashScriptedPolicyArgs.from_cli_args(args))
