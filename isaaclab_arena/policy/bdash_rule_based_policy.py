# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""B-DASH "rule-based 単体" baseline policy (brief §3 M8).

Same scripted pick → rule-based insertion as ``bdash_scripted``, except the **grasp point comes from
classic vision** (top-down colour segmentation + known-plane back-projection,
:class:`~isaaclab_arena.controllers.bdash_classic_vision.ClassicVisionGrasp`) instead of the
ground-truth peg pose — per brief §2.1 this baseline's grasp phase may not use the simulator's true
peg pose. The insertion phase is unchanged (socket pose + wrist-F/T model, both allowed).

The policy first holds still for a few steps while it perceives the peg (the camera needs a settled
render), caches the averaged grasp estimate into ``ScriptedPick.grasp_override``, then runs the pick.
A failed segmentation (e.g. under L2 lighting / L3 texture) yields a bad/zero estimate and the grasp
misses — that degradation is the E3 comparison point. Needs ``--enable_cameras`` (top-down cam).

Registered as ``bdash_rule_based``. Run single-env (the non-tiled overhead camera renders one env)."""

from __future__ import annotations

import argparse
import gymnasium as gym
import os
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.controllers.bdash_classic_vision import ClassicVisionGrasp
from isaaclab_arena.controllers.insertion_controller import InsertionController
from isaaclab_arena.controllers.scripted_pick import ScriptedPick
from isaaclab_arena.policy.policy_base import PolicyBase


@dataclass
class BDashRuleBasedPolicyArgs:
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> BDashRuleBasedPolicyArgs:
        return cls()


@register_policy
class BDashRuleBasedPolicy(PolicyBase):

    name = "bdash_rule_based"
    config_class = BDashRuleBasedPolicyArgs

    def __init__(self, config: BDashRuleBasedPolicyArgs):
        super().__init__(config)
        self._pick: ScriptedPick | None = None
        self._ins: InsertionController | None = None
        self._vision: ClassicVisionGrasp | None = None
        self._in_insertion: torch.Tensor | None = None
        self._grasp_sum: torch.Tensor | None = None
        self._est_count: torch.Tensor | None = None
        self._vision_steps = int(os.environ.get("BDASH_VISION_STEPS", "8"))
        self._settled = False
        self._step = 0
        self._adim = 7

    def _ensure(self, env):
        if self._pick is not None:
            return
        from isaaclab_arena_environments.mdp.bdash_peg_config import load_controllers_cfg

        task = env.unwrapped.cfg.isaaclab_arena_env.task
        names = task.names
        mouth_height = float(task.geom["mouth_height"])
        cc = load_controllers_cfg()
        grip_offset = float(cc["pick"]["grip_offset"])
        self._pick = ScriptedPick(names, cc["ee_control"], cc["pick"])
        self._ins = InsertionController(names, mouth_height, cc["ee_control"], cc["insertion"], grip_offset=grip_offset)
        self._vision = ClassicVisionGrasp("overhead_cam")
        n, dev = env.unwrapped.num_envs, env.unwrapped.device
        self._in_insertion = torch.zeros(n, dtype=torch.bool, device=dev)
        self._grasp_sum = torch.zeros(n, 3, device=dev)
        self._est_count = torch.zeros(n, device=dev)
        try:
            self._adim = int(env.action_space.shape[-1])
        except Exception:
            self._adim = 7

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        self._ensure(env)
        n, dev = env.unwrapped.num_envs, env.unwrapped.device

        # --- perception settle: hold still and average the classic-vision grasp estimate ---
        if not self._settled:
            est, valid = self._vision.estimate(env)
            self._grasp_sum = torch.where(valid.unsqueeze(-1), self._grasp_sum + est, self._grasp_sum)
            self._est_count = self._est_count + valid.float()
            self._step += 1
            if self._step >= self._vision_steps:
                cnt = self._est_count.clamp(min=1.0).unsqueeze(-1)
                self._pick.grasp_override = self._grasp_sum / cnt  # never-valid envs -> 0 (grasp misses)
                self._settled = True
            return torch.zeros(n, self._adim, device=dev)  # hold while perceiving

        # --- scripted pick (vision grasp) -> rule-based insertion ---
        pick_action, pick_finished = self._pick.step(env)
        self._in_insertion = self._in_insertion | pick_finished
        ins_action = self._ins.step(env, active=self._in_insertion)
        return torch.where(self._in_insertion.unsqueeze(-1), ins_action, pick_action)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._pick is None:
            return
        self._pick.reset(env_ids)
        self._ins.reset(env_ids)
        self._pick.grasp_override = None
        # single-env baseline: re-perceive from scratch on reset
        self._in_insertion[:] = False
        self._grasp_sum[:] = 0.0
        self._est_count[:] = 0.0
        self._settled = False
        self._step = 0

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> BDashRuleBasedPolicy:
        return BDashRuleBasedPolicy(BDashRuleBasedPolicyArgs.from_cli_args(args))
