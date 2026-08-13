# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Sim-free unit tests for the B-DASH §3.4 predicates.

Builds a fake env (plain torch tensors, no Isaac Sim) with 4 envs, one per scenario, and asserts
each predicate fires only where expected. Run fast with:
    /isaac-sim/python.sh -m pytest isaaclab_arena_environments/tests/test_bdash_predicates.py
or standalone:
    /isaac-sim/python.sh isaaclab_arena_environments/tests/test_bdash_predicates.py
"""

import importlib.util
import os
import torch

# Load the predicate module directly (bypassing isaaclab_arena_environments.mdp.__init__, which
# imports Isaac Sim). The predicates are torch-only, so this keeps the test sim-free.
_spec = importlib.util.spec_from_file_location(
    "bdash_peg_predicates", os.path.join(os.path.dirname(__file__), "..", "mdp", "bdash_peg_predicates.py")
)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


class _Data:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Entity:
    def __init__(self, **kw):
        self.data = _Data(**kw)


class _Scene:
    def __init__(self, entities):
        self._e = entities

    def __getitem__(self, k):
        return self._e[k]


class _Env:
    def __init__(self, scene):
        self.scene = scene


def _build_env():
    """4 envs: 0=grasped/high, 1=handoff, 2=inserted(released), 3=force_violation/off-axis."""
    n = 4
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
    socket_pos = torch.tensor([[0.5, 0.0, 0.0]]).repeat(n, 1)  # mouth at z=0.030
    peg_pos = torch.tensor([
        [0.500, 0.100, 0.200],  # env0: grasped but high & lateral-far
        [0.505, 0.000, 0.070],  # env1: handoff cylinder (lateral 5mm, height 40mm)
        [0.5005, 0.000, 0.010],  # env2: inserted (depth 20mm, lateral 0.5mm)
        [0.510, 0.000, 0.005],  # env3: deep but lateral 10mm off-axis (NOT inserted)
    ])
    peg_vel = torch.zeros(n, 3)
    # Franka: 7 arm + 2 finger joints; gripper width = |j[-1]|+|j[-2]|
    fingers = torch.tensor([[0.010, 0.010], [0.010, 0.010], [0.035, 0.035], [0.010, 0.010]])
    joint_pos = torch.cat([torch.zeros(n, 7), fingers], dim=1)
    # peg–finger filtered contact force (N,B=1,M=2 fingers,3): grasped envs ~5N/finger; env2 released
    ff = torch.zeros(n, 1, 2, 3)
    for i in (0, 1, 3):
        ff[i, 0, :, 2] = 5.0
    # peg net contact force (N,B=1,3): env3 = 80N violation; others small
    nf = torch.zeros(n, 1, 3)
    nf[:, 0, 2] = 1.0
    nf[3, 0, 2] = 80.0

    scene = _Scene({
        "peg": _Entity(root_pos_w=peg_pos, root_quat_w=ident, root_lin_vel_w=peg_vel),
        "socket": _Entity(root_pos_w=socket_pos),
        "robot": _Entity(joint_pos=joint_pos),
        "peg_finger_contact": _Entity(force_matrix_w=ff),
        "peg_contact": _Entity(net_forces_w=nf),
    })
    return _Env(scene)


def _check(name, got, exp):
    exp_t = torch.tensor(exp)
    assert torch.equal(got, exp_t), f"{name}: got {got.tolist()} expected {exp}"


def test_predicates():
    env = _build_env()
    _check("grasped", P.grasped(env), [True, True, False, True])
    _check("handoff", P.handoff(env), [False, True, False, False])
    _check("inserted", P.inserted(env, clearance=0.002), [False, False, True, False])
    _check("force_violation", P.force_violation(env, force_max=50.0), [False, False, False, True])
    _check("insertion_failed", P.insertion_failed(env, force_max=50.0), [False, False, False, True])


if __name__ == "__main__":
    test_predicates()
    print("BDASH_PREDICATES_OK")
