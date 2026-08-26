# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for the B-DASH chuck-loading predicates.

Builds a fake env (plain torch tensors, no Isaac Sim) with 4 envs and K=3 workpieces, one scenario
per env, and asserts each predicate fires only where expected. Run fast with:
    /isaac-sim/python.sh -m pytest isaaclab_arena_environments/tests/test_bdash_chuck_predicates.py
or standalone:
    /isaac-sim/python.sh isaaclab_arena_environments/tests/test_bdash_chuck_predicates.py
"""

import importlib.util
import math
import os
import torch

# Load the predicate module directly (bypassing isaaclab_arena_environments.mdp.__init__, which
# imports Isaac Sim). The predicates are torch-only, so this keeps the test sim-free.
_spec = importlib.util.spec_from_file_location(
    "bdash_chuck_predicates", os.path.join(os.path.dirname(__file__), "..", "mdp", "bdash_chuck_predicates.py")
)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

# --- scene constants mirroring configs/bdash/chuck_load/{task,assets}.yaml -------------------
WP = ("wp_0", "wp_1", "wp_2")  # stands for W-A, W-B, W-C
FINGER_SENSORS = tuple(f"wp_finger_contact_{k}" for k in range(3))
CONTACT_SENSORS = tuple(f"wp_contact_{k}" for k in range(3))
TRAY_XY, CHUCK_XY = (0.45, 0.17), (0.52, -0.16)
RIM_HEIGHT, FACE_HEIGHT = 0.030, 0.080
DATUM_OFFSETS = (0.090, 0.060, 0.072)  # W-A trailing end / W-B Ø32 shoulder / W-C flange underside
INSERT_DEPTHS = (0.050, 0.040, 0.072)
NOMINALS = tuple(d - i for d, i in zip(DATUM_OFFSETS, INSERT_DEPTHS))  # 0.040 / 0.020 / 0.000
TOLERANCES = (0.002, 0.0015, 0.002)
SEAT_ANGLE_RAD = tuple(math.radians(d) for d in (2.0, 2.0, 1.0))

GRASP_KW = dict(finger_sensors=FINGER_SENSORS, width_min=0.020, width_max=0.048, grasp_force=1.0)
LIFT_KW = dict(workpiece_names=WP, tray_name="tray", rim_height=RIM_HEIGHT, height=0.060, speed_max=0.08, **GRASP_KW)
SEAT_KW = dict(
    workpiece_names=WP,
    chuck_name="chuck",
    face_height=FACE_HEIGHT,
    seat_angle_rad=SEAT_ANGLE_RAD,
    min_insert_depth=0.030,
    bore_clearance=0.0075,
)
PROT_KW = dict(workpiece_names=WP, chuck_name="chuck", face_height=FACE_HEIGHT, datum_offsets=DATUM_OFFSETS)

# A workpiece seated to nominal depth has its leading end this far above the chuck root.
SEATED_TIP_Z = FACE_HEIGHT - INSERT_DEPTHS[0]  # 0.030
IDENT = (1.0, 0.0, 0.0, 0.0)
FLIPPED = (0.0, 1.0, 0.0, 0.0)  # 180 deg about +X -> local +Z maps to world -Z


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
    def __init__(self, scene, target_idx=None):
        self.scene = scene
        if target_idx is not None:
            self.bdash_target_idx = torch.tensor(target_idx, dtype=torch.long)


def _build_env(target_idx=(0, 1, 0, 0)):
    """4 envs x 3 workpieces.

    env0: target wp0 grasped but still down in the tray      -> grasped, NOT lifted
    env1: target wp1 grasped and 80 mm clear of the rim      -> grasped and lifted
    env2: target wp0 seated in the chuck, gripper released   -> seated, protrusion at nominal
    env3: target wp0 at the same depth but INVERTED, jamming -> NOT seated, force violation
    """
    n, k = 4, len(WP)
    tx, ty = TRAY_XY
    cx, cy = CHUCK_XY

    # default: every workpiece parked at a distinct spot resting in the tray
    pos = torch.tensor([[[tx - 0.06, ty, 0.0175], [tx, ty + 0.04, 0.0175], [tx + 0.06, ty, 0.0175]]]).repeat(n, 1, 1)
    quat = torch.tensor([IDENT]).repeat(n, k, 1)

    pos[0, 0] = torch.tensor([tx - 0.06, ty, 0.020])  # env0: held, barely off the tray floor
    pos[1, 1] = torch.tensor([tx, ty + 0.04, RIM_HEIGHT + 0.080])  # env1: 80 mm clear of the rim
    pos[2, 0] = torch.tensor([cx + 0.001, cy, SEATED_TIP_Z])  # env2: in the bore, 1 mm off axis
    pos[3, 0] = torch.tensor([cx + 0.001, cy, SEATED_TIP_Z])  # env3: same place...
    quat[3, 0] = torch.tensor(FLIPPED)  # ...but upside down

    vel = torch.zeros(n, k, 3)

    # Franka: 7 arm + 2 finger joints. Closed on a Ø25 shaft -> width 0.025; released -> 0.080.
    fingers = torch.tensor([[0.0125, 0.0125], [0.0125, 0.0125], [0.040, 0.040], [0.040, 0.040]])
    joint_pos = torch.cat([torch.zeros(n, 7), fingers], dim=1)

    # per-workpiece filtered finger contact (N, B=1, M=2 fingers, 3); only the held part is loaded
    finger_force = torch.zeros(n, k, 1, 2, 3)
    finger_force[0, 0, 0, :, 2] = 5.0
    finger_force[1, 1, 0, :, 2] = 5.0

    # per-workpiece net contact force (N, B=1, 3); env3's target is jamming in the bore
    net_force = torch.zeros(n, k, 1, 3)
    net_force[:, :, 0, 2] = 1.0
    net_force[3, 0, 0, 2] = 80.0

    entities = {
        "tray": _Entity(root_pos_w=torch.tensor([[tx, ty, 0.0]]).repeat(n, 1)),
        "chuck": _Entity(root_pos_w=torch.tensor([[cx, cy, 0.0]]).repeat(n, 1)),
        "robot": _Entity(joint_pos=joint_pos),
    }
    for i, name in enumerate(WP):
        entities[name] = _Entity(root_pos_w=pos[:, i], root_quat_w=quat[:, i], root_lin_vel_w=vel[:, i])
        entities[FINGER_SENSORS[i]] = _Entity(force_matrix_w=finger_force[:, i])
        entities[CONTACT_SENSORS[i]] = _Entity(net_forces_w=net_force[:, i])
    return _Env(_Scene(entities), target_idx)


def _check(name, got, exp):
    exp_t = torch.tensor(exp)
    assert torch.equal(got, exp_t), f"{name}: got {got.tolist()} expected {exp}"


def _close(name, got, exp, tol=1e-6):
    exp_t = torch.tensor(exp, dtype=got.dtype)
    assert torch.allclose(got, exp_t, atol=tol), f"{name}: got {got.tolist()} expected {exp}"


def test_pick_stage_predicates():
    env = _build_env()
    _check("grasped", P.grasped(env, **GRASP_KW), [True, True, False, False])
    _check("lifted", P.lifted(env, **LIFT_KW), [False, True, False, False])
    _check("dropped", P.dropped(env, workpiece_names=WP, min_z=-0.05), [False, False, False, False])
    _check("grasped_index", P.grasped_index(env, **GRASP_KW), [0, 1, -1, -1])


def test_load_stage_predicates():
    env = _build_env()
    # env3 is the sign-convention guard: same bore position and depth as env2, only inverted.
    _check("seated", P.seated(env, **SEAT_KW), [False, False, True, False])
    _check(
        "force_violation",
        P.force_violation(env, contact_sensors=CONTACT_SENSORS, force_max=60.0),
        [False, False, False, True],
    )
    # env2 sits exactly at the commanded depth, so its datum stands off by the nominal 40 mm.
    _close("protrusion_m", P.protrusion_m(env, **PROT_KW)[2], 0.040)
    _check(
        "protrusion_ok",
        P.protrusion_ok(env, nominals=NOMINALS, tolerances=TOLERANCES, **PROT_KW),
        [False, False, True, False],
    )


def test_target_index_actually_scopes_the_predicate():
    """Point every env at wp1: env0 holds wp0, so its grasp must stop counting."""
    env = _build_env(target_idx=(1, 1, 1, 1))
    _check("grasped@wp1", P.grasped(env, **GRASP_KW), [False, True, False, False])
    _check("seated@wp1", P.seated(env, **SEAT_KW), [False, False, False, False])


def test_any_fallback_without_a_selector():
    """With no latched target the predicates reduce over all workpieces instead of failing."""
    env = _build_env(target_idx=None)
    assert P.target_index(env) is None
    _check("grasped/any", P.grasped(env, **GRASP_KW), [True, True, False, False])
    _check("seated/any", P.seated(env, **SEAT_KW), [False, False, True, False])
    # protrusion has no meaningful `any`, so it falls back to the part nearest the bore axis
    _close("protrusion_m/any", P.protrusion_m(env, **PROT_KW)[2], 0.040)


if __name__ == "__main__":
    test_pick_stage_predicates()
    test_load_stage_predicates()
    test_target_index_actually_scopes_the_predicate()
    test_any_fallback_without_a_selector()
    print("BDASH_CHUCK_PREDICATES_OK")
