# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH chuck-loading assets, registered into the Arena asset registry.

Points at the locally generated USDs (``assets/bdash/chuck_load/``, produced by
``scripts/bdash/generate_chuck_assets.py``). Importing this module registers the assets; the
B-DASH environment imports it so ``asset_registry.get_asset_by_name`` can resolve
``bdash_workpiece_w{a,b,c}``, ``bdash_chuck_body``, ``bdash_chuck_jaw``, ``bdash_tray``,
``bdash_touchoff_block`` and ``bdash_vblock``.

Dynamics split:
  * workpieces are dynamic rigid bodies (picked up, so they must fall and collide)
  * every fixture is kinematic (bolted down; unaffected by contact)
  * jaws are kinematic too — the environment drives them closed and then attaches the held
    workpiece with a fixed joint, which is why no articulation or URDF is needed

Geometry lives in ``configs/bdash/chuck_load/assets.yaml`` and is re-read here rather than duplicated, so
the registry cannot drift from the meshes that were actually generated.
"""

import copy
import os
import yaml

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_utils import RIGID_BODY_PROPS_HIGH_PRECISION
from isaaclab_arena.assets.register import register_asset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BDASH_ASSET_DIR = os.path.join(_REPO_ROOT, "assets", "bdash", "chuck_load")
BDASH_ASSET_CONFIG = os.path.join(_REPO_ROOT, "configs", "bdash", "chuck_load", "assets.yaml")

# Kinematic fixtures: immovable, unaffected by contact.
_KINEMATIC = {"rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)}

# Same rationale as the bdash peg (docs/physics_params.md): the default high-precision props use
# max_depenetration_velocity=5.0, which against a stiff SDF fixture can turn a one-step deep
# penetration into a force spike. Cap it so PhysX bleeds penetrations out gently. Deepcopy because
# RIGID_BODY_PROPS_HIGH_PRECISION is shared and frozen — it must not be mutated in place.
_WORKPIECE_RIGID_PROPS = copy.deepcopy(RIGID_BODY_PROPS_HIGH_PRECISION)
_WORKPIECE_RIGID_PROPS.max_depenetration_velocity = 1.0


def _load_asset_config() -> dict:
    with open(BDASH_ASSET_CONFIG) as f:
        return yaml.safe_load(f) or {}


_CFG = _load_asset_config()


def _usd(name: str) -> str:
    return os.path.join(BDASH_ASSET_DIR, f"{name}.usd")


# --------------------------------------------------------------------------------------
# Workpieces (dynamic)
# --------------------------------------------------------------------------------------
class _BDashWorkpiece(LibraryObject):
    """Base for the three workpiece variants (dynamic rigid bodies).

    ``variant_id`` keys back into configs/bdash/chuck_load/assets.yaml, so per-variant tolerances and
    section geometry are looked up rather than restated here. Distinct diffuse colours make the
    variants separable both on video and to a colour-segmentation baseline; the VLA itself is
    conditioned on an ER 2 point plus an estimated axis, not on colour.
    """

    tags = ["object", "bdash", "workpiece"]
    object_type = ObjectType.RIGID
    variant_id: str = ""

    @classmethod
    def sections(cls) -> list[list[float]]:
        return _CFG["workpieces"][cls.variant_id]["sections"]

    @classmethod
    def total_length(cls) -> float:
        return sum(length for _, length in cls.sections())

    @classmethod
    def max_diameter(cls) -> float:
        return max(dia for dia, _ in cls.sections())

    @classmethod
    def tolerance(cls) -> dict:
        return _CFG["workpieces"][cls.variant_id]["tolerance"]


def _workpiece_spawn(color: tuple[float, float, float]) -> dict:
    return {
        "rigid_props": _WORKPIECE_RIGID_PROPS,
        "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.4, metallic=0.1),
        "visual_material_path": "material",
    }


@register_asset
class BDashWorkpieceWA(_BDashWorkpiece):
    """W-A: plain Ø25 x 90 cylinder. No axial stop — protrusion is pure insertion-depth error."""

    name = "bdash_workpiece_wa"
    variant_id = "W-A"
    usd_path = _usd("workpiece_wa")
    spawn_cfg_addon = _workpiece_spawn((0.05, 0.25, 0.95))


@register_asset
class BDashWorkpieceWB(_BDashWorkpiece):
    """W-B: stepped Ø25/Ø32 x 100. The Ø32 shoulder is the reference protrusion is measured to;
    it clears the Ø40 bore, so it is a datum and not a stop."""

    name = "bdash_workpiece_wb"
    variant_id = "W-B"
    usd_path = _usd("workpiece_wb")
    spawn_cfg_addon = _workpiece_spawn((0.95, 0.45, 0.05))


@register_asset
class BDashWorkpieceWC(_BDashWorkpiece):
    """W-C: flanged Ø25 x 80 with a Ø45 flange. The flange cannot enter the Ø40 bore, so it
    hard-seats on the chuck face — the clearest seating signal of the three."""

    name = "bdash_workpiece_wc"
    variant_id = "W-C"
    usd_path = _usd("workpiece_wc")
    spawn_cfg_addon = _workpiece_spawn((0.10, 0.75, 0.30))


# --------------------------------------------------------------------------------------
# Chuck + fixtures (kinematic)
# --------------------------------------------------------------------------------------
class _BDashFixture(LibraryObject):
    tags = ["object", "bdash", "fixed"]
    object_type = ObjectType.RIGID
    spawn_cfg_addon = _KINEMATIC


@register_asset
class BDashChuckBody(_BDashFixture):
    """Ø160 x 80 chuck body with a Ø40 through-bore and three flush radial jaw slots."""

    name = "bdash_chuck_body"
    usd_path = _usd("chuck_body")


@register_asset
class BDashChuckJaw(_BDashFixture):
    """One soft jaw. Spawned three times at 120 deg and driven radially by the environment."""

    name = "bdash_chuck_jaw"
    usd_path = _usd("chuck_jaw")


@register_asset
class BDashTray(_BDashFixture):
    """Coarse open tray, no dividers. Workpieces are scattered in it at random pose."""

    name = "bdash_tray"
    usd_path = _usd("tray")


@register_asset
class BDashTouchoffBlock(_BDashFixture):
    """Datum block. Probing its top face plus two orthogonal sides is what turns a nominal tip
    estimate into a measured e-hat (lateral XY + protrusion) — the gate's input signal."""

    name = "bdash_touchoff_block"
    usd_path = _usd("touchoff_block")


@register_asset
class BDashVBlock(_BDashFixture):
    """90 deg V-block. A side-lying cylinder set down here self-centres and its axis becomes
    known, which is what lets the 仮置き route re-erect stock the chuck cannot take directly."""

    name = "bdash_vblock"
    usd_path = _usd("vblock")


# --------------------------------------------------------------------------------------
# Lookup tables for the environment / eval grid
# --------------------------------------------------------------------------------------
BDASH_WORKPIECE_BY_VARIANT = {
    "W-A": "bdash_workpiece_wa",
    "W-B": "bdash_workpiece_wb",
    "W-C": "bdash_workpiece_wc",
}

BDASH_VARIANTS = tuple(BDASH_WORKPIECE_BY_VARIANT)


def chuck_geometry() -> dict:
    """Chuck dimensions the predicates and the loading skill both need."""
    c = _CFG["chuck"]
    j = _CFG["chuck_jaw"]
    return {
        "body_diameter": c["body_diameter"],
        "body_height": c["body_height"],
        "bore_diameter": c["bore_diameter"],
        "face_z": c["body_height"],  # seating plane for the W-C flange
        "jaw_radius_low": j["radius_low"],
        "jaw_radius_high": j["radius_high"],
        "jaw_height": j["height"],
        "jaw_step_height": j["step_height"],
    }


def workpiece_tolerance(variant: str) -> dict:
    """Per-variant acceptance window. Provisional until the R(g) sweep fills it in."""
    return _CFG["workpieces"][variant]["tolerance"]
