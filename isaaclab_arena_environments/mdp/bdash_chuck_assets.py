# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH chuck-loading assets, registered into the Arena asset registry.

Points at the locally generated USDs (``assets/bdash/chuck_load/``, produced by
``scripts/bdash/generate_chuck_assets.py``). Importing this module registers the assets; the
B-DASH environment imports it so ``asset_registry.get_asset_by_name`` can resolve
``bdash_workpiece_w{a,b,c}``, ``bdash_chuck_body``, ``bdash_chuck_jaw``, ``bdash_tray``,
``bdash_reerect_pad`` (which is also the touch-off datum).

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
# Rolling resistance. These parts are cylinders, and a cylinder resting on a flat tray floor is in
# NEUTRAL equilibrium against rolling: friction lets it roll without slipping, so any impulse -- even
# the one from resolving initial contact -- sends it rolling and nothing ever stops it. Measured
# without this, workpieces set down at rest drifted 50 mm within half a second and climbed on top of
# one another. PhysX rigid bodies have no rolling-friction term, so angular damping is the stand-in
# for it; real steel stock in a steel tray does not roll away when you set it down. It only acts on
# free bodies, so a gripped part is unaffected.
_WORKPIECE_RIGID_PROPS.angular_damping = 5.0
_WORKPIECE_RIGID_PROPS.linear_damping = 0.5


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
    """Spawn addon for a workpiece.

    ``color`` is only the SPAWN-TIME placeholder. The per-episode reset event
    (:func:`~isaaclab_arena_environments.mdp.bdash_chuck_materials.randomize_workpiece_appearance`)
    overwrites diffuse/roughness/metallic on every reset, sampling hue INDEPENDENTLY of the
    variant. Until 2026-08-16 these were three fixed, per-variant colours -- W-A blue, W-B orange,
    W-C green -- which made the colour a perfect family label and let a policy name the family from
    one pixel without looking at the geometry at all. The placeholders are kept distinct only so a
    scene built with randomisation disabled is still legible to a human.

    ``visual_material_path`` must stay set: it is what gives each workpiece its OWN material prim
    rather than a shared one, and randomising a shared material would recolour all three parts
    together.
    """
    return {
        "rigid_props": _WORKPIECE_RIGID_PROPS,
        "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5, metallic=0.2),
        "visual_material_path": "material",
    }


def _fixture_spawn(key: str) -> dict:
    """Spawn addon for a FIXED fixture material, read from materials.yaml.

    Fixtures are deliberately not randomised: they are the cell rather than the parts, so a policy
    should be able to rely on them, and randomising the tray would only make it harder to find for
    no transfer benefit.
    """
    from isaaclab_arena_environments.mdp.bdash_chuck_config import load_materials_cfg
    from isaaclab_arena_environments.mdp.bdash_chuck_materials import preview_surface_kwargs

    entry = load_materials_cfg()["fixtures"][key]
    return {
        **_KINEMATIC,
        "visual_material": sim_utils.PreviewSurfaceCfg(**preview_surface_kwargs(entry)),
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
    spawn_cfg_addon = _fixture_spawn("chuck_body")


@register_asset
class BDashChuckJaw(_BDashFixture):
    """One soft jaw. Spawned three times at 120 deg and driven radially by the environment."""

    name = "bdash_chuck_jaw"
    usd_path = _usd("chuck_jaw")
    spawn_cfg_addon = _fixture_spawn("chuck_jaw")


@register_asset
class BDashTray(_BDashFixture):
    """Coarse open tray, no dividers. Workpieces are scattered in it at random pose."""

    name = "bdash_tray"
    usd_path = _usd("tray")
    spawn_cfg_addon = _fixture_spawn("tray")


@register_asset
class BDashReerectPad(_BDashFixture):
    """spec §4-2 仮置き台: the raised face a turned-up part is stood on before it is re-picked.

    Raised rather than the bare bench because the arm measurably cannot reach the bench with the
    tool horizontal -- and holding a vertical part is exactly what makes the tool horizontal. The
    hand bottoms out 13.9-16.6 mm short at every wrist azimuth tried, with joint travel to spare, so
    it is the hand on the bench and not the arm running out of reach. See
    ``configs/bdash/chuck_load/assets.yaml`` for the numbers.
    """

    name = "bdash_reerect_pad"
    usd_path = _usd("reerect_pad")
    spawn_cfg_addon = _fixture_spawn("reerect_pad")


@register_asset
class BDashWorkbench(_BDashFixture):
    """The work surface everything else stands on. Top face at z = 0.

    Generated (scripts/bdash/generate_chuck_assets.py build_workbench) rather than taken from the
    background library: the stock SeattleLabTable's collision top is only 0.91 m wide in y, and that
    one number is what forced the tray to be yawed and the chuck round behind the robot's shoulder.
    The two other stock candidates were measured and rejected -- `packing_table` is a whole scene
    carrying its own 5001 m ground plane, and `maple_table_robolab` fails to load its vMaterials MDL
    modules and renders broken.
    """

    name = "bdash_workbench"
    usd_path = _usd("workbench")
    spawn_cfg_addon = _fixture_spawn("workbench")


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


def workpiece_datum_offset(variant: str) -> float:
    """Axial position (local +Z, from the leading end) of the feature protrusion is measured to.

    Derived from ``sections`` rather than restated, so it cannot drift from the mesh:

    In every variant this is the end of the leading Ø25 section: on W-A (a single section) that is
    the trailing end face, since it has no stop and protrusion is purely whatever was commanded; on
    W-B it is the Ø32 shoulder (measured, and small enough to pass into the Ø40 bore); on W-C it is
    the flange underside, which cannot enter the bore and hard-seats on the chuck face.
    """
    return float(_CFG["workpieces"][variant]["sections"][0][1])


def workpiece_profile(variant: str) -> dict:
    """Sim-free geometry summary of one variant, for the tray layout sampler and the predicates.

    ``sections`` is returned as ``(radius, length)`` pairs stacked from the leading end, which is
    the form :func:`~isaaclab_arena_environments.mdp.bdash_chuck_randomization.resting_pose` solves
    the lying-down pose from. Nothing here touches Isaac Sim, but this module does, so callers that
    must stay sim-free take the returned dict as a parameter instead of importing it.
    """
    sections = _CFG["workpieces"][variant]["sections"]
    return {
        "variant": variant,
        "sections": [(float(d) / 2.0, float(length)) for d, length in sections],
        "length": float(sum(length for _, length in sections)),
        "max_radius": float(max(d for d, _ in sections)) / 2.0,
        "lead_radius": float(sections[0][0]) / 2.0,
        "datum_offset": workpiece_datum_offset(variant),
    }
