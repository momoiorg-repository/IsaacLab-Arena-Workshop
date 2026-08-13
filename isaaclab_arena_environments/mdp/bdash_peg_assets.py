# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH parametric peg / socket assets, registered into the Arena asset registry.

Points at the locally generated USDs (``assets/bdash/peg_insert/``, produced by
``scripts/bdash/generate_assets.py``, M1). The peg is a dynamic rigid body (held in the gripper);
each ``socket_c{c}`` is a kinematic fixed body (bolted to the table). Importing this module
registers the assets; the B-DASH environment imports it so ``asset_registry.get_asset_by_name``
can resolve ``bdash_peg`` / ``bdash_socket_c{...}``.
"""

import copy
import os

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_utils import RIGID_BODY_PROPS_HIGH_PRECISION
from isaaclab_arena.assets.register import register_asset

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BDASH_ASSET_DIR = os.path.join(_REPO_ROOT, "assets", "bdash", "peg_insert")

# Kinematic socket: immovable on the table, unaffected by contact.
_KINEMATIC = {"rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)}

# M3b — SDF force-spike physics fix (brief §3). The peg's high-precision rigid props default to
# max_depenetration_velocity=5.0 m/s; against the stiff 1024-res socket SDF an axis-misaligned
# initial condition can resolve a deep penetration in one step into a large velocity / force spike
# (M1 known risk). Cap the peg's depenetration velocity to 1.0 m/s so PhysX bleeds penetrations out
# gently. This is a bdash-LOCAL override (deepcopy) — we must NOT mutate the shared, frozen
# RIGID_BODY_PROPS_HIGH_PRECISION, which other Arena assets reuse (§2.2). Recorded in
# docs/physics_params.md per §2.3.
_PEG_RIGID_PROPS = copy.deepcopy(RIGID_BODY_PROPS_HIGH_PRECISION)
_PEG_RIGID_PROPS.max_depenetration_velocity = 1.0


@register_asset
class BDashPeg(LibraryObject):
    """Ø8 mm peg with C0.5 tip and 20 mm grip (held asset). Dynamic rigid body."""

    name = "bdash_peg"
    tags = ["object", "bdash"]
    usd_path = os.path.join(BDASH_ASSET_DIR, "peg.usd")
    object_type = ObjectType.RIGID
    # M3b: depenetration-capped rigid props (see above).
    # M8: bind a saturated blue visual material so the classic-vision baseline can color-segment the
    # peg. This deliberately makes the grasp perception fragile to scene diversity (it breaks at
    # L2 lighting / L3 texture+distractors) — that breakage is the E3 comparison point (brief §3 M8).
    spawn_cfg_addon = {
        "rigid_props": _PEG_RIGID_PROPS,
        "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.25, 0.95), roughness=0.5),
        "visual_material_path": "material",
    }


class _BDashSocket(LibraryObject):
    """Base for the parametric sockets (kinematic fixed asset)."""

    tags = ["object", "bdash", "fixed"]
    object_type = ObjectType.RIGID
    spawn_cfg_addon = _KINEMATIC
    clearance_mm: float = 0.0


@register_asset
class BDashSocketC2(_BDashSocket):
    name = "bdash_socket_c2"
    usd_path = os.path.join(BDASH_ASSET_DIR, "socket_c2.usd")
    clearance_mm = 2.0


@register_asset
class BDashSocketC1(_BDashSocket):
    name = "bdash_socket_c1"
    usd_path = os.path.join(BDASH_ASSET_DIR, "socket_c1.usd")
    clearance_mm = 1.0


# Intermediate clearances (RSJ2026 clearance-cliff sweep): fill 2.0 -> 1.0.
@register_asset
class BDashSocketC1p75(_BDashSocket):
    name = "bdash_socket_c1p75"
    usd_path = os.path.join(BDASH_ASSET_DIR, "socket_c1p75.usd")
    clearance_mm = 1.75


@register_asset
class BDashSocketC1p5(_BDashSocket):
    name = "bdash_socket_c1p5"
    usd_path = os.path.join(BDASH_ASSET_DIR, "socket_c1p5.usd")
    clearance_mm = 1.5


@register_asset
class BDashSocketC1p25(_BDashSocket):
    name = "bdash_socket_c1p25"
    usd_path = os.path.join(BDASH_ASSET_DIR, "socket_c1p25.usd")
    clearance_mm = 1.25


@register_asset
class BDashSocketC0p5(_BDashSocket):
    name = "bdash_socket_c0p5"
    usd_path = os.path.join(BDASH_ASSET_DIR, "socket_c0p5.usd")
    clearance_mm = 0.5


@register_asset
class BDashSocketC0p25(_BDashSocket):
    name = "bdash_socket_c0p25"
    usd_path = os.path.join(BDASH_ASSET_DIR, "socket_c0p25.usd")
    clearance_mm = 0.25


# clearance (mm) -> registered socket asset name, for the env / eval grid
BDASH_SOCKET_BY_CLEARANCE = {
    2.0: "bdash_socket_c2",
    1.75: "bdash_socket_c1p75",
    1.5: "bdash_socket_c1p5",
    1.25: "bdash_socket_c1p25",
    1.0: "bdash_socket_c1",
    0.5: "bdash_socket_c0p5",
    0.25: "bdash_socket_c0p25",
}
