# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH parametric chuck-loading asset generation.

Builds the multi-variant machine-tending scene: three yaw-symmetric workpieces, a vertical
3-jaw power chuck (body + one jaw, spawned three times at 120 deg), a coarse tray, a
touch-off datum block, and a V-block for re-erecting side-lying stock.

Pipeline is the same as ``scripts/vdash/generate_assets.py``: trimesh CSG (manifold engine)
-> STL (Z-up) -> Isaac Lab ``MeshConverter`` -> USD. ``MeshConverter`` wraps
omni.kit.asset_converter, so the Simulation App must be running; hence the AppLauncher
boilerplate. ``frustum``/``to_usd``/``_mesh_collision_cfg`` are duplicated from the vdash
generator rather than imported, because that module runs argparse + AppLauncher at import
time and cannot be imported as a library.

Design notes that are load-bearing:
  * Every workpiece is a body of revolution => yaw-symmetric => the grasp stays position-only
    on the yaw axis. A yaw-relevant grip was previously found unlearnable (corr(yaw, wrist)
    = -0.93 -> 0% grasp); see build_peg in the vdash generator.
  * Jaws are KINEMATIC (env closes them, then a fixed joint holds the part), so there is no
    articulation and no URDF step. The jaw is emitted as its own mesh.
  * Jaws recess flush into the chuck face so the top face stays a clean seating annulus for
    the W-C flange.

Example::

    docker exec -u an isaaclab_arena-cuda_gr00t_gn17 bash -lc '
      cd /workspaces/isaaclab_arena && unset DISPLAY &&
      /isaac-sim/python.sh scripts/bdash/generate_chuck_assets.py --headless'
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate B-DASH chuck-loading USDs.")
parser.add_argument("--config", type=str, default="configs/bdash/assets.yaml")
parser.add_argument("--out", type=str, default=None, help="Output dir (overrides config output_dir).")
parser.add_argument(
    "--only",
    type=str,
    nargs="*",
    default=None,
    help="Subset to build: workpieces chuck jaw tray touchoff vblock (default: all).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import trimesh

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def load_config(path: str) -> dict:
    if yaml is None or not os.path.isfile(path):
        raise FileNotFoundError(f"B-DASH asset config required but unreadable: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("workpieces", "chuck", "chuck_jaw", "tray", "touchoff_block", "vblock", "collision", "mesh"):
        if key not in cfg:
            raise KeyError(f"missing required section {key!r} in {path}")
    return cfg


# --------------------------------------------------------------------------------------
# Mesh primitives
# --------------------------------------------------------------------------------------
def frustum(r_bottom: float, r_top: float, height: float, z0: float = 0.0, sections: int = 96) -> trimesh.Trimesh:
    """A capped conical frustum along +Z, from (r_bottom @ z0) to (r_top @ z0+height)."""
    ang = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    bottom = np.c_[r_bottom * np.cos(ang), r_bottom * np.sin(ang), np.full(sections, z0)]
    top = np.c_[r_top * np.cos(ang), r_top * np.sin(ang), np.full(sections, z0 + height)]
    cb = np.array([[0.0, 0.0, z0]])
    ct = np.array([[0.0, 0.0, z0 + height]])
    verts = np.vstack([bottom, top, cb, ct])
    ib, it = 2 * sections, 2 * sections + 1
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces += [[i, j, sections + j], [i, sections + j, sections + i]]  # side
        faces.append([ib, j, i])  # bottom cap
        faces.append([it, sections + i, sections + j])  # top cap
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)


def _box(extents, center) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b


# --------------------------------------------------------------------------------------
# Workpieces
# --------------------------------------------------------------------------------------
def build_workpiece(spec: dict, chamfer: float, sections: int) -> trimesh.Trimesh:
    """Stack ``sections`` [(diameter, length), ...] along +Z from z=0 (the LEADING end that
    enters the chuck), chamfering both free ends so the stock reads as deburred and gets a
    lead-in at the jaws."""
    secs = spec["sections"]
    parts: list[trimesh.Trimesh] = []
    z = 0.0
    for i, (dia, length) in enumerate(secs):
        r = dia / 2.0
        first, last = i == 0, i == len(secs) - 1
        z0, h = z, length
        if first and chamfer > 0.0:
            parts.append(frustum(max(r - chamfer, 1e-4), r, chamfer, z0=z0, sections=sections))
            z0 += chamfer
            h -= chamfer
        if last and chamfer > 0.0:
            h -= chamfer
        if h <= 0.0:
            raise ValueError(f"section {i} of length {length} is too short for chamfer {chamfer}")
        cyl = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
        cyl.apply_translation([0.0, 0.0, z0 + h / 2.0])
        parts.append(cyl)
        if last and chamfer > 0.0:
            parts.append(frustum(r, max(r - chamfer, 1e-4), chamfer, z0=z0 + h, sections=sections))
        z += length
    return trimesh.boolean.union(parts, engine="manifold")


# --------------------------------------------------------------------------------------
# Chuck
# --------------------------------------------------------------------------------------
def build_chuck_body(c: dict, jaw: dict, sections: int) -> trimesh.Trimesh:
    """Chuck body: cylinder with a through-bore and three radial jaw slots cut into the top
    face at 120 deg. The slots let the jaws sit flush, leaving the top face a clean annulus
    that the W-C flange can seat against."""
    radius, height = c["body_diameter"] / 2.0, c["body_height"]
    bore_r = c["bore_diameter"] / 2.0
    slack = c.get("jaw_slot_clearance", 0.001)

    body = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    body.apply_translation([0.0, 0.0, height / 2.0])

    bore = trimesh.creation.cylinder(radius=bore_r, height=height + 0.004, sections=sections)
    bore.apply_translation([0.0, 0.0, height / 2.0])
    tools = [bore]

    # Radial slots: wide enough for the jaw + slack, deep enough for it to sit flush, and long
    # enough to span from inside the bore out past the jaw's retracted position.
    slot_w = jaw["width"] + 2.0 * slack
    slot_d = jaw["height"] + slack
    slot_len = radius  # from the axis outward; the inner end is open into the bore
    for k in range(3):
        slot = _box([slot_len, slot_w, slot_d], [slot_len / 2.0, 0.0, height - slot_d / 2.0])
        slot.apply_transform(trimesh.transformations.rotation_matrix(k * 2.0 * np.pi / 3.0, [0, 0, 1]))
        tools.append(slot)

    return trimesh.boolean.difference([body, trimesh.boolean.union(tools, engine="manifold")], engine="manifold")


def build_chuck_jaw(j: dict) -> trimesh.Trimesh:
    """One soft jaw (生爪) with a stepped gripping face.

    Local frame: the gripping face is at x=0 and the body extends toward +x (radially
    outward); z=0 is the jaw bottom. The upper portion is set back by
    (radius_high - radius_low) so the lower step grips Ø25 and the upper step grips Ø32.
    The environment places it at radius = radius_low and rotates by k*120 deg.
    """
    length, width, height = j["radial_length"], j["width"], j["height"]
    step_h = j["step_height"]
    setback = j["radius_high"] - j["radius_low"]
    if not 0.0 < step_h < height:
        raise ValueError(f"step_height {step_h} must lie inside (0, height={height})")
    if setback <= 0.0:
        raise ValueError("radius_high must exceed radius_low for a stepped jaw face")

    block = _box([length, width, height], [length / 2.0, 0.0, height / 2.0])
    # remove the inner sliver above the step -> the Ø32 relief
    relief = _box(
        [setback, width + 0.002, height - step_h],
        [setback / 2.0, 0.0, step_h + (height - step_h) / 2.0],
    )
    return trimesh.boolean.difference([block, relief], engine="manifold")


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
def build_tray(t: dict) -> trimesh.Trimesh:
    """Open tray with no dividers: outer box minus an inner cavity."""
    ox, oy, oz = t["outer"]
    wall, floor = t["wall"], t["floor"]
    outer = _box([ox, oy, oz], [0.0, 0.0, oz / 2.0])
    cavity = _box(
        [ox - 2.0 * wall, oy - 2.0 * wall, oz - floor + 0.002],
        [0.0, 0.0, floor + (oz - floor + 0.002) / 2.0],
    )
    return trimesh.boolean.difference([outer, cavity], engine="manifold")


def build_touchoff_block(b: dict) -> trimesh.Trimesh:
    """Plain datum block. The touch-off skill probes its top face plus two orthogonal sides,
    which is what turns a tip estimate into a measured ê (lateral XY + protrusion)."""
    sx, sy, sz = b["size"]
    return _box([sx, sy, sz], [0.0, 0.0, sz / 2.0])


def build_vblock(v: dict, sections: int) -> trimesh.Trimesh:
    """Block with a 90 deg V-groove running along X. A side-lying cylinder dropped in the
    groove self-centers and its axis becomes known, which is what makes the 仮置き route able
    to re-erect stock the chuck cannot accept directly."""
    sx, sy, sz = v["size"]
    depth = v["groove_depth"]
    half = np.deg2rad(v["groove_angle_deg"]) / 2.0
    # A groove of depth d with half-angle a is d*tan(a) wide at each side of the centreline.
    half_w = depth * np.tan(half)
    if 2.0 * half_w > sy:
        raise ValueError(f"V-groove ({2 * half_w:.3f} m wide) does not fit block width {sy} m")

    block = _box([sx, sy, sz], [0.0, 0.0, sz / 2.0])
    # Cutting tool: a triangular prism along X, apex down at z = sz - depth.
    verts = np.array([
        [-sx / 2.0 - 0.001, 0.0, sz - depth],
        [-sx / 2.0 - 0.001, -half_w, sz + 0.002],
        [-sx / 2.0 - 0.001, half_w, sz + 0.002],
        [sx / 2.0 + 0.001, 0.0, sz - depth],
        [sx / 2.0 + 0.001, -half_w, sz + 0.002],
        [sx / 2.0 + 0.001, half_w, sz + 0.002],
    ])
    faces = np.array([[0, 1, 2], [3, 5, 4], [0, 2, 5], [0, 5, 3], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2]])
    groove = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return trimesh.boolean.difference([block, groove], engine="manifold")


# --------------------------------------------------------------------------------------
# USD conversion (mirrors scripts/vdash/generate_assets.py)
# --------------------------------------------------------------------------------------
def _mesh_collision_cfg(approximation: str, sdf_resolution: int):
    if approximation == "sdf":
        return schemas_cfg.SDFMeshPropertiesCfg(sdf_resolution=sdf_resolution)
    if approximation in ("convexDecomposition", "convex_decomposition"):
        return schemas_cfg.ConvexDecompositionPropertiesCfg()
    if approximation in ("convexHull", "convex_hull"):
        return schemas_cfg.ConvexHullPropertiesCfg()
    raise ValueError(f"Unsupported collision approximation: {approximation!r}")


def to_usd(mesh: trimesh.Trimesh, out_dir: str, name: str, approximation: str, mass: float, coll: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stl_path = os.path.join(out_dir, f"{name}.stl")
    mesh.export(stl_path)  # STL => Z-up assumed by MeshConverter
    cfg = MeshConverterCfg(
        asset_path=os.path.abspath(stl_path),
        usd_dir=os.path.abspath(out_dir),
        usd_file_name=f"{name}.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        mass_props=schemas_cfg.MassPropertiesCfg(mass=mass),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        collision_props=schemas_cfg.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=coll["contact_offset"], rest_offset=coll["rest_offset"]
        ),
        mesh_collision_props=_mesh_collision_cfg(approximation, int(coll.get("sdf_resolution", 256))),
    )
    conv = MeshConverter(cfg)
    print(
        f"[gen] {name}: watertight={mesh.is_watertight} verts={len(mesh.vertices)} "
        f"bounds={np.round(mesh.bounds, 4).tolist()} approx={approximation} -> {conv.usd_path}"
    )
    return conv.usd_path


def main():
    cfg = load_config(args_cli.config)
    out_dir = args_cli.out or cfg["output_dir"]
    sections = cfg["mesh"]["circle_sections"]
    coll = cfg["collision"]
    want = set(args_cli.only) if args_cli.only else {"workpieces", "chuck", "jaw", "tray", "touchoff", "vblock"}

    if "workpieces" in want:
        chamfer = cfg["workpiece_common"]["end_chamfer"]
        for wid, spec in cfg["workpieces"].items():
            mesh = build_workpiece(spec, chamfer, sections)
            name = f"workpiece_{wid.replace('-', '').lower()}"  # W-A -> workpiece_wa
            to_usd(mesh, out_dir, name, coll["workpiece_approximation"], spec["mass"], coll)

    if "chuck" in want:
        body = build_chuck_body(cfg["chuck"], cfg["chuck_jaw"], sections)
        to_usd(body, out_dir, "chuck_body", coll["chuck_body_approximation"], cfg["chuck"]["mass"], coll)

    if "jaw" in want:
        jaw = build_chuck_jaw(cfg["chuck_jaw"])
        to_usd(jaw, out_dir, "chuck_jaw", coll["jaw_approximation"], cfg["chuck_jaw"]["mass"], coll)

    if "tray" in want:
        tray = build_tray(cfg["tray"])
        to_usd(tray, out_dir, "tray", coll["tray_approximation"], cfg["tray"]["mass"], coll)

    if "touchoff" in want:
        blk = build_touchoff_block(cfg["touchoff_block"])
        to_usd(blk, out_dir, "touchoff_block", coll["block_approximation"], cfg["touchoff_block"]["mass"], coll)

    if "vblock" in want:
        vb = build_vblock(cfg["vblock"], sections)
        to_usd(vb, out_dir, "vblock", coll["vblock_approximation"], cfg["vblock"]["mass"], coll)

    print("[gen] DONE")


if __name__ == "__main__":
    main()
    simulation_app.close()
