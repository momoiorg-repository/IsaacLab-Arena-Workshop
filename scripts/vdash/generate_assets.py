# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""V-DASH parametric peg / socket asset generation (M1).

Builds the brief's peg-in-hole assets at absolute dimensions and a parametric radial
clearance ``c`` (mm), exporting one USD per asset with proper collision approximation:
  * socket_c{c}.usd  -> SDF mesh collision (a bore cannot be a convex hull)
  * peg.usd          -> convex decomposition (shaft + grip = 2 convex parts)

Pipeline: trimesh CSG (manifold engine) -> STL (Z-up) -> Isaac Lab ``MeshConverter`` -> USD.
``MeshConverter`` wraps omni.kit.asset_converter, so the Simulation App must be running;
hence the AppLauncher boilerplate. Run headless with ``unset DISPLAY`` (see docs/milestones/M0.md).

Example::

    docker exec isaaclab_arena-latest bash -lc '
      cd /workspaces/isaaclab_arena && unset DISPLAY &&
      /isaac-sim/python.sh scripts/vdash/generate_assets.py --headless'
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate V-DASH parametric peg/socket USDs.")
parser.add_argument("--config", type=str, default="configs/vdash/assets.yaml")
parser.add_argument("--out", type=str, default=None, help="Output dir (overrides config output_dir).")
parser.add_argument("--clearances", type=float, nargs="*", default=None, help="Radial clearances c (mm).")
parser.add_argument("--peg-only", action="store_true")
parser.add_argument("--socket-only", action="store_true")
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

# --------------------------------------------------------------------------------------
# Defaults (mirror configs/vdash/assets.yaml; used if yaml/file unavailable)
# --------------------------------------------------------------------------------------
DEFAULTS = {
    "peg": {"shaft_diameter": 0.008, "shaft_length": 0.050, "tip_chamfer": 0.0005, "grip_size": 0.020, "mass": 0.019},
    "socket": {"outer_xy": 0.060, "height": 0.030, "bore_depth": 0.025, "mouth_chamfer": 0.0005, "mass": 0.050},
    "clearances_mm": [2.0, 1.0, 0.5, 0.25],
    "collision": {
        "contact_offset": 0.005,
        "rest_offset": 0.0,
        "socket_approximation": "sdf",
        "peg_approximation": "convexDecomposition",
    },
    "mesh": {"circle_sections": 96},
    "output_dir": "assets/vdash",
}


def load_config(path: str) -> dict:
    cfg = {k: v.copy() if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
    if yaml is not None and os.path.isfile(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
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


def build_peg(p: dict, sections: int) -> trimesh.Trimesh:
    """Peg: tip chamfer frustum + shaft cylinder + **cylindrical grip**, tip at local z=0.

    The grip is a cylinder (diameter = height = ``grip_size``). A cylinder grip is **yaw-symmetric**,
    so the grasp does not depend on the peg's yaw — the pick stays a position-only skill the VLA can
    learn from vision. (A square grip made yaw grasp-relevant, which required the policy to perceive
    the peg's yaw from the camera views; that was found unlearnable from 200 demos -> ~0% grasp, with
    corr(peg_yaw, grasp_wrist) = -0.93. Reverted 2026-06-23.) Grip *center* stays at
    ``shaft_length + grip/2`` so the controller's ``grip_offset`` is unchanged."""
    r = p["shaft_diameter"] / 2.0
    ch = p["tip_chamfer"]
    shaft_len, grip = p["shaft_length"], p["grip_size"]
    tip = frustum(max(r - ch, 1e-4), r, ch, z0=0.0, sections=sections)
    shaft = trimesh.creation.cylinder(radius=r, height=shaft_len - ch, sections=sections)
    shaft.apply_translation([0, 0, ch + (shaft_len - ch) / 2.0])
    grip_cyl = trimesh.creation.cylinder(radius=grip / 2.0, height=grip, sections=sections)
    grip_cyl.apply_translation([0, 0, shaft_len + grip / 2.0])
    peg = trimesh.boolean.union([tip, shaft, grip_cyl], engine="manifold")
    return peg


def build_socket(s: dict, peg_r: float, c_m: float, sections: int) -> trimesh.Trimesh:
    """Socket: 60×60×30 box with blind bore (radius peg_r + c) + conical mouth countersink.
    Base at z=0, top (mouth) at z=height.

    The countersink is a funnel widening the bore opening from ``bore_r`` to ``bore_r +
    countersink_radius`` over the top ``countersink_depth``. A generous funnel (vs the cosmetic C0.5
    edge) provides the lead-in the rule-based insertion controller needs: a slightly off / lightly
    tilted tip rides the cone to center instead of point-contacting the flat top face and pitching
    the peg. The *bore clearance* ``c`` (the swept variable) is unchanged, so the deep insertion at
    tight clearance stays hard — only the entrance is funneled."""
    outer, h, depth = s["outer_xy"], s["height"], s["bore_depth"]
    cs_depth = s.get("countersink_depth", s["mouth_chamfer"])
    cs_radius = s.get("countersink_radius", s["mouth_chamfer"])
    bore_r = peg_r + c_m
    box = trimesh.creation.box(extents=[outer, outer, h])
    box.apply_translation([0, 0, h / 2.0])
    # bore: cut from the top, slightly over-tall to guarantee a clean opening
    bore = trimesh.creation.cylinder(radius=bore_r, height=depth + 0.002, sections=sections)
    bore.apply_translation([0, 0, h - depth + (depth + 0.002) / 2.0])
    # conical countersink funnel: bore_r (at z=h-cs_depth) -> bore_r+cs_radius (at the mouth z=h)
    cs_ring = frustum(bore_r, bore_r + cs_radius, cs_depth, z0=h - cs_depth, sections=sections)
    tool = trimesh.boolean.union([bore, cs_ring], engine="manifold")
    socket = trimesh.boolean.difference([box, tool], engine="manifold")
    return socket


# --------------------------------------------------------------------------------------
# USD conversion
# --------------------------------------------------------------------------------------
def _mesh_collision_cfg(approximation: str, sdf_resolution: int):
    """Map an approximation name to the proper MeshCollisionPropertiesCfg subclass.

    The base class leaves ``usd_func``/``physx_func`` = MISSING; only the subclasses wire the
    USD/PhysX API setters, so the subclass must be used (else: '_MISSING_TYPE' not callable).
    """
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
            max_depenetration_velocity=5.0,
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
        f"approx={approximation} -> {conv.usd_path}"
    )
    return conv.usd_path


def main():
    cfg = load_config(args_cli.config)
    out_dir = args_cli.out or cfg["output_dir"]
    clearances = args_cli.clearances if args_cli.clearances is not None else cfg["clearances_mm"]
    peg_r = cfg["peg"]["shaft_diameter"] / 2.0
    sections = cfg["mesh"]["circle_sections"]
    coll = cfg["collision"]

    if not args_cli.socket_only:
        peg = build_peg(cfg["peg"], sections)
        to_usd(peg, out_dir, "peg", coll["peg_approximation"], cfg["peg"]["mass"], coll)

    if not args_cli.peg_only:
        for c in clearances:
            c_m = c / 1000.0
            socket = build_socket(cfg["socket"], peg_r, c_m, sections)
            tag = f"{c:g}".replace(".", "p")
            to_usd(socket, out_dir, f"socket_c{tag}", coll["socket_approximation"], cfg["socket"]["mass"], coll)

    print("[gen] DONE")


if __name__ == "__main__":
    main()
    simulation_app.close()
