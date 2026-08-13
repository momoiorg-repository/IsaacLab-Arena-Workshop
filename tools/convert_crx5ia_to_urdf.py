#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Convert CRX5iA + Robotiq 140 xacro to a flattened URDF with relative mesh paths.

This script:
  1. Uses xacro to process the combined .urdf.xacro
  2. Rewrites all package:// mesh URIs to relative file paths (relative to the URDF location)
  3. Saves the result to isaaclab_arena/embodiments/crx5ia/usd/crx5ia_robotiq140.urdf

Relative paths ensure the URDF works both on the host and inside the Isaac Sim Docker container,
since the repo directory structure is the same regardless of where it's mounted.

Usage (from the repo root, with ROS2 workspace sourced):
  source ros2_ws/install/setup.bash
  python tools/convert_crx5ia_to_urdf.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

# Paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
XACRO_DIR = REPO_ROOT / "isaaclab_arena" / "embodiments" / "crx5ia" / "usd"
XACRO_FILES = [
    XACRO_DIR / "crx5ia.urdf.xacro",
    XACRO_DIR / "robotiq_140.urdf.xacro",
    XACRO_DIR / "robotiq_85.urdf.xacro",
    XACRO_DIR / "crx5ia_robotiq85.urdf.xacro",
]

# ROS2 source directories for mesh resolution (real files, not symlinks in install/)
PACKAGE_DIRS = {
    "fanuc_crx_description": REPO_ROOT / "ros2_ws" / "src" / "fanuc_description" / "fanuc_crx_description",
    "robotiq_description": REPO_ROOT / "ros2_ws" / "src" / "ros2_robotiq_gripper" / "robotiq_description",
}


def resolve_package_uri(match: re.Match, urdf_dir: Path) -> str:
    """Replace package://pkg_name/path with a relative file path from the URDF location."""
    pkg_name = match.group(1)
    rel_path = match.group(2)
    if pkg_name in PACKAGE_DIRS:
        abs_path = PACKAGE_DIRS[pkg_name] / rel_path
        if not abs_path.exists():
            print(f"  WARNING: mesh not found: {abs_path}", file=sys.stderr)
        # Compute relative path from the URDF output directory to the mesh file
        try:
            relative = os.path.relpath(abs_path, urdf_dir)
        except ValueError:
            relative = str(abs_path)
        return f'filename="{relative}"'
    else:
        print(f"  WARNING: unknown package '{pkg_name}' — leaving URI as-is", file=sys.stderr)
        return match.group(0)


def process_xacro(xacro_file: Path):
    output_urdf = xacro_file.with_suffix("").with_suffix(".urdf")
    urdf_dir = output_urdf.parent

    print(f"\nProcessing xacro: {xacro_file}")
    result = subprocess.run(
        ["xacro", str(xacro_file)],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    if result.returncode != 0:
        print(f"ERROR: xacro failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    urdf_xml = result.stdout

    print("Rewriting mesh paths to relative paths (from URDF location)...")
    pattern = r'filename="package://([^/]+)/([^"]+)"'
    # Use lambda to pass urdf_dir
    urdf_xml = re.sub(pattern, lambda m: resolve_package_uri(m, urdf_dir), urdf_xml)

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    output_urdf.write_text(urdf_xml)
    print(f"URDF written to: {output_urdf}")
    print(f"File size: {output_urdf.stat().st_size} bytes")

    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(str(output_urdf))
        root = tree.getroot()
        links = root.findall(".//link")
        joints = root.findall(".//joint")
        print(f"Validation OK — {len(links)} links, {len(joints)} joints")
    except Exception as e:
        print(f"WARNING: XML validation failed: {e}", file=sys.stderr)


def main():
    try:
        subprocess.run(["xacro", "--help"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: 'xacro' not found. Please install it or source ROS2:", file=sys.stderr)
        sys.exit(1)

    for xf in XACRO_FILES:
        process_xacro(xf)


if __name__ == "__main__":
    main()
