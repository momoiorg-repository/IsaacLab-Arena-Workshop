# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Loader for the externalized B-DASH scene layout and thresholds (``configs/bdash/*.yaml``).

Mirrors :mod:`isaaclab_arena_environments.mdp.vdash_config`: scene poses, tray spawn ranges,
predicate thresholds and episode timing live in YAML so nothing is hard-coded in the environment,
task, predicate or controller code. This module only reads YAML into plain dicts; consumers pull
the sub-sections they need.
"""

from __future__ import annotations

import os
import yaml
from functools import lru_cache

# repo root = .../IsaacLab-Arena-An  (this file is isaaclab_arena_environments/mdp/bdash_config.py)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BDASH_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs", "bdash")
TASK_CFG_PATH = os.path.join(BDASH_CONFIG_DIR, "task.yaml")
ASSETS_CFG_PATH = os.path.join(BDASH_CONFIG_DIR, "assets.yaml")
CONTROLLERS_CFG_PATH = os.path.join(BDASH_CONFIG_DIR, "controllers.yaml")


@lru_cache(maxsize=8)
def load_task_cfg(path: str | None = None) -> dict:
    """Read ``configs/bdash/task.yaml`` (or an override path) into a dict. Cached per-path."""
    cfg_path = path or TASK_CFG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=8)
def load_assets_cfg(path: str | None = None) -> dict:
    """Read ``configs/bdash/assets.yaml`` (workpiece/chuck geometry). Cached per-path."""
    cfg_path = path or ASSETS_CFG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=8)
def load_controllers_cfg(path: str | None = None) -> dict:
    """Read ``configs/bdash/controllers.yaml`` (pick / touch-off / chuck-load gains).

    Written once the classical skills land; returns an empty dict until then so callers can rely
    on the same accessor throughout.
    """
    cfg_path = path or CONTROLLERS_CFG_PATH
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}
