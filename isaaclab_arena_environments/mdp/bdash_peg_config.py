# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Loader for the externalized B-DASH thresholds/geometry (``configs/bdash/peg_insert/task.yaml``).

All §3.4 predicate thresholds, §3.5 logging geometry, and episode timing live in YAML so nothing
is hard-coded in the task/predicate/logger code (brief cross-cutting requirement). This module just
reads the YAML into a plain dict; consumers pull the sub-sections they need.
"""

from __future__ import annotations

import os
import yaml
from functools import lru_cache

# repo root = .../IsaacLab-Arena-An  (this file is isaaclab_arena_environments/mdp/bdash_peg_config.py)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BDASH_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs", "bdash", "peg_insert")
TASK_CFG_PATH = os.path.join(BDASH_CONFIG_DIR, "task.yaml")
CONTROLLERS_CFG_PATH = os.path.join(BDASH_CONFIG_DIR, "controllers.yaml")
ASSETS_CFG_PATH = os.path.join(BDASH_CONFIG_DIR, "assets.yaml")


@lru_cache(maxsize=8)
def load_task_cfg(path: str | None = None) -> dict:
    """Read ``configs/bdash/peg_insert/task.yaml`` (or an override path) into a dict. Cached per-path."""
    cfg_path = path or TASK_CFG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=8)
def load_controllers_cfg(path: str | None = None) -> dict:
    """Read ``configs/bdash/peg_insert/controllers.yaml`` (scripted-pick + insertion gains). Cached per-path."""
    cfg_path = path or CONTROLLERS_CFG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f)
