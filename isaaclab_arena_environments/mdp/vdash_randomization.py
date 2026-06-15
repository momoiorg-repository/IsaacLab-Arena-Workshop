# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""V-DASH scene-diversity randomization profiles L0–L3 (brief §3.6 / scene-diversity sweep).

The scene-diversity sweep is one of the paper's two core axes: as the scene gets harder the
rule-based insertion controller is expected to *collapse* (it has no perception), while the VLA
front-end is expected to absorb the variation. The four levels:

  * **L0** — fully fixed: socket + peg at canonical poses, no randomization. Used for the predicate /
    logging integration check.
  * **L1** — pose randomization: socket & peg planar position ±``pos`` and yaw ±π.
  * **L2** — L1 + lighting randomization (dome-light intensity + color jitter at reset).
  * **L3** — L2 + visual texture randomization + ``n`` distractor objects.

This module exposes :func:`profile` returning a :class:`LevelProfile` (pose range, optional lighting
/ texture event terms, distractor count). The environment threads the pose range and event terms
into :class:`~isaaclab_arena.tasks.vdash_pick_insert_task.VDashPickInsertTask` and adds the distractor
assets to the scene. Ranges are overridable from ``configs/vdash`` but carry sane defaults here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from isaaclab.managers import EventTermCfg

LEVELS = ("L0", "L1", "L2", "L3")

# default planar pose jitter (meters) and yaw range used from L1 onward
_DEFAULT_POS = 0.10
_DEFAULT_YAW = math.pi
# workspace center (env-frame, on the table in front of the Franka). sample_object_poses adds
# env_origins, so the pose range is absolute — it must sit in the arm's reach, NOT at the origin.
_WS_CENTER = (0.5, 0.0)
# minimum planar separation between the sampled socket and peg (m) so they don't spawn overlapping
_MIN_SEPARATION = 0.12


# --------------------------------------------------------------------------------------
# reset-time lighting randomizer (operates on the single global dome-light prim)
# --------------------------------------------------------------------------------------
def randomize_dome_light(
    env,
    env_ids,
    prim_path: str = "/World/Light",
    intensity_range: tuple[float, float] = (800.0, 3000.0),
    color_jitter: tuple[float, float] = (0.6, 1.0),
):
    """Reset event: jitter the dome light's intensity and (greyscale) color.

    The Arena dome light lives at a single global prim (``DomeLight.default_prim_path``), so we set
    the USD attributes once per reset rather than per-env. Best-effort: guarded so a missing light or
    USD API change never breaks the episode loop.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdLux

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        light = UsdLux.DomeLight(prim)
        intensity = random.uniform(*intensity_range)
        c = random.uniform(*color_jitter)
        light.GetIntensityAttr().Set(float(intensity))
        light.GetColorAttr().Set(Gf.Vec3f(c, c, c))
    except Exception:
        # randomization is non-critical; never let it abort a reset
        pass


@dataclass
class LevelProfile:
    """Resolved randomization knobs for one diversity level."""

    level: str
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    min_separation: float = 0.0
    lighting_event: EventTermCfg | None = None
    texture_event: EventTermCfg | None = None
    num_distractors: int = 0


def _pose_range(pos: float, yaw: float) -> dict[str, tuple[float, float]]:
    """Absolute env-frame workspace box centered on :data:`_WS_CENTER` (within the Franka's reach),
    with ±``pos`` planar jitter and ±``yaw`` heading; assets stay upright (roll/pitch = 0)."""
    cx, cy = _WS_CENTER
    return {"x": (cx - pos, cx + pos), "y": (cy - pos, cy + pos), "z": (0.0, 0.0), "yaw": (-yaw, yaw)}


def profile(
    level: str,
    *,
    pos: float = _DEFAULT_POS,
    yaw: float = _DEFAULT_YAW,
    intensity_range: tuple[float, float] = (800.0, 3000.0),
    color_jitter: tuple[float, float] = (0.6, 1.0),
    num_distractors: int = 3,
    texture_event: EventTermCfg | None = None,
) -> LevelProfile:
    """Build the randomization profile for ``level`` ∈ {L0, L1, L2, L3}.

    ``texture_event`` is passed through for L3 only when the caller supplies a configured
    ``randomize_visual_texture_material`` term (it needs concrete texture files); when ``None`` the
    L3 visual domain still varies via lighting + distractors.
    """
    level = level.upper()
    if level not in LEVELS:
        raise ValueError(f"Unknown diversity level {level!r}; expected one of {LEVELS}")

    if level == "L0":
        return LevelProfile(level=level)

    pose_range = _pose_range(pos, yaw)
    if level == "L1":
        return LevelProfile(level=level, pose_range=pose_range, min_separation=_MIN_SEPARATION)

    lighting_event = EventTermCfg(
        func=randomize_dome_light,
        mode="reset",
        params={"intensity_range": intensity_range, "color_jitter": color_jitter},
    )
    if level == "L2":
        return LevelProfile(
            level=level, pose_range=pose_range, min_separation=_MIN_SEPARATION, lighting_event=lighting_event
        )

    # L3
    return LevelProfile(
        level=level,
        pose_range=pose_range,
        min_separation=_MIN_SEPARATION,
        lighting_event=lighting_event,
        texture_event=texture_event,
        num_distractors=num_distractors,
    )
