# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic contract checks for B-DASH skill assignment.

The planner (organizer) delegates every numeric decision to this module:
given a front-end skill's measured residual-error distribution and a
back-end skill's measured tolerance, decide whether the handoff is within
budget and estimate the expected success. VLMs never judge numbers.
"""

from __future__ import annotations

import math
import yaml
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LIBRARY = Path(__file__).parent / "skill_library.yaml"


def load_library(path: str | Path = DEFAULT_LIBRARY) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["skills"]


def _normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2.0))))


@dataclass
class ContractReport:
    front: str
    back: str
    feasible: bool
    within_budget_prob: float
    expected_success: float
    reasons: list[str] = field(default_factory=list)


def check_handoff(
    front_spec: dict,
    back_spec: dict,
    clearance_mm: float | None = None,
    front_name: str = "front",
    back_name: str = "back",
) -> ContractReport:
    """Check front residual error against back tolerance (blind budget)."""
    reasons: list[str] = []
    tol = back_spec["tolerance"]

    # Environment constraint: clearance cliff of the back-end.
    env_factor = 1.0
    if clearance_mm is not None:
        env = tol.get("environment", {})
        if clearance_mm < env.get("clearance_min_mm", 0.0):
            return ContractReport(
                front_name,
                back_name,
                False,
                0.0,
                0.0,
                [f"clearance {clearance_mm} mm below back-end limit {env['clearance_min_mm']} mm"],
            )
        if clearance_mm < env.get("clearance_ok_mm", 0.0):
            env_factor = 0.75  # measured: 75 % at c=1.5 mm with a clean grasp
            reasons.append(f"clearance {clearance_mm} mm marginal (env factor {env_factor})")

    # Blind budget. Validated on the bench (paper Sec. 5(3)): the lateral-only
    # model reproduces the measured end-to-end rate (20.8 % vs 21 %), while the
    # effective-tip model e = lateral + L*sin(tilt) over-penalizes tilt. Use
    # lateral-only for the expected-success estimate; keep effective-tip as a
    # conservative reference.
    res = front_spec["residual_error"]
    lat, tilt = res["grasp_lateral_mm"], res["grasp_tilt_deg"]
    budget = tol["blind"]["effective_tip_mm"]
    p_within = _normal_cdf(budget, lat["mean"], lat["std"])
    reasons.append(
        f"grasp lateral ~N({lat['mean']:.1f}, {lat['std']:.1f}) mm vs budget {budget} mm "
        f"-> P(within)={p_within:.2f} (lateral-only model, bench-validated)"
    )
    lever = tol["blind"].get("tip_lever_arm_mm", 60.0)
    e_mean = lat["mean"] + lever * math.sin(math.radians(tilt["mean"]))
    e_std = math.hypot(lat["std"], lever * math.radians(tilt["std"]))
    p_conservative = _normal_cdf(budget, e_mean, e_std)
    reasons.append(f"conservative effective-tip model: P(within)={p_conservative:.2f}")

    completion = front_spec.get("completion_rate", 1.0)
    expected = completion * p_within * env_factor
    feasible = p_within >= 0.5 and expected > 0.0
    reasons.append(
        f"expected end-to-end success ~{expected:.2f} "
        f"(completion {completion} x within {p_within:.2f} x env {env_factor})"
    )
    return ContractReport(front_name, back_name, feasible, p_within, expected, reasons)


def rank_frontends(
    library: dict, back_name: str, clearance_mm: float | None = None, capability: str = "grasp"
) -> list[ContractReport]:
    """Rank all front-end candidates for a given back-end skill."""
    back = library[back_name]
    reports = []
    for name, spec in library.items():
        if spec.get("type") != "front" or capability not in spec.get("capabilities", []):
            continue
        reports.append(check_handoff(spec, back, clearance_mm, name, back_name))
    return sorted(reports, key=lambda r: r.expected_success, reverse=True)
