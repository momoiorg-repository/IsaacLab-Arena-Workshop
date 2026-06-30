# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Precision-budget handoff gate (V-DASH RSJ): make the VLA→rule-based switch *budget-aware*.

The naive V-DASH switch latches into the insertion back-end as soon as the §3.4 handoff predicate
fires (``grasped ∧ tip-in-region ∧ low-speed``) — it never checks whether the back-end can actually
succeed from the *current grasp*. This module turns the precision budget into the switching policy:
at the handoff moment it estimates the (otherwise blind) in-gripper grasp error from **observable
signals only** and decides go / no-go against the back-end's measured tolerance.

§2.1 compliance: the estimate uses only the gripper's own per-finger contact forces (the de-cant cue
``dz = f0_z - f1_z ∝ in-gripper cock``, validated ≈0.75 N/deg) and, optionally, a tactile-probe
lateral estimate — never the peg's ground-truth pose.

Budget model: an in-gripper tilt ``φ`` displaces the (blind) tip by ``L·sin φ`` (lever L≈60 mm from
grip to tip), so the effective tip error is ``e = l + L·sin φ``. The back-end's measured convergence
basin tolerates ``e ≲ budget`` (≈5 mm at c=2.0; §3 basin). Gate: hand off iff ``e_est ≤ budget``.

Modes (env ``VDASH_BUDGET_GATE``): unset → off (headline path unchanged); ``log`` → evaluate + log
the decision but still hand off (lets us score decision-vs-outcome without perturbing dynamics);
``active`` → on a no-go, invoke the in-place de-cant correction before handing off (closed loop).
"""

from __future__ import annotations

import math
import os
import torch


class BudgetGate:
    """Estimate the in-gripper grasp error from observable signals and gate the handoff on the budget."""

    def __init__(self):
        mode = (os.environ.get("VDASH_BUDGET_GATE", "") or "").lower()
        self.enabled = mode in ("log", "active", "correct", "1", "on", "true")
        self.active = mode in ("active", "correct")
        self.gain = float(os.environ.get("VDASH_DECANT_GAIN", "0.7") or "0.7")  # finger dz (N) per deg
        self.L_mm = float(os.environ.get("VDASH_GATE_L_MM", "60") or "60")  # grip->tip lever (mm)
        self.budget_mm = float(os.environ.get("VDASH_GATE_BUDGET_MM", "5") or "5")  # back-end tolerance (mm)

    def _tilt_from_fingers(self, env, sensor_name: str):
        """Observable in-gripper tilt estimate (deg) and raw dz (N) from per-finger force-z asymmetry."""
        try:
            fm = env.unwrapped.scene[sensor_name].data.force_matrix_w  # (N,B,M,3)
            ff = fm.sum(dim=1)  # (N,M,3) per-finger net force
            if ff.shape[1] < 2:
                return None, None
            dz = ff[:, 0, 2] - ff[:, 1, 2]  # (N,) ∝ cock about the finger-perp axis
        except Exception:  # noqa: BLE001
            return None, None
        tilt_deg = dz.abs() / max(self.gain, 1e-6)
        return tilt_deg, dz

    def evaluate(self, env, sensor_name: str, lat_est_mm: torch.Tensor | None = None):
        """Return (go_mask (N,bool), tilt_est_deg (N), e_est_mm (N), dz (N)). go = effective error in budget."""
        n, dev = env.unwrapped.num_envs, env.unwrapped.device
        tilt_deg, dz = self._tilt_from_fingers(env, sensor_name)
        if tilt_deg is None:  # no finger sensor -> fail open (behave like the naive switch)
            ones = torch.ones(n, dtype=torch.bool, device=dev)
            z = torch.zeros(n, device=dev)
            return ones, z, z, z
        lat = lat_est_mm if lat_est_mm is not None else torch.zeros_like(tilt_deg)
        e_mm = lat + self.L_mm * torch.sin(tilt_deg * (math.pi / 180.0))  # effective tip error
        go = e_mm <= self.budget_mm
        return go, tilt_deg, e_mm, dz
