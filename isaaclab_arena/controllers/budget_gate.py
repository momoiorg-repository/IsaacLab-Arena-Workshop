# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""spec §5: the three-way budget gate. Route on a MEASURED error, or admit you did not measure.

This is the whole point of the architecture, so it is deliberately tiny and pure: a measured axial
grip error ``e`` against two per-variant thresholds.

* ``|e| <= go``     -> **GO**: insert directly. The error is inside what the insertion process can
  absorb while staying inside the QC protrusion window.
* ``|e| <= reject`` -> **REFIX**: the grip is wrong but the PART is fine -- stand it on the 仮置き台
  and re-grasp at the commanded station, which resets the error to process-nominal.
* otherwise         -> **REJECT**: the measurement itself says something is off beyond what a
  re-grasp fixes (part slipping, wrong part, probe artefact). Do not load it.

A probe that never made contact is REFIX, not GO: no measurement is not a good measurement, and the
refix route re-establishes a known grip without needing one.

The thresholds are per variant and are DERIVED, not chosen (§9 derived-window rule): the R(g) sweep
measures where insertion quality actually degrades and writes them back. The config carries
provisional values only so the plumbing can be exercised before the sweep lands.
"""

from __future__ import annotations

import torch

GO, REFIX, REJECT = 0, 1, 2
DECISION_NAMES = ("go", "refix", "reject")


def decide(e_hat: torch.Tensor, measured: torch.Tensor, go_below: torch.Tensor, reject_above: torch.Tensor):
    """(N,) decisions from (N,) measured axial error [m] and per-env thresholds [m]."""
    e = e_hat.abs()
    out = torch.full_like(e, REFIX, dtype=torch.long)
    out = torch.where(measured & (e <= go_below), torch.full_like(out, GO), out)
    out = torch.where(measured & (e > reject_above), torch.full_like(out, REJECT), out)
    return out
