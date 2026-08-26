# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""spec §5: the three-way gate's decision table, pinned.

The gate is one comparison, which is exactly why it gets a test: a flipped inequality or a wrong
default routes every part the wrong way and nothing crashes. The rows below are the decision table
from the module docstring, including the two edges that carry policy: an UNMEASURED part is REFIX
(no measurement is not a good measurement), and the boundary values land on the safe side.

Sim-free: torch only.
"""

from __future__ import annotations

import torch

from isaaclab_arena.controllers.budget_gate import GO, REFIX, REJECT, decide


def test_decision_table():
    e = torch.tensor([0.0, 0.0005, 0.001, 0.0011, -0.003, 0.015, 0.0151, -0.020, 0.002])
    measured = torch.tensor([True, True, True, True, True, True, True, True, False])
    go = torch.full_like(e, 0.001)
    reject = torch.full_like(e, 0.015)
    want = [GO, GO, GO, REFIX, REFIX, REFIX, REJECT, REJECT, REFIX]
    got = decide(e, measured, go, reject).tolist()
    assert got == want, f"decision table broke: {got} != {want}"


def test_thresholds_are_per_env():
    """W-B's tighter window must not leak onto the other variants (or vice versa)."""
    e = torch.tensor([0.0009, 0.0009])
    measured = torch.ones(2, dtype=torch.bool)
    go = torch.tensor([0.001, 0.00075])  # W-A-ish, W-B-ish
    reject = torch.full_like(e, 0.015)
    got = decide(e, measured, go, reject).tolist()
    assert got == [GO, REFIX], got


if __name__ == "__main__":
    test_decision_table()
    test_thresholds_are_per_env()
    print("BDASH_GATE_OK")
