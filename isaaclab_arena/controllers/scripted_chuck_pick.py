# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted grasp+lift expert for the B-DASH chuck-loading tray.

The teacher for the VLA's slice of the task: approach, descend, close, lift, done. Everything after
the lift -- touch-off, the V-block re-erect, chuck loading, jaw closing -- stays with the classic
controllers, so the recorded demos are short and their length is predictable.

This subclasses :class:`~isaaclab_arena.controllers.scripted_pick.ScriptedPick` rather than copying
it. Copying would double the maintenance surface on the one controller whose behaviour must not
drift (it produced the frozen v6 numbers), and parameterising it in place would put chuck branches
inside the exact ``step()`` that produced them. Subclassing needs only the three hooks that file
already exposes, each a no-op when the corresponding override is unset.

Both the grasp point and the grasp orientation must be injected by the caller: unlike the peg
expert there is no ground-truth fallback here, because with six candidate workpieces in the tray
"the object" is not well defined until an episode has picked one.
"""

from __future__ import annotations

import torch

from isaaclab_arena.controllers.scripted_pick import ScriptedPick


class ScriptedChuckPick(ScriptedPick):
    """Grasp + lift. Requires ``grasp_override`` and ``grasp_quat_override`` to be set.

    Set ``transport_xy`` (N,2) to add a transport leg over the chuck for the load stage; leave it
    ``None`` for the pick stage, which then behaves exactly as before.
    """

    #: (N,2) world xy the lift is carried to before the insertion controller takes over, or None.
    transport_xy: torch.Tensor | None = None

    def _grasp_point(self, env) -> torch.Tensor:
        if self.grasp_override is None:
            raise RuntimeError("ScriptedChuckPick requires grasp_override; the policy must set it each step")
        return self.grasp_override

    def _grasp_quat(self, env) -> torch.Tensor:
        if self.grasp_quat_override is None:
            raise RuntimeError("ScriptedChuckPick requires grasp_quat_override; the policy must set it each step")
        return self.grasp_quat_override

    def _waypoints(self, env, gp: torch.Tensor):
        """Approach, lift, then transport to the bore -- or stop at the lift if nothing asked for more.

        With ``transport_xy`` unset (the pick stage, which is what every recorded demo and all of
        GATE 1 runs on) TRANSPORT reuses the lift waypoint, so the phase is already satisfied when
        it is entered and the FSM falls straight through to DONE. That keeps the inherited
        transition table untouched and ends the episode where the demo is cut.

        With it set (the load stage) the same waypoint carries the part over the chuck axis at lift
        height, which is where the insertion controller takes over. The handoff needs the part above
        the bore and roughly on its axis; it does the rest with its own force servo and spiral.
        """
        above = gp.clone()
        above[:, 2] += self.cfg["approach_height"]
        lift = gp.clone()
        lift[:, 2] = self.cfg["lift_height"]
        if self.transport_xy is None:
            return above, lift, lift
        transport = lift.clone()
        transport[:, :2] = self.transport_xy
        return above, lift, transport
