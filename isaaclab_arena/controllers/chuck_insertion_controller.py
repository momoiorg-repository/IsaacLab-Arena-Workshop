# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""spec §4-3 ``chuck_load``: force-limited descent into the chuck bore, for the B-DASH chuck task.

This subclasses :class:`~isaaclab_arena.controllers.insertion_controller.InsertionController`
rather than copying or parameterising it. That controller produced the frozen v4/v6 peg numbers, so
its ``step()`` must not gain task branches; and its control law needs nothing chuck-specific to
begin with -- it servos to "a vertical axis through ``socket.root_pos_w[:, :2]`` with a mouth at
``root_z + mouth_height``", which the chuck body satisfies exactly at ``chuck_face_height``.

Two things genuinely differ, and both are hooks the parent already exposes:

* **Which sensor.** The peg scene has one contact sensor; the chuck scene has one per workpiece and
  the episode's part is latched on ``env.bdash_target_idx``. So :meth:`_read_wrench` gathers.
* **Where the tip is.** ``tip_estimate`` is proprioceptive: it assumes the tip sits ``grip_offset``
  below the EE along the held approach axis. For the peg that is one constant. Here each variant is
  gripped at its own axial station -- side-lying at ``grip_station_side``, upright at
  ``length - grip_below_top_upright`` -- so ``grip_offset`` is a per-env tensor, set by the policy
  once the target is known.

Not implemented here, deliberately: closing the jaws and the air-seat check (§4-3's tail). The jaws
are kinematic and nothing drives them, so there is nothing to hand the part over to; the episode
ends on the geometric ``seated`` predicate with the gripper still holding.
"""

from __future__ import annotations

import torch

from isaaclab_arena.controllers.insertion_controller import InsertionController


class ChuckInsertionController(InsertionController):
    """Insertion into the chuck bore. Set :attr:`sensor_names` and :attr:`grip_offset` per episode."""

    #: Per-workpiece unfiltered contact sensors, ordered like the task's workpiece list.
    sensor_names: tuple[str, ...] = ()
    #: Last wrench read, kept for the JSONL sidecar. The force servo stopping short and the
    #: position command not arriving look identical in `best_depth`; this separates them.
    last_wrench: torch.Tensor | None = None

    def _read_wrench(self, env) -> torch.Tensor:
        """Contact force on the LATCHED target from the CHUCK ONLY, (N,3) -- not its net force.

        This is the one place the chuck cannot reuse the peg's reading, and the reason is mass. The
        controller servos ``fz`` toward ``f_target`` as "how hard am I pressing". On the peg that
        works because the net contact force on a held Ø8 peg is dominated by the socket: the peg
        weighs about 0.19 N. These workpieces weigh 3.4-4.7 N, so a net reading is 60-80% the
        gripper's reaction to their own weight before the part has touched anything at all.

        Measured, W-C: ``fz`` = 4.8-5.6 N against ``f_target`` 6.0 N while the contact force
        against the chuck body and all three jaws was **0.0 N** -- the part was hanging free above
        the bore and the servo had already stopped descending, because it read the weight it was
        carrying as a press it had achieved.

        Filtering to the fixtures removes the gripper from the signal entirely: holding the part
        contributes nothing, and ``fz`` is a real insertion reaction or it is zero. The jam
        termination (``cp.load_failed``) keeps using the NET sensor, which is right for a jam -- any
        excessive force is a jam, whoever is applying it.
        """
        uenv = env.unwrapped
        # (N, K, 3): sum over this workpiece's contact points AND over the fixture filters, so a
        # part resting on the face and one wedged against a jaw both register.
        forces = torch.stack(
            [
                uenv.scene[name].data.force_matrix_w.flatten(start_dim=1, end_dim=-2).sum(dim=1)
                for name in self.sensor_names
            ],
            dim=1,
        )
        idx = getattr(uenv, "bdash_target_idx", None)
        if idx is None:
            self.last_wrench = forces.sum(dim=1)
        else:
            self.last_wrench = forces.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        return self.last_wrench
