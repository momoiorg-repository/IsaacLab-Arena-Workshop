# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""spec §0-5 / §2.1: the RUNTIME skills may not read a workpiece's true pose.

The teacher is allowed to be privileged -- it drives from ground truth and the VLA learns from its
demonstrations. The classical skills that run BESIDE the VLA at inference time are not: touch-off,
the re-erect and the insertion have to work from proprioception, surveyed FIXTURE poses and contact
force, because at runtime the workpiece's pose is exactly the thing nobody knows.

This is easy to violate by accident and impossible to see afterwards. A skill that reads
``root_pos_w`` on a workpiece behaves perfectly in sim and cannot run on hardware at all, and
nothing in the measured success rate says so -- it says the opposite, because ground truth makes the
skill work BETTER. The check therefore has to be structural.

**Fixture poses are allowed and workpiece poses are not**, which is the whole distinction: a chuck
or a V-block is surveyed once and known to the cell; a workpiece arrives in an unknown pose. The
allow-list below names the fixtures.

Sim-free: pure ``ast``.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Skills that run at inference time beside the VLA. The teacher policy is deliberately NOT here.
RUNTIME_SKILLS = (
    "isaaclab_arena/controllers/reerect_controller.py",
    "isaaclab_arena/controllers/touch_off_controller.py",
)

#: Reading these is fine -- they are surveyed cell geometry, not workpieces.
FIXTURE_NAMES = ("chuck", "socket", "vblock", "touchoff", "block", "pad", "tray", "workbench")

PRIVILEGED_ATTRS = ("root_pos_w", "root_quat_w", "root_state_w", "root_lin_vel_w", "root_ang_vel_w")


def _privileged_reads(path: pathlib.Path):
    """(line, attr, context) for every true-pose read that is not clearly a fixture."""
    src = path.read_text()
    hits = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Attribute) and node.attr in PRIVILEGED_ATTRS):
            continue
        # The subscript that selected the asset is the best available clue about WHAT was read.
        segment = ast.get_source_segment(src, node) or ""
        if any(name in segment.lower() for name in FIXTURE_NAMES):
            continue
        hits.append((node.lineno, node.attr, segment.strip()[:90]))
    return hits


def test_runtime_skills_do_not_read_workpiece_pose():
    offences = []
    for rel in RUNTIME_SKILLS:
        for line, attr, seg in _privileged_reads(ROOT / rel):
            offences.append(f"  {rel}:{line}  {attr}  <- {seg}")
    assert not offences, (
        "spec §0-5: a runtime skill is reading a workpiece's TRUE pose. It will score well in sim "
        "and cannot run on hardware:\n"
        + "\n".join(offences)
    )


def test_the_guard_catches_a_real_violation():
    """The guard must fail on the pattern it exists to reject, not merely pass today."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        bad = pathlib.Path(d) / "bad_skill.py"
        bad.write_text("def step(env, name):\n    return env.scene[name].data.root_quat_w\n")
        assert _privileged_reads(bad), "the guard does not catch a plain workpiece pose read"
        ok = pathlib.Path(d) / "ok_skill.py"
        ok.write_text("def step(env):\n    return env.scene['bdash_chuck_body'].data.root_pos_w\n")
        assert not _privileged_reads(ok), "the guard rejects a FIXTURE read, which is allowed"


if __name__ == "__main__":
    test_runtime_skills_do_not_read_workpiece_pose()
    test_the_guard_catches_a_real_violation()
    print("BDASH_NO_PRIVILEGED_SKILLS_OK")
