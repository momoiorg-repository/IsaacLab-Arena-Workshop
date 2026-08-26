# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""spec §9: every write to ``ee_pose_action`` must declare an owner.

The chuck teacher blends four layers into one action tensor (pick, insertion, creep, hold) plus a
separate gripper column. The blend is lossless about the numbers and totally lossy about the
*provenance*: two layers that disagree produce a tensor indistinguishable from one layer agreeing
with itself. Three defects in that file were exactly this, and each was found only by inference from
downstream symptoms -- W-B 43 mm past target, parts falling through the bore, W-A stuck in SETTLE in
12 of 12 episodes.

So the rule is structural, not stylistic: overrides go through ``_take_action`` / ``_take_gripper``,
which take a phase-condition mask and record who won. This test fails the build if a raw override is
added instead, because a comment asking future changes to be careful has already not worked.

Sim-free: pure ``ast``, no torch, no Isaac Sim.
"""

from __future__ import annotations

import ast
import pathlib

POLICY = pathlib.Path(__file__).resolve().parents[2] / "isaaclab_arena/policy/bdash_chuck_policy.py"
# The helpers themselves are where the real writes live, by construction.
EXEMPT = {"_take_action", "_take_gripper", "_begin_action"}


def _tree() -> ast.Module:
    return ast.parse(POLICY.read_text())


def _is_torch_where(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "where"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    )


def _writes_to_action(tree: ast.Module):
    """(function name, line, kind) for every assignment to `action` outside the exempt helpers."""
    offences = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or func.name in EXEMPT:
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                # `action = torch.where(...)` -- an unlabelled blend
                if isinstance(target, ast.Name) and target.id == "action" and _is_torch_where(node.value):
                    offences.append((func.name, node.lineno, "torch.where blend"))
                # `action[:, 6] = ...` -- an unlabelled column write
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "action"
                ):
                    offences.append((func.name, node.lineno, "column write"))
    return offences


def test_no_unlabelled_action_override():
    offences = _writes_to_action(_tree())
    assert not offences, (
        "spec §9: these write the action without declaring an owner -- route them through "
        "_take_action / _take_gripper with a PHASE-CONDITION mask:\n"
        + "\n".join(f"  {POLICY.name}:{line}  in {name}()  ({kind})" for name, line, kind in offences)
    )


def test_every_owner_code_is_named():
    """The code written into the log must index a name, or the log is unreadable."""
    src = POLICY.read_text()
    namespace: dict = {}
    for line in src.splitlines():
        if line.startswith(("ACTION_OWNERS", "OWNER_", "GRIP_OWNERS", "GRIP_")):
            exec(line, {"range": range}, namespace)  # noqa: S102 -- module-level literals only
    assert namespace["ACTION_OWNERS"] == ("pick", "reerect", "touchoff", "insert", "creep", "hold")
    assert namespace["GRIP_OWNERS"] == ("pick", "release")
    for prefix, names in (("OWNER_", "ACTION_OWNERS"), ("GRIP_", "GRIP_OWNERS")):
        codes = {k: v for k, v in namespace.items() if k.startswith(prefix) and isinstance(v, int)}
        assert sorted(codes.values()) == list(range(len(namespace[names]))), f"{prefix}* is not 0..n-1"


# The legs that BORROW the arm from the pick. Each drives it to a fixture of its own -- the datum
# block, the set-down station -- i.e. away from the transport waypoint the pick is still holding.
BORROWING_OWNERS = {"OWNER_REERECT", "OWNER_TOUCHOFF"}


def _functions_taking_action_as(tree: ast.Module, owners: set[str]) -> dict[str, set[str]]:
    """{function name: set of `_take_action` owner names it passes} for the given owners."""
    found: dict[str, set[str]] = {}
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_take_action"
                and node.args
                and isinstance(node.args[-1], ast.Name)
                and node.args[-1].id in owners
            ):
                found.setdefault(func.name, set()).add(node.args[-1].id)
    return found


def test_every_borrowing_leg_stands_the_pick_watchdog_down():
    """A leg that takes the arm must also mark the env borrowed, or the pick retries under it.

    The pick keeps commanding its transport waypoint after DONE, and its stall watchdog keeps
    measuring progress toward it. A borrowed leg drives the arm somewhere else, so the watchdog sees
    the distance stop improving, calls a stall, and drops `phase` back to APPROACH -- with the part
    still in the fingers. Measured on the re-erect leg: it never once left its first phase (72 active
    steps, retries = 1) while its planned turn was a correct 90 deg that never got to run, and every
    episode ended in `load_failed` as the pick drove the held part back down into the tray.

    `_in_insertion` covers the insertion handoff and nothing else -- during a borrowed leg it is
    still False, which is exactly how this stayed open. The same hole was live in the touch-off leg
    and had simply not been measured yet. So the rule is structural: pass a non-pick, non-insertion
    owner to `_take_action` and you must contribute to `_borrowed` in the same function.
    """
    tree = _tree()
    src = POLICY.read_text()
    legs = _functions_taking_action_as(tree, BORROWING_OWNERS)
    assert legs, "no borrowing leg found -- has an owner been renamed?"

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or func.name not in legs:
            continue
        marks = [
            n
            for n in ast.walk(func)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "_borrowed" for t in n.targets)
        ]
        assert marks, (
            f"{func.name}() takes the arm as {sorted(legs[func.name])} but never marks the env in "
            "`self._borrowed`, so the pick's stall watchdog keeps running under it"
        )

    # And the suppression itself must actually consume the flag.
    assert (
        "watchdog_suppress = self._in_insertion | self._borrowed" in src
    ), "`_borrowed` is collected but not wired into `watchdog_suppress`"


# The chuck legs. The frozen peg path is deliberately not covered: it is not to be touched.
GRIPPER_POLARITY_FILES = (
    "isaaclab_arena/controllers/reerect_controller.py",
    "isaaclab_arena/controllers/touch_off_controller.py",
)


def test_gripper_argument_is_always_passed_by_keyword():
    """``ee_pose_action``'s gripper argument must be named at the call site, in these files.

    Its parameter is ``gripper_open`` and TRUE opens the fingers. Passed positionally, the call site
    is free to name its local whatever it likes, and both chuck legs named theirs ``closed`` and
    handed it straight over -- inverting the polarity in both.

    In the re-erect leg the effect was total and silent: the fingers opened on the first step the leg
    owned the arm, the part fell out during the turn, and the leg -- open loop by §9 -- went on
    turning an empty hand through its whole trajectory. Every log line looked like a turn in
    progress. It took a per-step trace of the gripper WIDTH (27.8 mm opening to 76.5 mm over nine
    steps) to see it. The touch-off leg carried the identical inversion, unmeasured and therefore
    unnoticed.

    A keyword makes the polarity readable at the point where it is chosen, which is the only place
    the mistake can be caught by eye.
    """
    import pathlib as _pathlib

    root = _pathlib.Path(__file__).resolve().parents[2]
    offences = []
    for rel in GRIPPER_POLARITY_FILES:
        path = root / rel
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ee_pose_action"):
                continue
            if len(node.args) > 3:
                offences.append(f"  {rel}:{node.lineno}  passes the gripper positionally")
            elif not any(kw.arg == "gripper_open" for kw in node.keywords):
                offences.append(f"  {rel}:{node.lineno}  does not name `gripper_open`")
    assert not offences, "ee_pose_action's gripper argument must be passed as `gripper_open=`:\n" + "\n".join(offences)


def test_guard_catches_a_raw_override():
    """The guard must actually fail on the pattern it exists to reject, not merely pass today."""
    tree = ast.parse(
        "import torch\n"
        "class P:\n"
        "    def _blend(self, action, mask, other):\n"
        "        action = torch.where(mask.unsqueeze(-1), other, action)\n"
        "        action[:, 6] = 1.0\n"
        "        return action\n"
    )
    kinds = sorted(kind for _, _, kind in _writes_to_action(tree))
    assert kinds == ["column write", "torch.where blend"]


if __name__ == "__main__":
    test_no_unlabelled_action_override()
    test_every_owner_code_is_named()
    test_every_borrowing_leg_stands_the_pick_watchdog_down()
    test_gripper_argument_is_always_passed_by_keyword()
    test_guard_catches_a_raw_override()
    print("BDASH_ACTION_OWNER_OK")
