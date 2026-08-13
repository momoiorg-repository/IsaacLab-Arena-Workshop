# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH organizer: task -> subtasks -> skill assignment.

Division of labor (by design):
  - Subtask decomposition is semantic and may use a VLM/LLM (or the built-in
    template for known tasks).
  - Skill assignment is numeric and is ALWAYS done by contracts.py against
    the measured specs in skill_library.yaml.

Usage:
  /isaac-sim/python.sh -m isaaclab_arena.bdash.organizer \
      --task "insert the peg into the socket" --clearance 1.5
"""

from __future__ import annotations

import argparse
import json

from isaaclab_arena.bdash import contracts

# Built-in decomposition templates (VLM call can replace this; the output
# schema is the same: a list of {subtask, capability}).
TEMPLATES = {
    "peg_insert": [
        {"subtask": "grasp and carry the peg to the socket mouth", "capability": "grasp"},
        {"subtask": "insert the peg into the socket", "capability": "precision_insert"},
    ],
}


def decompose(task: str, use_vlm: bool = False) -> list[dict]:
    if use_vlm:
        return vlm_decompose(task)
    if "insert" in task.lower() or "嵌合" in task or "挿入" in task:
        return TEMPLATES["peg_insert"]
    raise ValueError(f"no decomposition template for task: {task!r} (try --vlm)")


def vlm_decompose(task: str) -> list[dict]:
    """Decompose a task into subtasks with a VLM. Semantics only: the VLM
    never sees or judges numeric specs; assignment stays in contracts.py."""
    import json as _json
    import os
    from pathlib import Path

    from google import genai
    from google.genai import types

    key_file = os.environ.get("BDASH_GEMINI_KEY_FILE", str(Path.home() / ".gemini_paid_key"))
    client = genai.Client(api_key=Path(key_file).read_text().strip())
    prompt = (
        "You are a robot task planner. Decompose the task below into an ordered list of "
        "subtasks for a single robot arm workcell. Use ONLY these capability tags: "
        "grasp, precision_insert. Reply with a JSON array of objects "
        '{"subtask": <short description>, "capability": <tag>} and nothing else.\n'
        f"Task: {task}"
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    subtasks = _json.loads(resp.text)
    assert isinstance(subtasks, list) and all("capability" in st for st in subtasks)
    return subtasks


def plan(
    task: str,
    clearance_mm: float | None = None,
    library_path=None,
    use_vlm: bool = False,
    exclude: list[str] | None = None,
    best_effort: bool = False,
) -> dict:
    library = contracts.load_library(library_path or contracts.DEFAULT_LIBRARY)
    if exclude:
        library = {k: v for k, v in library.items() if k not in exclude}
    subtasks = decompose(task, use_vlm=use_vlm)

    assignments = []
    # Back-end is fixed by capability; front-end is chosen by contract ranking.
    back_name = next(n for n, s in library.items() if "precision_insert" in s["capabilities"])
    ranked = contracts.rank_frontends(library, back_name, clearance_mm)
    best = ranked[0] if ranked else None

    if best is None or best.expected_success <= 0.0 or (not best.feasible and not best_effort):
        return {
            "task": task,
            "clearance_mm": clearance_mm,
            "feasible": False,
            "candidates": [r.__dict__ for r in ranked],
            "decision": "reject: no front-end skill satisfies the back-end budget",
        }

    degraded = not best.feasible
    assignments.append(
        {"subtask": subtasks[0]["subtask"], "skill": best.front, "expected_success": round(best.expected_success, 3)}
    )
    assignments.append({"subtask": subtasks[1]["subtask"], "skill": back_name, "gate": library[back_name]["gate"]})
    result = {
        "task": task,
        "clearance_mm": clearance_mm,
        "feasible": True,
        "assignments": assignments,
        "candidates": [r.__dict__ for r in ranked],
    }
    if degraded:
        result["degraded"] = True
        result["decision"] = (
            "best-effort: budget not satisfied in expectation; run relies on the gate (no-go detection / recovery)"
        )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="insert the peg into the socket")
    ap.add_argument("--clearance", type=float, default=2.0)
    ap.add_argument("--vlm", action="store_true", help="use a VLM for subtask decomposition")
    ap.add_argument("--exclude", nargs="*", default=None, help="skills unavailable in this cell")
    ap.add_argument(
        "--best-effort",
        action="store_true",
        help="pick the best candidate even if the budget is not met in expectation",
    )
    args = ap.parse_args()
    print(
        json.dumps(
            plan(args.task, args.clearance, use_vlm=args.vlm, exclude=args.exclude, best_effort=args.best_effort),
            ensure_ascii=False,
            indent=2,
        )
    )
