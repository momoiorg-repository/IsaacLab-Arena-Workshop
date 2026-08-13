# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH executor: turn an organizer plan into an evaluation run.

Maps the assigned front-end skill to the corresponding registered policy of
`scripts/vdash/run_eval_grid.py` and carries the gate configuration as
environment variables. The gate itself and the rule-based back-end are already
wired inside the vdash policies; this module only orchestrates.

Usage (dry run, host or container):
  python3 -m isaaclab_arena.bdash.executor --task "insert the peg" \
      --clearance 1.5 --episodes 20 --dry-run
Real runs must be launched inside the Isaac container.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess

from isaaclab_arena.bdash import contracts, organizer

REPO = "/workspaces/isaaclab_arena"
PYTHON = "/isaac-sim/python.sh"
GRID = "scripts/vdash/run_eval_grid.py"


def build_run(
    plan: dict, episodes: int = 20, seed: int = 0, out_csv: str = "results/vdash/bdash_demo.csv"
) -> dict | None:
    """Build the command + env for an executable plan. None if plan infeasible."""
    if not plan.get("feasible"):
        return None
    library = contracts.load_library()
    front = plan["assignments"][0]["skill"]
    policy = library[front]["runner_policy"]
    back = plan["assignments"][1]["skill"]
    gate_env = {k: str(v) for k, v in library[back]["gate"].get("env", {}).items()}

    cmd = [
        PYTHON,
        GRID,
        "--policies",
        policy,
        "--clearances",
        str(plan["clearance_mm"]),
        "--levels",
        "L1",
        "--num_episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--log_dir",
        "logs/vdash",
        "--no_resume",
        "--out",
        out_csv,
    ]
    return {"cwd": REPO, "cmd": cmd, "env": gate_env, "policy": policy}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="insert the peg into the socket")
    ap.add_argument("--clearance", type=float, default=2.0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/vdash/bdash_demo.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = organizer.plan(args.task, args.clearance)
    print("=== organizer plan ===")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    run = build_run(plan, args.episodes, args.seed, args.out)
    if run is None:
        print("=== executor: plan rejected, nothing to run ===")
        return
    env_str = " ".join(f"{k}={v}" for k, v in run["env"].items())
    print("=== executor command ===")
    print(f"cd {run['cwd']} && {env_str} {shlex.join(run['cmd'])}")
    if not args.dry_run:
        env = dict(os.environ, **run["env"])
        env.pop("DISPLAY", None)
        subprocess.run(run["cmd"], cwd=run["cwd"], env=env, check=True)


if __name__ == "__main__":
    main()
