# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH demo: plan-time budget decisions vs measured outcomes.

For each clearance condition, the organizer plans (or rejects) and the
executor runs the plan; predictions and measurements land in one table.

Run inside the Isaac container:
  /isaac-sim/python.sh -m isaaclab_arena.bdash.demo --clearances 2.0 1.5 1.25 --episodes 20
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

from isaaclab_arena.bdash import executor, organizer

REPO = Path("/workspaces/isaaclab_arena")


def run_cell(
    clearance: float,
    episodes: int,
    seed: int,
    task: str,
    use_vlm: bool = False,
    exclude: list[str] | None = None,
    best_effort: bool = False,
) -> dict:
    plan = organizer.plan(task, clearance, use_vlm=use_vlm, exclude=exclude, best_effort=best_effort)
    row = {"clearance_mm": clearance, "feasible": plan["feasible"]}
    best = plan["candidates"][0]
    row["chosen_skill"] = best["front"] if plan["feasible"] else "-"
    row["predicted_success"] = round(best["expected_success"], 3)
    if not plan["feasible"]:
        row.update(measured_success="-", n=0, note="rejected by contract check")
        return row

    out_csv = f"results/vdash/bdash_demo_c{clearance}.csv"
    run = executor.build_run(plan, episodes, seed, out_csv)
    env = dict(os.environ, **run["env"])
    env.pop("DISPLAY", None)
    subprocess.run(run["cmd"], cwd=run["cwd"], env=env, check=True)

    with open(REPO / out_csv, encoding="utf-8") as f:
        result = list(csv.DictReader(f))[-1]
    note = f"policy={run['policy']}"
    if plan.get("degraded"):
        note += " (best-effort: budget not met, gate-monitored)"
    row.update(measured_success=result["success_rate"], n=result["n"], note=note)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clearances", type=float, nargs="+", default=[2.0, 1.5, 1.25])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="insert the peg into the socket")
    ap.add_argument("--out", default="results/vdash/bdash_demo_table.csv")
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--best-effort", action="store_true")
    args = ap.parse_args()

    rows = [
        run_cell(
            c, args.episodes, args.seed, args.task, use_vlm=args.vlm, exclude=args.exclude, best_effort=args.best_effort
        )
        for c in args.clearances
    ]
    out = REPO / args.out
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[bdash-demo] wrote {out}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
