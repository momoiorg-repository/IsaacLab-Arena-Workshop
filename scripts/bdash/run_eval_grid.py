# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""B-DASH evaluation grid runner (brief §3 M4).

Sweeps ``policy × clearance × level`` (from ``configs/bdash/peg_insert/eval_grid.yaml``), running each cell
seed-fixed for ``num_episodes`` trials by shelling out one ``policy_runner.py`` process per cell
(each Isaac Sim env config needs a fresh process). The per-episode §3.5 JSONL emitted by
:class:`~isaaclab_arena.metrics.bdash_handoff_logger.BDashHandoffRecorder` is the source of truth;
this script only orchestrates and aggregates.

Resumable: a cell whose newest JSONL already holds >= ``num_episodes`` records is skipped (``--no_resume``
to force re-run). Aggregates to a CSV with success rate ± 95% Wilson CI plus insertion-force and
handoff summaries (``--aggregate_only`` to re-aggregate existing logs without running).

Run inside the container:

    unset DISPLAY
    /isaac-sim/python.sh scripts/bdash/run_eval_grid.py                       # full grid
    /isaac-sim/python.sh scripts/bdash/run_eval_grid.py \\
        --clearances 2.0 1.0 --levels L1 --num_episodes 5                     # dry-run (2 cells × 5)
    /isaac-sim/python.sh scripts/bdash/run_eval_grid.py --aggregate_only      # just rebuild the CSV
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
import subprocess
import sys
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CFG = os.path.join(_REPO_ROOT, "configs", "bdash", "peg_insert", "eval_grid.yaml")
_POLICY_RUNNER = os.path.join(_REPO_ROOT, "isaaclab_arena", "evaluation", "policy_runner.py")

# GR00T VLA front-end policies. Referenced by dotted import path (lazy — keeps the gr00t dependency
# out of the base package) and run in the `franka_joint` joint-action env with cameras + the
# closed-loop config. The short key is the grid/run-tag name; bdash_scripted etc. need none of this.
_VLA_POLICIES = {
    "bdash_vla": {
        "policy_type": "isaaclab_arena.policy.bdash_vla_policy.BDashVLAPolicy",
        "config": "isaaclab_arena_gr00t/policy/config/bdash_pick_insert_gr00t_closedloop_config.yaml",
    },
    # v3 = GR00T fine-tuned on the square-peg + yaw-aligned-grasp dataset (same policy class, v3 ckpt).
    "bdash_vla_v3": {
        "policy_type": "isaaclab_arena.policy.bdash_vla_policy.BDashVLAPolicy",
        "config": "isaaclab_arena_gr00t/policy/config/bdash_pick_insert_gr00t_closedloop_v3_config.yaml",
    },
    # v4 = GR00T N1.7 fine-tuned on the v4 dataset (same policy class, N1.7 ckpt). Needs the gn17
    # container (isaaclab_arena-cuda_gr00t_gn17), not gn16.
    "bdash_vla_v4": {
        "policy_type": "isaaclab_arena.policy.bdash_vla_policy.BDashVLAPolicy",
        "config": "isaaclab_arena_gr00t/policy/config/bdash_pick_insert_gr00t_closedloop_v4_config.yaml",
    },
    # v4_28000 = same v4 run, trained longer (step-28000 checkpoint). gn17 container.
    "bdash_vla_v4_28000": {
        "policy_type": "isaaclab_arena.policy.bdash_vla_policy.BDashVLAPolicy",
        "config": "isaaclab_arena_gr00t/policy/config/bdash_pick_insert_gr00t_closedloop_v4_28000_config.yaml",
    },
    # v5_vision = v4 dataset re-trained with the vision encoder unfrozen (TUNE_VISUAL=true), to test
    # whether the v4 peg-mislocalization (vision bottleneck) lifts. gn17 container.
    "bdash_vla_v5_vision": {
        "policy_type": "isaaclab_arena.policy.bdash_vla_policy.BDashVLAPolicy",
        "config": "isaaclab_arena_gr00t/policy/config/bdash_pick_insert_gr00t_closedloop_v5_vision_config.yaml",
    },
    # v6_recovery = FROZEN-ViT N1.7 fine-tuned on the v5-recovery handoff set RE-RECORDED with the
    # camera fix (MOVING frames — all prior datasets trained on a single frozen reset frame). First
    # eval where both record + eval paths feed live frames; the real test of the frozen-frames fix.
    "bdash_vla_v6_recovery": {
        "policy_type": "isaaclab_arena.policy.bdash_vla_policy.BDashVLAPolicy",
        "config": "isaaclab_arena_gr00t/policy/config/bdash_pick_insert_gr00t_closedloop_v6_recovery_config.yaml",
    },
}


def _cstr(c: float) -> str:
    """Match the clearance token the env bakes into the log filename (``c2.0``, ``c0.25``)."""
    return str(float(c))


def _run_tag(policy: str) -> str:
    # the env appends ``_c{clearance}_{level}`` to the run_tag in the JSONL filename, so the policy is
    # all we need here; the cell is disambiguated by the env-appended tokens.
    return f"grid_{policy}"


def _cell_logs(log_dir: str, policy: str, c: float, level: str) -> list[str]:
    pat = os.path.join(log_dir, f"bdash_handoff_{_run_tag(policy)}_c{_cstr(c)}_{level}_*.jsonl")
    return sorted(glob.glob(pat), key=os.path.getmtime)


def _newest_complete(log_dir: str, policy: str, c: float, level: str, n_need: int) -> str | None:
    for path in reversed(_cell_logs(log_dir, policy, c, level)):
        try:
            with open(path) as fh:
                if sum(1 for _ in fh) >= n_need:
                    return path
        except OSError:
            continue
    return None


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal at the 0%/100% edges)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_cell(args, policy: str, c: float, level: str) -> str | None:
    """Run one grid cell via policy_runner (unless resumable); return the JSONL path or None."""
    if not args.no_resume:
        done = _newest_complete(args.log_dir, policy, c, level, args.num_episodes)
        if done is not None:
            print(f"[grid] skip (resume): {policy} c={c} {level}  ->  {os.path.basename(done)}")
            return done

    vla = _VLA_POLICIES.get(policy)
    policy_type = vla["policy_type"] if vla else policy
    pre_env = ["--policy_config_yaml_path", vla["config"], "--enable_cameras"] if vla else []
    post_env = ["--embodiment", "franka_joint"] if vla else []
    cmd = [
        args.python,
        _POLICY_RUNNER,
        "--policy_type",
        policy_type,
        "--num_episodes",
        str(args.num_episodes),
        # VLA closed-loop renders non-tiled cameras -> single env; scripted baselines can batch.
        "--num_envs",
        "1" if vla else str(args.num_envs),
        "--seed",
        str(args.seed),
        *pre_env,
        args.env,
        "--clearance",
        _cstr(c),
        "--level",
        level,
        "--log_dir",
        args.log_dir,
        "--run_tag",
        _run_tag(policy),
        *post_env,
    ]
    env = dict(os.environ)
    env.pop("DISPLAY", None)  # headless fast-start (memory: unset DISPLAY ~12s vs ~7min)
    print(f"[grid] RUN: {policy} c={c} {level}  ({args.num_episodes} ep)")
    subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=True)
    logs = _cell_logs(args.log_dir, policy, c, level)
    return logs[-1] if logs else None


def aggregate(args, rows_spec) -> list[dict]:
    out = []
    for policy, c, level in rows_spec:
        logs = _cell_logs(args.log_dir, policy, c, level)
        path = logs[-1] if logs else None
        rec = {
            "policy": policy,
            "clearance_mm": c,
            "level": level,
            "n": 0,
            "success": 0,
            "success_rate": "",
            "ci95_low": "",
            "ci95_high": "",
            "insert_force_median_N": "",
            "handoff_fired": 0,
            "jsonl": os.path.basename(path) if path else "",
        }
        if path:
            data = [json.loads(line) for line in open(path)]  # noqa: SIM115
            n = len(data)
            succ = sum(d["result"] == "success" for d in data)
            fi = [d.get("force_insert_max_N", 0.0) for d in data]
            lo, hi = _wilson_ci(succ, n)
            rec.update(
                n=n,
                success=succ,
                success_rate=round(succ / n, 4) if n else "",
                ci95_low=round(lo, 4),
                ci95_high=round(hi, 4),
                insert_force_median_N=round(st.median(fi), 3) if fi else "",
                handoff_fired=sum(d.get("t_handoff_s") is not None for d in data),
            )
        out.append(rec)
    return out


def write_csv(rows: list[dict], out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    cols = [
        "policy",
        "clearance_mm",
        "level",
        "n",
        "success",
        "success_rate",
        "ci95_low",
        "ci95_high",
        "insert_force_median_N",
        "handoff_fired",
        "jsonl",
    ]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[grid] wrote {out_csv}  ({len(rows)} cells)")


def main():
    cfg = {}
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=_DEFAULT_CFG)
    known, _ = pre.parse_known_args()
    if os.path.exists(known.config):
        cfg = yaml.safe_load(open(known.config)) or {}  # noqa: SIM115

    p = argparse.ArgumentParser(parents=[pre], description="B-DASH evaluation grid runner (M4)")
    p.add_argument("--policies", nargs="+", default=cfg.get("policies", ["bdash_scripted"]))
    p.add_argument("--clearances", type=float, nargs="+", default=cfg.get("clearances_mm", [2.0, 1.0, 0.5, 0.25]))
    p.add_argument("--levels", nargs="+", default=cfg.get("levels", ["L0", "L1", "L2", "L3"]))
    p.add_argument("--num_episodes", type=int, default=cfg.get("num_episodes", 50))
    p.add_argument("--num_envs", type=int, default=cfg.get("num_envs", 8))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    p.add_argument("--env", default=cfg.get("env", "bdash_pick_insert"))
    p.add_argument("--log_dir", default=cfg.get("log_dir", "logs/bdash"))
    p.add_argument("--out", dest="out_csv", default=cfg.get("out_csv", "results/bdash/eval_grid.csv"))
    p.add_argument("--python", default="/isaac-sim/python.sh")
    p.add_argument("--no_resume", action="store_true", help="re-run cells even if a complete JSONL exists")
    p.add_argument("--aggregate_only", action="store_true", help="skip running; rebuild the CSV from existing logs")
    args = p.parse_args()

    rows_spec = [(pol, c, lv) for pol in args.policies for c in args.clearances for lv in args.levels]
    print(f"[grid] {len(rows_spec)} cells | episodes={args.num_episodes} envs={args.num_envs} seed={args.seed}")

    if not args.aggregate_only:
        for policy, c, level in rows_spec:
            try:
                run_cell(args, policy, c, level)
            except subprocess.CalledProcessError as e:
                print(f"[grid] CELL FAILED ({policy} c={c} {level}): {e} — continuing", file=sys.stderr)

    rows = aggregate(args, rows_spec)
    write_csv(rows, args.out_csv)
    print("\n[grid] summary:")
    for r in rows:
        print(
            f"  {r['policy']:14s} c={r['clearance_mm']:<5} {r['level']:<3} "
            f"{r['success']:>3}/{r['n']:<3} = {r['success_rate']}  "
            f"CI95[{r['ci95_low']},{r['ci95_high']}]  f_ins~{r['insert_force_median_N']}N"
        )
    print("GRID_DONE")


if __name__ == "__main__":
    main()
