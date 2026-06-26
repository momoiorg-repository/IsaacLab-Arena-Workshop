# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""E4 convergence-zone harness (brief §3 M7).

Holds the grasped peg tip at a controlled lateral offset ``(dx,dy)`` and tilt ``θ`` just above the
socket mouth, then runs **only** the insertion controller in straight-press mode (spiral / retry /
re-center off — :meth:`InsertionController.set_m7`) to map the **funnel capture region** = the
convergence zone (funnel entrance). Sweeps ``r × azimuth × θ`` for a fixed clearance and writes a
per-cell success CSV; ``make_e4_figure.py`` turns that into the heatmap + the handoff-error overlay.

Single Isaac Sim process: the (r,az,θ) grid is flattened and run ``num_envs`` cells at a time (each
parallel env = one cell), so the whole grid is a handful of batched episodes. At L0 the placement is
deterministic, so one trial per cell is a clean binary capture test.

Run inside the container:

    unset DISPLAY
    /isaac-sim/python.sh scripts/vdash/run_convergence_zone.py \\
        --policy_type vdash_scripted --num_envs 16 --seed 0 \\
        --r_mm 0 1 2 3 4 5 --az_deg 0 90 180 270 --tilt_deg 0 2 4 6 8 \\
        --out results/vdash/e4_convergence_c2.csv \\
        vdash_pick_insert --clearance 2.0 --level L0
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import gymnasium as gym
import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _build_cells(r_mm, az_deg, tilt_deg):
    """Flatten the (r, azimuth, θ) grid into cells, dropping degenerate duplicates at r=0,θ=0."""
    cells = []
    seen_origin = False
    for r in r_mm:
        for az in az_deg:
            for th in tilt_deg:
                if r == 0 and th == 0:
                    if seen_origin:
                        continue
                    seen_origin = True
                cells.append({"r_mm": float(r), "az_deg": float(az), "tilt_deg": float(th)})
    return cells


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--r_mm", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--az_deg", type=float, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--tilt_deg", type=float, nargs="+", default=[0, 2, 4, 6, 8])
    parser.add_argument("--max_steps", type=int, default=900, help="per-batch step budget")
    parser.add_argument("--recover", action="store_true",
                        help="recovery basin: deliver the peg to (r,θ) then release so the controller "
                             "recenters on the true axis (vs the default held static-tolerance map)")
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "results", "vdash", "e4_convergence.csv"))
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        from isaaclab_arena.evaluation.policy_runner import get_policy_cls

        add_policy_runner_arguments(parser)
        env_parser = get_isaaclab_arena_environments_cli_parser(parser)
        policy_cls = get_policy_cls(parser.parse_known_args()[0].policy_type)
        env_parser = policy_cls.add_args_to_parser(env_parser)
        args_cli = env_parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        name, cfg = arena_builder.build_registered()
        env = gym.make(name, cfg=cfg)
        if args_cli.seed is not None:
            set_seed(args_cli.seed, env)
        dev = env.unwrapped.device
        n_envs = env.unwrapped.num_envs
        clearance_mm = float(env.unwrapped.cfg.isaaclab_arena_env.task.clearance_m * 1000.0)

        policy = policy_cls.from_args(args_cli)

        cells = _build_cells(args_cli.r_mm, args_cli.az_deg, args_cli.tilt_deg)
        print(f"[e4] clearance={clearance_mm}mm  cells={len(cells)}  num_envs={n_envs}")

        results = []
        for b0 in range(0, len(cells), n_envs):
            batch = cells[b0:b0 + n_envs]
            offset = torch.zeros(n_envs, 2, device=dev)
            tilt = torch.zeros(n_envs, 2, device=dev)
            for i, cell in enumerate(batch):
                az = math.radians(cell["az_deg"])
                r = cell["r_mm"] / 1000.0
                th = math.radians(cell["tilt_deg"])
                offset[i, 0] = r * math.cos(az)
                offset[i, 1] = r * math.sin(az)
                tilt[i, 0] = th * math.cos(az)
                tilt[i, 1] = th * math.sin(az)

            done = torch.zeros(n_envs, dtype=torch.bool, device=dev)
            succ = torch.zeros(n_envs, dtype=torch.bool, device=dev)
            with torch.inference_mode():  # policy state tensors are inference tensors; keep resets inside
                obs, _ = env.reset()
                policy.reset()
                policy.set_m7(offset, tilt, recover=args_cli.recover)
                for _ in range(args_cli.max_steps):
                    action = policy.get_action(env, obs)
                    obs, _, term, trunc, _ = env.step(action)
                    fin = term | trunc
                    success_now = env.unwrapped.termination_manager.get_term("success")
                    newly = fin & ~done
                    succ = torch.where(newly, success_now, succ)
                    done = done | fin
                    if newly.any():
                        policy.reset(env_ids=newly.nonzero().flatten())
                    if bool(done.all()):
                        break

            for i, cell in enumerate(batch):
                az = math.radians(cell["az_deg"])
                results.append({
                    "clearance_mm": clearance_mm,
                    "r_mm": cell["r_mm"], "az_deg": cell["az_deg"], "tilt_deg": cell["tilt_deg"],
                    "dx_mm": round(cell["r_mm"] * math.cos(az), 3),
                    "dy_mm": round(cell["r_mm"] * math.sin(az), 3),
                    "success": int(bool(succ[i].item())),
                })
            ok = sum(r["success"] for r in results[-len(batch):])
            print(f"[e4] batch {b0//n_envs}: {ok}/{len(batch)} captured")

        os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
        with open(args_cli.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["clearance_mm", "r_mm", "az_deg", "tilt_deg", "dx_mm", "dy_mm", "success"])
            w.writeheader()
            w.writerows(results)
        cap = sum(r["success"] for r in results)
        print(f"[e4] wrote {args_cli.out}  ({len(results)} cells, {cap} captured)")
        print("E4_HARNESS_DONE")
        env.close()


if __name__ == "__main__":
    main()
