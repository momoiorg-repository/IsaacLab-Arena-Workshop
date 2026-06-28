# V-DASH — How to Run a Policy (cheatsheet)

Closed-loop evaluation of policies in the `vdash_pick_insert` task.
**Run these from inside the GR00T container** (`isaaclab_arena-cuda_gr00t_gn16`), at the repo root `/workspaces/isaaclab_arena`.

## Prereqs (every command)

- Be inside the **GR00T container** — VLA inference needs the GR00T deps.
- **`unset DISPLAY`** first — otherwise Kit stalls ~7 min probing a dead X server (vs ~12 s).
- Interpreter is **`/isaac-sim/python.sh`**.
- VLA policies force `--num_envs 1` + cameras + the `franka_joint` env automatically.

```bash
unset DISPLAY    # run once per shell
```

Outputs: per-episode JSONL in `logs/vdash/` (source of truth), aggregated CSV in `results/vdash/eval_grid.csv`.

---

## Runnable now

### 1. Rule-based baseline (no model needed)

```bash
/isaac-sim/python.sh scripts/vdash/run_eval_grid.py \
  --policies vdash_scripted --clearances 2.0 --levels L1 --num_episodes 5
```

### 2. VLA hierarchy — v2 handoff model (local)

VLA does pick→handoff, then the rule-based `InsertionController` inserts. Uses `models/vdash-gr00t-n1-6-pick-handoff`.

```bash
/isaac-sim/python.sh scripts/vdash/run_eval_grid.py \
  --policies vdash_vla --clearances 2.0 --levels L1 --num_episodes 5
```

> ⚠️ v2 was trained on the old **cylinder** peg → ~0% handoff against the current **square** peg env. Needs the v3 retrain.

---

## Grid sweeps

```bash
# Full grid: clearances {2.0,1.0,0.5,0.25} × levels {L0,L1,L2,L3} × 50 episodes
/isaac-sim/python.sh scripts/vdash/run_eval_grid.py --policies vdash_scripted

# Multiple policies + custom cells
/isaac-sim/python.sh scripts/vdash/run_eval_grid.py \
  --policies vdash_scripted vdash_vla --clearances 2.0 1.0 --levels L1 --num_episodes 20

# Re-aggregate CSV from existing logs (no sim re-run)
/isaac-sim/python.sh scripts/vdash/run_eval_grid.py \
  --policies vdash_vla --clearances 2.0 --levels L1 --num_episodes 5 --aggregate_only

# Force re-run even if a complete JSONL exists
/isaac-sim/python.sh scripts/vdash/run_eval_grid.py ... --no_resume
```

---

## Watch it live (WebRTC)

Call `policy_runner.py` **directly** (the grid runner is resumable and will `skip` a cell that already has a
complete JSONL — use `--no_resume` to force it). Set `LIVESTREAM=2` and **still `unset DISPLAY`** — a stale
`DISPLAY` stalls Kit ~7 min even for WebRTC, which needs no X. Then open the Isaac Sim WebRTC client
(host ports 49100 signaling + 8011).

```bash
unset DISPLAY
LIVESTREAM=2 /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena.policy.vdash_vla_policy.VDashVLAPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/vdash_pick_insert_gr00t_closedloop_config.yaml \
  --enable_cameras --num_episodes 5 --num_envs 1 --seed 0 \
  vdash_pick_insert --clearance 2.0 --level L1 --embodiment franka_joint
```

---

## Pull a model from HF, then run

### v3 handoff model (when the upload finishes — repo currently has no weights)

```bash
hf download umegan/vdash-gr00t-n1-6-pick-handoff-v3 \
  --local-dir models/vdash-gr00t-n1-6-pick-handoff-v3

# point model_path (line 16) at the v3 dir, then run the §2 vdash_vla command:
sed -i 's#models/vdash-gr00t-n1-6-pick-handoff$#models/vdash-gr00t-n1-6-pick-handoff-v3#' \
  isaaclab_arena_gr00t/policy/config/vdash_pick_insert_gr00t_closedloop_config.yaml
```

### insert-full model — end-to-end (VLA does the insertion too, NOT the hierarchy)

✅ Already installed: model at `models/vdash-gr00t-n1-6-pick-insert-full/`, config at
`isaaclab_arena_gr00t/policy/config/vdash_pick_insert_full_gr00t_closedloop_config.yaml`.
Run it directly with the plain `Gr00tClosedloopPolicy`:

```bash
unset DISPLAY
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/vdash_pick_insert_full_gr00t_closedloop_config.yaml \
  --enable_cameras --num_episodes 5 --num_envs 1 --seed 0 \
  --log_dir logs/vdash --run_tag insertfull \
  vdash_pick_insert --clearance 2.0 --level L1 --embodiment franka_joint
```

To watch it live, prefix with `LIVESTREAM=2` (keep `unset DISPLAY`).

To (re)create the config from scratch:

```bash
hf download umegan/vdash-gr00t-n1-6-pick-insert-full \
  --local-dir models/vdash-gr00t-n1-6-pick-insert-full
sed -e 's#models/vdash-gr00t-n1-6-pick-handoff#models/vdash-gr00t-n1-6-pick-insert-full#' \
    -e 's#language_instruction:.*#language_instruction: "Pick up the peg and insert it into the socket."#' \
  isaaclab_arena_gr00t/policy/config/vdash_pick_insert_gr00t_closedloop_config.yaml \
  > isaaclab_arena_gr00t/policy/config/vdash_pick_insert_full_gr00t_closedloop_config.yaml
```

---

## Models inventory

| Name | Where | Policy to use |
|---|---|---|
| `vdash-gr00t-n1-6-pick-handoff` (v2) | local + HF | `vdash_vla` (hierarchy) |
| `vdash-gr00t-n1-6-pick-handoff-v3` | HF (⚠️ empty, no weights) | `vdash_vla` (hierarchy) |
| `vdash-gr00t-n1-6-pick-insert-full` | local + HF | `Gr00tClosedloopPolicy` (end-to-end) |
| *(none)* — `vdash_scripted` | rule-based baseline | `vdash_scripted` |

## Gotchas

- **Root-owned `results/vdash/`** → CSV write crashes *after* episodes run (JSONL is safe). Fix from the host:
  `docker exec isaaclab_arena-cuda_gr00t_gn16 chown -R an:an /workspaces/isaaclab_arena/results/vdash`
- **Square-peg incompatibility**: models trained on the old cylinder peg (v2, insert-full) do poorly on the current square-peg env. Use square-peg-trained weights (v3).
- Metric to watch: `t_handoff` rate (handoff fired) — where the VLA grasp quality shows.
