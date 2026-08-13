# B-DASH recovery / diversity data plan (v5-recovery)

Draft — 2026-06-25. Goal: lift the closed-loop pick→handoff grasp rate off the ~0/20–2/20 floor
without retraining the vision encoder.

## 1. Why this, and not more vision-tuning

The diagnostics (gated `BDASH_VLA_DEBUG` geometry probe, same seed-0 pegs each time) say the bottleneck
is **covariate shift at the grasp**, not the frozen encoder and not training duration:

| model | grasp | handoff | diagnostic behavior |
|---|---|---|---|
| v4 (frozen ViT) | 2/20 | 0 | descends to peg height but lands **12–14 cm off in xy** → regresses to the spawn-distribution mean |
| v4 step-28000 (frozen ViT) | 1/20 | 0 | same; longer training didn't help |
| v5-vision (`TUNE_VISUAL=true`) | 0/20 | 0 | **worse** — closes the gripper at a fixed step ~71 while 15–16 cm **above** the table, never descends (open-loop collapse from unfreezing a 2B ViT on 200 demos) |

Critically, the v4 data **already covers the L1 peg-position range** (peg x∈[0.40,0.60], y∈[−0.10,0.10]),
and `--arm_init_std` already randomizes the *start* pose. What the data never shows is the arm
**off-target in the middle of a descent, then correcting toward the peg**. The scripted expert
(`ScriptedPick`) always runs a clean `approach→descend→grasp` from a near-nominal configuration. So at
eval the closed-loop policy's slightly-wrong perception puts it in states the expert never demonstrated
→ it has no learned corrective action → it plows down at the mean (v4) or collapses (v5).

**Fix:** demonstrate recovery. Put the arm into off-target states during recording and let the
GT-driven expert correct from them. This is a cheap, in-recorder approximation of DAgger and directly
covers the failure states. Keep the vision encoder **frozen** (`TUNE_VISUAL=false`) — v5-vision proved
unfreezing on this data volume regresses.

## 2. New recorder capability: mid-trajectory perturbation injection

`scripts/bdash/record_vla_demos.py` today only perturbs the *start* (`--arm_init_std`, applied via the
`randomize_franka_joint_state` reset event). Add a one-shot **mid-episode joint kick** so the recorded
trajectory contains an off-target excursion followed by the expert's recovery.

New CLI knobs (default off → existing behavior unchanged):

- `--perturb_frac FLOAT` (default `0.0`): probability per episode that a kick is injected.
- `--perturb_mag FLOAT` (default `0.15`): max |Δ| (rad) applied to arm joints 1–6 (fingers untouched).
- `--perturb_phases` (default `APPROACH,DESCEND`): only kick while the pick is pre-grasp, so the
  recovery covers the grasp-relevant approach, not the transport.

Mechanism (in the recording loop, inside the existing `torch.inference_mode()` block): per episode, if
`rand() < perturb_frac`, schedule a kick step `k` ~ uniform over the early window. At step `k`, *before*
`policy.get_action`, if `policy._pick.phase[0] ∈ perturb_phases`, write a bounded random delta to the
robot's arm joint positions (`robot.write_joint_state_to_sim`, velocities zeroed), then continue. The
expert re-targets the GT grasp point every step, so it recovers automatically; the post-kick steps are
valid expert labels for off-distribution states. The recorder only exports episodes that still reach
`handoff`, so kicks that break the demo are auto-filtered (expect a modest drop in success rate / more
attempts). Optionally allow ≤2 kicks/episode.

Bounds keep it physical: ≤0.15 rad on 6 joints ≈ a few cm of EE excursion — enough to leave the demo
tube, small enough to stay reachable and collision-free.

This is additive and reversible; it does not touch the v4 recording path.

## 2b. Camera fix — side views are too far (do this in the SAME re-record)

Inspecting the v4 training frames (`datasets/.../lerobot/videos/.../{left,right,ego}_view`) plus the
intrinsics in `isaaclab_arena/embodiments/franka/franka.py`:

| view | offset (robot frame) | dist to peg | mm/px (H) | peg footprint |
|---|---|---|---|---|
| left_cam / right_cam | (0.05, ±0.57, 0.66), focal 2.8, 88° HFOV | **97 cm** | **7.3** | **~3–4 px wide** |
| ego/wrist (@hover z=0.27) | on panda_hand | 25 cm | 1.9 | ~13 px wide |

The side cameras are 97 cm back with an 88° wide-angle lens, so each frame spans ~1.9 m and the peg is
a **~3-pixel blue speck** lost in a sea of floor/table. L1 moves the peg over ~18 cm (~25 px), so the
model must resolve sub-cm grasp position from a 3-px blob — physically insufficient signal. This is a
prime suspect for the **mean-regression** failure (the close-up wrist view only helps once the arm is
already above the peg → chicken-and-egg with the side views).

**Fix (one number per camera):** raise `focal_length` 2.8 → ~6–8 on `left_cam`/`right_cam` (and
optionally `overhead_cam`) in `franka.py`. The cameras are already aimed at the workspace, so zooming
needs no re-aiming. At focal 8 the side view spans ~65 cm (vs 186 cm), the 20 cm L1 workspace still fits
with margin, and the peg footprint grows ~3 px → ~10 px wide. Verify the L1 extremes
(x∈[0.40,0.60], y∈[−0.10,0.10]) stay in-frame on a rendered reset before committing.

**The camera view is baked into the recorded demos**, so this requires regenerate + retrain — which is
exactly why it folds into the v5 re-record below (change cameras *and* add recovery data in one pass).
Consider this the **first lever to pull**: it's cheaper than recovery data and may be the dominant cause.

## 3. Dataset composition (target ~1200 demos, ~6× v4)

Record into one HDF5, `datasets/bdash/vla_pick_handoff_v5_recovery.hdf5`, L1 / c2.0, seed 0:

| slice | demos | flags | purpose |
|---|---|---|---|
| clean | ~600 | `--arm_init_std 0.02` | volume for the position→pixels→reach mapping |
| wide-start | ~300 | `--arm_init_std 0.15` | approach diversity |
| recovery | ~300 | `--arm_init_std 0.05 --perturb_frac 1.0 --perturb_mag 0.15` | the covariate-shift fix |

Rationale: ~6× the data alone often fixes "regression to mean," and the recovery slice covers the exact
off-target states v4 fails from. Keep peg range at the L1 default (0.10) so it stays comparable to the
eval cell; optionally widen to `pos 0.12` later so the eval range is strictly interior to training
(reduces edge extrapolation) — defer to a second pass.

Recording is the scripted expert + Isaac Sim only (no 3B training) → runs **locally** in
`isaaclab_arena-latest` on the 5080. v4 was 200 demos in ~25 min (100% first-attempt); ~1200 demos ≈
**~2.5–3 h** (recovery slice will need extra attempts). Record the three slices to separate temp HDF5s
or append; merge to one file before converting.

## 4. Convert + train (frozen ViT, N1.7)

1. Converter config `isaaclab_arena_gr00t/lerobot/config/bdash_pick_handoff_v5_recovery_config.yaml`
   (clone of `…_v4_config.yaml`, `hdf5_name → vla_pick_handoff_v5_recovery.hdf5`, output
   `datasets/bdash/vla_pick_handoff_v5_recovery/lerobot`). Convert in the **gn17** container.
2. Train `nvidia/GR00T-N1.7-3B` on the cloud A100/H100 (won't fit the 5080):
   ```
   TUNE_VISUAL=false TUNE_LLM=false TUNE_PROJECTOR=true TUNE_DIFFUSION=true \
   DATASET_PATH=datasets/bdash/vla_pick_handoff_v5_recovery/lerobot \
   OUTPUT_DIR=models/bdash-gr00t-n1-7-pick-handoff-v5-recovery \
   MAX_STEPS=20000 SAVE_STEPS=2000 SAVE_TOTAL_LIMIT=6 \
   bash scripts/bdash/train_vla.sh
   ```
   `TUNE_VISUAL=false` is deliberate (v5-vision regressed). Scale `MAX_STEPS` ~20k for the larger set;
   `SAVE_TOTAL_LIMIT=6` so we can A/B intermediate checkpoints and catch the best before overfit.

## 5. Eval + decision gates

Pull each saved checkpoint, wire `bdash_vla_v5_recovery` + config into `run_eval_grid.py` (clone the v4
entry, model_path only), eval at **c2.0 / L1, n=20** — same cell as every prior comparison. Track
**grasp-rate then handoff-rate** as the leading indicators (success needs the full chain; grasp/handoff
move first).

- **Working:** grasp-rate climbs from 2/20 toward >30% and handoffs start firing → recovery data is the
  right lever; scale demos further and/or tighten clearance.
- **Flat (~0):** the in-recorder perturbation isn't covering the real eval states → escalate to **full
  closed-loop DAgger**: roll out the *actual VLA* (`bdash_vla_v4`), log the states it visits, relabel
  each with the scripted expert (GT grasp point), append, retrain. More engineering; only if Phase 1
  doesn't move the needle.

## 6. Status / next action

- [x] **Camera fix** — side-cam `focal_length` 2.8→8 in `franka.py`; verified L1 extremes stay framed,
      peg ~3px→~12–15px (`logs/bdash/camcheck/`). 2026-06-25.
- [x] **Perturbation hook** — `--perturb_frac/--perturb_mag/--perturb_window` in `record_vla_demos.py`
      (+ `_apply_kick`). Smoke-tested (5 demos, `--perturb_frac 1.0`): every demo kicked, 5/6 reached
      handoff (1 auto-filtered), kick visible as a 0.141 rad spike + recovery, HDF5 structure == v4.
      2026-06-25.
- [x] **Recorded the 1200-demo mixed set** (600 clean 100% / 300 wide 100% / 300 recovery 95%) +
      merged → `datasets/bdash/vla_pick_handoff_v5_recovery.hdf5` (5.0GB, total_steps 221574). Verified:
      kick spike present only in the recovery region (0.135 vs 0.013 rad), keys consistent. 2026-06-25.
- [x] **Converted + pushed**: `bdash_pick_handoff_v5_recovery_config.yaml` → gn17 convert → 1200 parquet
      + 3600 mp4 (181MB) → HF **private** `umegan/vdash-pick-handoff-v5-recovery` (verified). 2026-06-25.
- [ ] Hand the train command to the cloud box (frozen ViT). See §4 — `TUNE_VISUAL=false`, batch 32,
      MAX_STEPS 20000 (~2.9 epochs over 221k samples), SAVE_TOTAL_LIMIT 10 for A/B.
- [ ] Eval each checkpoint at c2.0/L1 n=20; apply the §5 gate.

First runnable step is the recorder hook — small, local, reversible.
