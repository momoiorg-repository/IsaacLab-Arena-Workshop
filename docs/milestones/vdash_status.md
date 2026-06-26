# V-DASH — consolidated status (single-page overview)

**Updated:** 2026-06-26 · **Branch:** `an/vdash-vla-training` · **Targets:** figure freeze 7/2, paper 7/9
This is the one-page "全貌": the story, the verified results, what we can/can't claim, open decisions,
and where every artifact lives. Numbers are audit-verified (`share/vdash_paper_claims_audit.md`); all
**single-seed (0)** unless noted.

---

## TL;DR

A **diagnostic** result, not a high-success-rate result. We found the root cause of the long-standing
0% (a silent frozen-camera bug), fixed it, and then **decomposed the residual gap from first
principles**: classical control inserts a clean handoff at 98% with a wide convergence basin, so the
limiter is the **learned front-end's grasp quality** (the VLA holds the peg cocked ~8°/9 mm in the
gripper), which a §2.1 peg-pose-blind controller **provably cannot** correct downstream.

**Paper claim:** in precision assembly with a learned front-end + classical insertion, the binding
constraint is **handoff/grasp precision**, and the role split (learning = perception/grasp, classical =
precise insertion) is sound **but bounded by that precision** — quantified by the convergence basin and
clearance sweep.

---

## The causal chain (the spine of the paper)

```
frozen-frames fix (M10)  →  VLA perceives the peg (~5 mm tracking), grasps 85%, hands off 75%
classical control (M11)  →  inserts a CLEAN handoff at 98%; basin tolerates ~5 mm / ~8° of handoff error
BUT the VLA grasp is cocked ~8° / ~9 mm IN the gripper
§2.1 peg-blind controller cannot see/correct it  →  4/15 clean handoffs convert  =  20% end-to-end
and the same precision axis sets a ~2 mm clearance floor (E2 cliff)
```

---

## Verified results (with figures)

### 1. Frozen-camera root cause + fix — `M10.md`
Every prior VLA scored 0% because `sim.render()` did **not** refresh the RTX camera sensors on this box
(RTX 5080 / Isaac Sim 5.1, headless) — the policy trained/evaluated on a **single static reset frame**.
Fix: a render-coupled `sim.step(render=True)` on both the record and eval paths. Recorded frames went
from **1 unique → 108/108 unique**. First non-zero result followed.

### 2. Closed-loop hierarchy works — `M11.md §1` (c=2.0/L1, n=20)
| stage | v6-recovery |
|---|---:|
| grasp | 17/20 (85%) |
| clean handoff | 15/20 (75%) |
| **insert** | **4/20 = 20%**, CI95 [0.081, 0.416] |

Single model, single seed → **existence-level**.

### 3. Classical control is not the bottleneck — `M11.md §2-3`, `figs/vdash/e4_basin.png`
- Scripted (clean handoff) insert: **49/50 = 98%** at c2.0/L1 (f_ins 0.6 N); 50/50 at L0.
- Convergence basin @ c=2.0 (L0 = fixed scene → deterministic; each cell = mean over 4 azimuths):
  **held 37% vs recovery 83%** — the real controller recovers lateral offset to **≥5 mm** and tilt to
  **~8°**. RCC rotational compliance (the only tilt term) **makes it worse** (83→29%); shipped off.

### 4. Residual = grasp quality (in-gripper) — `M11.md §4`, `figs/vdash/ingripper.png`
| in-gripper at handoff | scripted (98%) | VLA v6 (20%) |
|---|---|---|
| tilt | 0.1–0.6° | **~8°** (3.3–11.8°) |
| lateral | 0.7–0.8 mm | **~9 mm** (5.4–13.1 mm) |

Rigid grasp → the controller's nominal tip-estimate is wrong by ~10–17 mm → it presses the peg in
canted (failure final-tilt mean 32° / median 19° / max 83°).
VLA handoff (logger, n=15): tilt mean 5.6° / median 4.9° / max 15.7°; \|dxy\| median 6.5 mm.

### 5. Unfixable downstream (§2.1 limiter) — `M11.md §5`, `figs/vdash/cheat.png`
8 mm in-gripper offset (scripted test-bed), c2.0/L1, n=20:
baseline 0/20 · +wider spiral 0/20 · +slow spiral 0/20 · +RCC translational 0/20 · **+true tip (cheat)
11/20 = 55%**. The controller's capture logic runs in *assumed-tip* coordinates, so it can't even detect
the true tip 9 mm away. Only knowing the tip helps — and caps at 55% (peg still physically off-axis).
Reseat is additionally blocked by the binary gripper (open=drop, close=rigid).

### 6. E2 clearance cliff — `figs/vdash/e2_clearance.png`
Scripted, n=50: c=2.0 → **100% (L0) / 98% (L1)**; **c≤1.0 → 0% at every level**. Verified precision-
limited, not a config gap (c=1.0 with `use_hop_search` ON still 0/50, f_ins 31 N). The ~4 mm scripted
handoff lateral exceeds what fits a ≤1 mm hole → **classical baseline has a ~2 mm clearance floor**.

---

## What we can / can't claim

**Strong (publishable as-is):**
- Frozen-frames root-cause discovery + fix (method + lesson).
- Classical control covers a wide convergence basin (98% from a clean handoff at c=2.0).
- First-principles decomposition: residual = front-end grasp quality; §2.1 makes it unfixable downstream.
- Methodology: convergence-basin harness (held/recovery), in-gripper diagnostic, cheat ablation.

**Do NOT over-claim:**
- High closed-loop success — 20% is one model / one seed (existence).
- Tight-clearance insertion — **c≤1.0 is 0%**; precision floor ~2 mm.
- "Predicate switching" — true for the **VLA** path only (scripted/M8 switch on `pick_finished`); the
  §3.4 predicate is ground-truth and has no uprightness term.

---

## Open decisions / next items

1. **Reproducibility (audit #1):** the 98%/20%/basin numbers came from an **uncommitted** working tree;
   `insertion_controller.py` was edited after some runs. **Freeze code+config on a commit and re-run**
   before any number is load-bearing. *(Not committed — awaiting your go.)*
2. **Main-result framing:** anchor the paper on **c=2.0** (current achieved domain), or pursue sub-mm?
   Sub-mm needs tightening the **scripted handoff precision** (a precision-pick task) — won't make 7/2.
3. **Multi-seed the VLA 20%** if it's cited as more than existence (carries ~5–6% image-aug variance).
4. **c3.0 socket** must be generated if a looser-clearance E2 point is wanted (series ships only
   {2.0, 1.0, 0.5, 0.25}; `--clearance 3.0` silently snaps to 2.0).
5. **Deferred to next VLA iter (request v3 / Sept):** grasp **quality** (not rate) — the actual lever.

---

## Artifact index (where everything lives)

**Docs / narrative**
- `docs/milestones/M10.md` — frozen-frames breakthrough (corrected; §6 superseded by M11)
- `docs/milestones/M11.md` — insertion diagnosis (the headline result)
- `docs/milestones/V-DASH_next_actions_M10.md` — the action plan this work executed
- `share/vdash_paper_claims_audit.md` — every number traced to source, discrepancies flagged
- `docs/physics_params.md` — physics provenance (contact_offset reconciled to 0.0002)

**Figures** — `figs/vdash/` (`README.md` = index + headline numbers)
- `ingripper.png` (grasp quality) · `cheat.png` (tip-error proof) · `e4_basin.png` (held vs recovery) ·
  `e2_clearance.png` (clearance cliff). Regenerate: `scripts/vdash/plot_paper_figs.py`.

**Live data** — `results/vdash/`
- `eval_grid.csv` (scripted 98% headline) · `e2_clearance_final.csv` (E2 sweep) ·
  `e4_{held,recover}_c2.0_s0..s4.csv` (basin) · VLA eval JSONL
  `logs/vdash/vdash_handoff_grid_vdash_vla_v6_recovery_c2.0_L1_20260625_233557.jsonl`
- In-gripper diagnostic logs: `logs/vdash/graspdiag_{scripted,vla}.out`
- Cheat/test-bed logs: `logs/vdash/cock8_{test,cheat}.out`

**Code (clean, on this tree)**
- `isaaclab_arena/controllers/insertion_controller.py` — recovery-basin mode (`set_m7(...,recover=True)`),
  `use_hop_search` default fixed True→False
- `isaaclab_arena/policy/{vdash_vla_policy,vdash_scripted_policy}.py` — `VDASH_GRASP_DIAG` instrument
- `isaaclab_arena/evaluation/policy_runner.py` & `scripts/vdash/record_vla_demos.py` — camera fix
- `scripts/vdash/{run_eval_grid,run_convergence_zone,plot_paper_figs,train_and_push_v6_recovery}.py`

**Model / dataset (HF, private, umegan)**
- model `umegan/vdash-gr00t-n1-7-pick-handoff-v6-recovery` (local `models/…v6-recovery`, 12 GB)
- dataset `umegan/vdash-pick-handoff-v5-recovery` (1000 demos, camera-fixed/moving frames)

**Full chronology:** the assistant's working memory
`~/.claude/projects/-home-an-Workspace/memory/vdash-vla-training-pipeline.md` (complete timeline v3→v6
+ diagnosis + finalization).

---

## Housekeeping (stale / duplicate data — safe to ignore or prune)
- `results/vdash/e4_convergence_c3.0.csv` — **mislabeled**, is actually c2.0 (clearance snapped).
- `e4_convergence_c2.csv`, `e4_convergence_c1.0.csv`, `e4_dbg.csv`, `e4_smoke.csv`, `e2_clearance.csv`
  (old), `eval_grid_dryrun.csv`, `eval_v3/v4/v4_28000/v5_vision.csv` — superseded by the runs above.
- `e4_recover_rcc_c2.0.csv` — the "RCC hurts" run (29%); keep as evidence for M11 §3.
- Paused/shelved: full pick→insert recording (`datasets/vdash/_v5_full_parts/full_clean.hdf5`, 500-demo
  slice) — the next-actions doc says **don't** run the end-to-end A/B; left on disk, not used.
