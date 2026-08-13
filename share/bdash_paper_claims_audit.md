# B-DASH Paper-Claims Verification Audit

**Date:** 2026-06-26 · **Mode:** read-only (no code/behavior changed) · **Branch:** `an/bdash-vla-training`

Every number below is taken from the actual source / CSV / JSONL on disk, not from memory or the prose
docs. Where a doc and the data disagree, the data wins and the disagreement is flagged.

---

## A. Audit table (claim → source → verdict)

| # | Claim in the draft | Source of truth | Verdict |
|---|---|---|---|
| 1 | 98% scripted insertion = 49/50 at c2.0/L1 | `results/eval_grid.csv`, jsonl `..._20260626_005107.jsonl` | **VERIFIED** (n=50, seed 0, scripted, no cameras) |
| 1 | "…used current controllers.yaml, no uncommitted changes" | `git diff` | **DISCREPANCY** — controllers.yaml **and** insertion_controller.py are uncommitted; headline not bit-reproducible from any commit |
| 1 | InsertionController uses only socket pose + tip_estimate + wrist F/T | `insertion_controller.py:125,126,135-137,162` | **VERIFIED** — no peg pose in control path |
| 1 | Peg ground-truth used only for predicates/logging | `bdash_peg_predicates.py:121-136`, `bdash_handoff_logger.py:150-189` | **VERIFIED** |
| 2 | Handoff switch = §3.4 state predicate ("verifiable") | VLA: `bdash_vla_policy.py:202` / Scripted: `bdash_scripted_policy.py:85` | **PARTIAL** — true for VLA path; **scripted & M8 baselines switch on `pick_finished`, not the predicate** |
| 2 | Predicate signals | `bdash_peg_predicates.py:89-118` | **CAVEAT** — evaluated on **ground-truth** peg pose+velocity; **no "upright"/tilt term** |
| 3 | 0.5° scripted handoff tilt; 4.1mm lateral | jsonl `...scripted...005107` | **VERIFIED** (tilt mean 0.50°, \|dxy\| mean 4.06mm) |
| 3 | 3.7° / 15.7° VLA tilt; ~6.5mm lateral | jsonl `...vla_v6_recovery...233557` | **PARTIAL** — \|dxy\| 6.5mm ✓; **15.7° is the max not a typical**; **3.7° not reproducible** (mean 5.6°/median 4.9°) |
| 4 | ScriptedPick reads peg true pose (expert, allowed) | `scripted_pick.py:58-61` | **VERIFIED** |
| 4 | scripted_pick reads true yaw even on grasp_override path | `scripted_pick.py:56-57,76-78` | **CONTRADICTS BELIEF** — no longer reads yaw at all (fixed wrist); override path reads no peg pose |
| 4 | VLA gets only RGB + proprio, no peg pose in state | `franka_modality_config.py`, `gr00t_9dof_joint_space.yaml` | **VERIFIED** |
| 4 | M8 baseline grasp = vision, no GT/yaw leak | `bdash_rule_based_policy.py:91-97`, `bdash_classic_vision.py:8,33` | **VERIFIED** |
| 5 | Camera fix on both record + eval paths, gated | `policy_runner.py:89-99`, `record_vla_demos.py:223-225` | **VERIFIED** |
| 5 | Frames actually vary per step | HDF5 check (ran) + `smoke_fix.out` + `eval_v6_recovery.out` | **VERIFIED** |
| 6 | Physics = Factory/FORGE-derived, no drift | `env_callbacks.py:92-115`, `configs/bdash/peg_insert/assets.yaml`, `bdash_peg_assets.py:36` | **VERIFIED w/ 2 documented deltas** (decimation 2≠8; contact_offset 0.0002≠doc 0.005) |
| 7 | n / seed / Wilson CI per headline number | CSVs + recompute | **VERIFIED**; **all single-seed**; **L0 is unstable** |

---

## 1. The 98% scripted-handoff insertion number

**Provenance.** The 49/50 at c2.0/L1 appears in two runs, both `bdash_scripted`:

- **Headline (most recent):** `results/bdash/eval_grid.csv` (written 2026-06-26 09:52):
  `bdash_scripted,2.0,L1,50,49,0.98,0.895,0.9965,0.605,50` → jsonl
  `logs/bdash/bdash_handoff_grid_bdash_scripted_c2.0_L1_20260626_005107.jsonl`
  (50 records; 49 `success`, 1 `insertion_failed`; the failure is episode 1, force_max 63.4 N).
  *(Filename ts `005107` is container-UTC; host mtime 09:51 JST = the same instant, +9h. No anomaly.)*
- **Earlier:** `results/bdash/e2_clearance.csv` (working tree) L1 → `..._20260621_111959.jsonl`, also 49/50.

- **n = 50** — `configs/bdash/peg_insert/eval_grid.yaml: num_episodes: 50`; jsonl has exactly 50 lines.
- **seed = 0** — `eval_grid.yaml: seed: 0`; `run_eval_grid.py` passes `--seed 0`. **Single seed.**
- **Scripted, not VLA** — policy `bdash_scripted` (`BDashScriptedPolicy`). For non-VLA policies
  `run_eval_grid.run_cell` sets `pre_env = []` (no `--enable_cameras`) and `--num_envs 8`. In
  `policy_runner.py:89` `cam_eval = uenv.sim.has_rtx_sensors()` → False, so the render-coupled step is
  skipped. **No cameras / no VLA in the loop. Confirmed.**

**InsertionController inputs (control path) — `insertion_controller.py`:**
- `tip = tip_estimate(env, grip_offset)` (`:125`) → `ee_control.py:34-46`: uses only the `ee_frame`
  sensor pose + the `grip_offset` constant. **Proprioceptive; no peg pose.**
- `cur_ee, cur_quat = read_ee_pose(env)` (`:126`) → `ee_frame` sensor (`ee_control.py:23-25`).
- socket: `sock.root_pos_w[:, :2]`, `mouth_z` (`:135-137`) — **jig-fixed fixture pose, allowed (§2.1).**
- wrist F/T: `W = _wrench(env, names["peg_sensor"])` (`:162`, def `:40-47`) = `net_forces_w` summed on
  the peg contact sensor. **The only contact signal.**
- M7 offset/tilt come from the harness via `set_m7(...)`, not from peg pose.
- **The controller never reads `peg.root_pos_w`/`root_quat_w` for control.** The module docstring
  states this explicitly (`:16-23`).

**Peg ground-truth is used only for predicates/logging:**
- success termination = `inserted()` (`bdash_peg_predicates.py:121-136`) reads peg tip + socket + clearance.
- `bdash_handoff_logger.py:150-189` reads `peg.root_quat_w`/tip for handoff dx/dy/tilt, depth, final tilt.
- `bdash_scripted_policy.py:104-124` `_dbg()` reads peg pose — **debug print only.**
- None of these feed the action.

### ⚠ Discrepancy 1 — "current controllers.yaml, no uncommitted local changes" is FALSE
- `git diff configs/bdash/peg_insert/controllers.yaml`: **+24/−2 uncommitted** — adds the M5a `use_hop_search:`
  block (default `false`) + reworded jam/retry comments.
- `git diff isaaclab_arena/controllers/insertion_controller.py`: **+76/−15 uncommitted**, and the file
  was **last edited 2026-06-26 10:33 — *after* the 09:51 headline run** (during the M7/recover work).
- The **committed** tree has **no `use_hop_search` key**, while the code reads
  `c.get("use_hop_search", True)` (`insertion_controller.py:315`) — i.e. **default True**. So committing
  the controller without also committing the yaml's `use_hop_search: false` would silently flip
  hop-search ON and change c2.0 behavior.
- **Consequence:** the exact code+config that produced 49/50 is not captured in any commit and is not
  bit-reproducible. The number is internally consistent (two independent runs agree) but should be
  re-run and frozen on a clean commit before it goes in the paper.

---

## 2. The handoff switch — what actually flips control

- **VLA hierarchy (the paper's VLA results):** `bdash_vla_policy.py:202`
  `self._in_insertion = self._in_insertion | vp.handoff(uenv, **self._hp)` — latches on the **§3.4
  predicate**. ✅ matches "verifiable state-predicate switching".
- **Scripted policy (the 98% headline):** `bdash_scripted_policy.py:84-85`
  `self._in_insertion = self._in_insertion | pick_finished` — latches on the **ScriptedPick state
  machine reaching DONE** (TRANSPORT waypoint reached, `scripted_pick.py:119,122`). **NOT the predicate.**
- **M8 rule-based baseline:** `bdash_rule_based_policy.py:103` — also `pick_finished`. **NOT the predicate.**

> The §3.4 predicate *is* evaluated every step by the logger in all runs (hence `handoff_fired=50` in
> the CSV), so the two coincide in practice for the scripted flow — but the **control switch** in the
> scripted/M8 paths is an internal pick flag, not the predicate.

**What the predicate reads (`bdash_peg_predicates.py:89-118`):** grasped ∧ tip in cylinder ∧ speed<max,
computed from **ground-truth**: `peg.root_pos_w`/`root_quat_w` (tip, `:37-42`), `peg.root_lin_vel_w`
(speed, `:117`), `socket.root_pos_w` (`:45-49`), gripper joint_pos + finger-contact force. Module
docstring `:3`: *"all decided from simulator state."*

### ⚠ Discrepancy 2 — two honesty caveats for the paper
1. The predicate is evaluated on **privileged ground-truth** peg pose/velocity, **not perceived/estimated
   state**. "Verifiable state predicate" is fair; "from perception" would not be.
2. The predicate has **no "upright"/tilt term** — only `grasped ∧ in-cylinder ∧ speed<speed_max`. If the
   draft says the switch checks uprightness, that is not implemented.
3. The "state-predicate switching" claim only describes the **VLA** path. The 98% headline uses the
   **scripted** path, which switches on `pick_finished`. State this precisely if the 98% is cited as
   evidence of predicate switching.

---

## 3. Tilt / lateral numbers (computed from the JSONL)

Same measurement for both paths: the logger computes tilt = `acos(|peg_axis·ẑ|)` from `peg.root_quat_w`
(handoff `bdash_handoff_logger.py:154-162`; final `:185-189`) and `|dxy|` from tip−socket_axis at the
**same** `new_handoff` instant. Identical code for scripted and VLA. ✅ same definition, same frame.

| Metric (c2.0/L1) | Scripted (`...005107`, n=50) | VLA v6-recovery (`...233557`, n=20, 15 handoffs) |
|---|---|---|
| handoff tilt mean / median / max | **0.50° / 0.39° / 1.63°** | **5.62° / 4.87° / 15.68°** |
| handoff \|dxy\| mean / median | **4.06mm / 3.79mm** | **7.01mm / 6.50mm** |
| handoff dx mean / median | −2.59 / −3.23mm | −0.66 / −0.81mm |
| final tilt mean / median | 0.49° / 0.38° | 32.3° / 18.7° |

- "0.5° scripted" ✅ · "4.1mm scripted" ✅ (mean 4.06mm).
- "~6.5mm VLA" ✅ (median 6.50mm).
- **"15.7° VLA"** = the **single worst-episode** handoff tilt (max 15.68°), **not a mean** — label it.

### ⚠ Discrepancy 3 — VLA "3.7° / −5.7mm" not supported by the log
`docs/milestones/M10.md` §6 states *"Handoff quality… dx ≈ −5.7 mm, tilt ≈ 3.7°."* The cited
v6-recovery JSONL gives **handoff tilt mean 5.62° / median 4.87°** and **dx mean −0.66 / median −0.81mm**.
No subset reproduces 3.7° / −5.7mm (success-only subset: tilt 2.98°, dx −0.46mm). The "~5–6 mm" in M10
§6 is the *EE→peg tracking distance* from the `BDASH_VLA_DEBUG` stream, a different quantity than the
handoff lateral error. **Recompute the VLA handoff tilt/lateral straight from the JSONL before quoting.**

---

## 4. Privileged-information audit (whole pipeline)

- **ScriptedPick (expert/data-collection):** `_grasp_point` reads `peg.root_pos_w` + `root_quat_w`
  (`scripted_pick.py:58-61`). Privileged, **allowed**. State plainly.
- **Yaw question — contradicts the stated belief.** scripted_pick **no longer reads true yaw**. The
  uncommitted diff removed `level_to_down(cur_quat)` and now holds a **fixed** wrist
  `q_down0 = (0,1,0,0)` (`:76-78`). On the `grasp_override` path `_grasp_point` returns the override and
  **reads no peg pose at all** (`:56-57`). So there is **no yaw leak** anywhere (cylinder grip is
  yaw-symmetric). The belief "still reads true yaw unconditionally" is **out of date** — good for E3,
  but note it is uncommitted.
- **VLA (GR00T) input** — `isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py`:
  `video = [ego_view, left_view, right_view]`, `state = [single_arm, gripper]`, `language`.
  `state` maps to `gr00t_9dof_joint_space.yaml` = the 7 panda arm joints + 2 finger joints **only**.
  **No peg/socket pose channel.** ✅ This is the exact leak you wanted to rule out — it is absent.
- **M8 rule-based baseline** — `bdash_rule_based_policy.py:91-97`: grasp point comes from
  `ClassicVisionGrasp` (colour segmentation + plane back-projection), cached to `grasp_override`. The
  vision module docstring (`bdash_classic_vision.py:8,33`) states it must not use ground-truth peg pose;
  a quick scan shows no `peg.root_*` reads (only the known camera calibration). Orientation = the same
  fixed yaw-agnostic wrist. **No GT leak, no yaw leak → E3 comparison valid on that axis.**

---

## 5. Camera-fix correctness

- **Eval path:** `policy_runner.py:89` `cam_eval = uenv.sim.has_rtx_sensors()`; `:95-99` after
  `env.step` → `sim.step(render=True)` + `scene.update` + `observation_manager.compute()`. **Gated.**
- **Record path:** `record_vla_demos.py:223-225` — same render-coupled pump (unconditional; recording
  always has cameras).
- **Gate leaves non-camera runs untouched:** `has_rtx_sensors()` is False for the scripted 98% run, so
  the block is skipped. ✅

**Frame-uniqueness (ran it):**
- Recorded dataset `datasets/bdash/vla_pick_handoff_v5_recovery.hdf5` (the v6 training set):
  `demo_0` → left/right/wrist each **108/108 unique frames**, mean consecutive abs-diff 2.0–3.3;
  `demo_1` → **105/105 unique**. **All frames vary.** ✅
- Record-path runtime: `logs/bdash/smoke_fix.out` `framediff` 0.68–3.2 (post-fix) vs
  `smoke_v5_diag.out` `live-framediff=0.0000` (the pre-fix frozen state). ✅ clean contrast.
- Eval-path runtime: `logs/bdash/eval_v6_recovery.out` shows the VLA driving EE→peg `dxy` 0.081→0.010
  m over a pick (steps 1→41). A frozen-frame model regresses to the ~0.12–0.14 m spawn mean; converging
  to 0.01 m is only possible on live frames. ✅ (indirect but conclusive).

---

## 6. Physics-setting provenance

**SimulationCfg** (`env_callbacks.py:92-115`, `bdash_assembly_env_cfg_callback`, wired at
`bdash_pick_insert_environment.py:125`) matches `docs/physics_params.md` §1:
dt 1/120 ✓, gravity −9.81 ✓, TGS solver_type 1 ✓, pos_iter 192 ✓, vel_iter 1 ✓, bounce 0.2 ✓,
friction_offset 0.01 ✓, friction_corr 0.00625 ✓, GPU caps 2²³/2²⁸ ✓, partitions 1 ✓, material 1.0/1.0 ✓.

**Per-asset:** peg `max_depenetration_velocity = 1.0` (`bdash_peg_assets.py:36`) matches §7 M3b. Socket SDF
`sdf_resolution: 1024`, `rest_offset: 0.0` (`configs/bdash/peg_insert/assets.yaml:27,29`). ✅

### Documented deltas (with reasons)
1. **decimation = 2** (`env_callbacks.py:115`, 60 Hz control), vs Factory's documented **8** (§1). In-code
   rationale: "supports the 60–120 Hz insertion controller; VLA subsamples." Deliberate, not drift.
2. **contact_offset = 0.0002 m** (`configs/bdash/peg_insert/assets.yaml:26`), vs `physics_params.md` §2 which lists
   the Factory value **0.005**. Inline reason: "must be < tightest clearance (0.25 mm); 5 mm (Factory)
   blocks entry." Deliberate; **physics_params.md §2 was never reconciled to 0.0002** → doc gap, not drift.

### No drift during the tilt/insertion work
- `configs/bdash/peg_insert/assets.yaml` (the source-of-truth socket params) is **unmodified**.
- `assets/bdash/peg_insert/config.yaml` *is* modified, but it is the **MeshConverter per-asset record** for the
  last asset generated (now `peg.usd`, convexDecomposition — correct for a solid peg). It is a transient
  artifact, **not** the socket's runtime physics; the socket still uses SDF via `assets.yaml`. `.asset_hash`
  changed because assets were regenerated 2026-06-23. **No physics-parameter drift.**

---

## 7. Statistics hygiene

**All eval-grid / e2 cells are SINGLE SEED (seed 0).** Wilson 95% CIs (recomputed, match the CSV):

| Number | k/n | rate | Wilson 95% | seeds |
|---|---|---|---|---|
| Scripted insert, c2.0/**L1** (headline) | 49/50 | 0.980 | **[0.895, 0.997]** | 1 (seed 0) |
| Scripted insert, c2.0/**L0** (working tree) | 28/52 | 0.538 | [0.405, 0.667] | 1 (seed 0) |
| Scripted insert, c2.0/L0 (committed, Jun 13) | 50/50 | 1.000 | [0.929, 1.0] | 1 (seed 0) |
| VLA v6-recovery, c2.0/L1 | 4/20 | 0.200 | [0.081, 0.416] | 1 (seed 0) |
| Convergence map (static hold), c2.0 | 44/117 | 0.376 | [0.294, 0.467] | 1 (seed 0) |
| Recovery basin, c2.0 (`e4_recover_c2.0`) | 98/117 | 0.838 | [0.760, 0.894] | 1 (seed 0) |

**Single vs multi-seed.** Every headline is single-seed. The documented 5–6% image-aug variance applies
to the **VLA** numbers (4/20) — treat as soft, multi-seed not run. The scripted numbers have no image
path, so seed-0 determinism should hold *for a fixed code/config* — but see L0 below.

### ⚠ Discrepancy 4 — L0 is unstable across runs at the same seed
Same seed 0, c2.0/L0: **50/50 (committed Jun 13) → 28/52 = 54% (working tree Jun 21)**; M10's
next-actions doc cites an "85%" run too. The spread is **not** seed or image variance — it is
**code/config change between runs** (the uncommitted controller/yaml tuning). **Do not cite a single L0
number.** The **L1** number is stable (100% → 98% → 98%) and is the safe headline.

### ⚠ Discrepancy 5 — scripted L1/L2/L3 are not independent
For the **cameraless** scripted policy, L2 adds lighting and L3 adds texture+distractors
(`bdash_peg_randomization.py:9-13`) — none of which the controller perceives. The CSV shows L2≡L3 (both
49/50, identical force median 7.582). Treat L1/L2/L3 for the scripted baseline as **one** result (98%)
replicated, not three independent points.

### ⚠ Discrepancy 6 — there is no c3.0 socket; "c3.0" files are c2.0
`results/bdash/e4_convergence_c3.0.csv` is **byte-identical** to `e4_convergence_c2.0.csv` and its
`clearance_mm` column reads **2.0**. `bdash_pick_insert_environment.py:50,163-165` snaps `--clearance`
to the nearest shipped socket `(2.0, 1.0, 0.5, 0.25)`, so **`--clearance 3.0` silently ran at 2.0**. The
next-actions doc's "c3.0+ re-evaluation" task was **not actually run at 3.0 mm** — a wider socket must be
generated first.

---

## What would weaken a draft claim (read this before the figure freeze)

1. **Reproducibility (highest priority).** The 49/50 was produced from an uncommitted working tree, and
   `insertion_controller.py` was edited *after* that run. With `use_hop_search` defaulting to `True` in
   code but `false` only in the uncommitted yaml, a clean checkout does **not** reproduce the run. Freeze
   code+config on a commit and re-run before the number is load-bearing.
2. **"Predicate switching" applies to the VLA path only.** The 98% scripted headline switches on
   `pick_finished`, and the predicate it logs is ground-truth (not perception) and has no uprightness
   term. Phrase the SELF-VLA differentiation around the VLA hierarchy, precisely.
3. **VLA handoff tilt/lateral.** Quote from the JSONL (tilt ~5.6° mean / 4.9° median, max 15.7°; \|dxy\|
   ~6.5mm), not M10's "3.7° / −5.7 mm," which don't reconcile with the log.
4. **L0 instability + single seed.** Use L1 as the scripted headline; if L0 appears, show the run-to-run
   range. Mark every number single-seed; multi-seed the VLA if the 20% is cited as more than existence.
5. **Doc gap:** `physics_params.md` §2 still lists contact_offset 0.005 while the env uses 0.0002 — fix
   the doc so a reviewer auditing "Factory-derived physics" doesn't catch the mismatch.

---

### Files cited
`isaaclab_arena/controllers/insertion_controller.py`, `.../ee_control.py`, `.../scripted_pick.py`,
`.../bdash_classic_vision.py`; `isaaclab_arena/policy/{bdash_scripted_policy,bdash_vla_policy,bdash_rule_based_policy}.py`;
`isaaclab_arena/metrics/bdash_handoff_logger.py`; `isaaclab_arena/evaluation/policy_runner.py`;
`isaaclab_arena_environments/mdp/{bdash_peg_predicates,env_callbacks,bdash_peg_assets,bdash_peg_randomization}.py`;
`isaaclab_arena_environments/bdash_pick_insert_environment.py`;
`isaaclab_arena_gr00t/embodiments/franka/{franka_modality_config.py,gr00t_9dof_joint_space.yaml}`;
`configs/bdash/peg_insert/{controllers.yaml,eval_grid.yaml,assets.yaml}`; `scripts/bdash/{run_eval_grid,run_convergence_zone,record_vla_demos}.py`;
`results/bdash/{eval_grid,e2_clearance,e4_convergence_c2.0,e4_convergence_c3.0,e4_recover_c2.0}.csv`;
`logs/bdash/*.jsonl`, `*.out`; `datasets/bdash/vla_pick_handoff_v5_recovery.hdf5`; `docs/physics_params.md`, `docs/milestones/M10.md`.
