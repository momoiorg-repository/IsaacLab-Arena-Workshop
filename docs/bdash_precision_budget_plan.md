# B-DASH: precision-budget theory + §2.1-honest interface-widening — research plan

**Status:** active (2026-06-27). Supersedes the "defer grasp fix to Sept" stance in `M11.md §6`.
Synthesized from a multi-agent design campaign (understand / novelty / design / adversarial critique);
raw findings saved under the session workflow result.

## Context (why)

M11 localized the B-DASH end-to-end ceiling (20%, 4/20) to the VLA's **in-gripper grasp error** (peg
held cocked ~8°/~9 mm), which the §2.1 peg-pose-blind insertion controller architecturally cannot
correct (cheat ablation: feeding the true tip lifts 8 mm-offset insertion 0→55%). As written that is a
single-point *negative* result. This plan turns it into (1) a **predictive theory** and (2) a
**§2.1-honest fix** that recovers the lost success with **no VLA retraining**.

## 1. Headline contribution — the precision-budget theory

A modular interface defined by the controller's observation map `O` partitions all front-end error into:
- **Correctable channel** — error that *enters* `O` and is servoed out; tolerance = the measured
  convergence basin `B(r,θ)` (E4). For B-DASH: handoff-delivery error `d=(r,θ_h)` (assumed-tip vs known
  socket) is correctable.
- **Blind channel** — error *hidden* by `O`, surviving as a fixed bias; tolerance = the measured blind
  response `R(g)`. For B-DASH: in-gripper grasp `g=(l,φ)` is blind (`tip_estimate` assumes `g=0`,
  `ee_control.py:29`), becoming a tip bias `e(g)=l+grip_offset·sin φ` (≈9 mm + 60·sin8° ≈ 17 mm).

**Predictive model (non-circular):** `P(success) ≈ E_{d∼p_handoff}[B(d)] · E_{g∼p_grasp}[R(g)]`, where
`B` (from E4, VLA-free), `R` (from the scripted test-bed, VLA-free), and `p_handoff`/`p_grasp` (measured
**at handoff, before the insert resolves**) are all independent of the insert outcome. It reproduces every
anchor: scripted 98% (g≈0), VLA 20% (handoff inside basin → blind channel binds), oracle 55% (cheat moves
g blind→correctable), and the clearance cliff (`R(g;c)→0` once `e+d>c`).

**The fix is then "moving the blind→correctable boundary":** an estimator `ĝ=g+ε` fed to a corrected
`tip_estimate'` interpolates `R` (ε→∞) to `R_oracle` (ε→0). Predicts the lift from estimator noise.

## 2. The fix — §2.1-honest interface-widening (an instantiation, NOT the novelty)

Estimate the in-gripper grasp pose **without reading peg ground-truth**, feed it to correct `tip_estimate`
(single injection at `insertion_controller.py:123`: `tip = tip_estimate(...) + self.tip_correction`).

- **Primary: active tactile tip-localization (bore-edge bisection).** Inputs allowed by §2.1 only:
  EE proprioception (`ee_frame`), fixed socket pose, wrist-F/T (`_wrench`). The bore is a disk radius
  `bore_r=0.004+c` at known socket xy; the set of *assumed*-tip xy that drops the true tip into the bore
  is a disk centered at `socket_xy − e_xy`. Probe-descend at several xy, detect drop-in (low EE-z at
  contact) vs rim/top (high EE-z), bisect the boundary in ≥4 directions → bore center in assumed coords
  → `e_xy`. A flat-top touch gives `e_z`. The 3 mm countersink is a *refinement* (graded ramp), not the
  workhorse — the error (9–17 mm) far exceeds the funnel band, so **bisection, not cone-sinusoid, is
  primary** (critique #1). Robust at large magnitude; degrades as `bore_r→peg_r` (tight clearance) →
  scope the fix to `c≥2.0`.
- **Tilt is force-only-unobservable** (peg_contact gives `net_forces_w`, no torque) → position-only
  correction reproduces the ~55% oracle ceiling. **Stretch:** the per-finger `force_matrix_w`
  (`peg_finger_contact`) moment about the grip center is a §2.1-legal cock cue — try it to partially
  observe tilt and push past 55%.
- **Secondary: onboard wrist-cam estimate** (reuse `bdash_classic_vision` segment+back-project) — but the
  wrist cam is RGB-only and looks down the peg axis (near-degenerate, may clip), and `segment_blue` breaks
  at L2/L3. Keep as an **E3 robustness contrast** (vision good for lateral/one-shot, bad on tilt; tactile
  robust to lighting), **not the headline**.

**Honest ceiling:** ~55% position-only (≈2.7× over 20%), not 70-90%. State this as the theory-predicted result.

## 3. Novelty & positioning (lit grounding)

The *fix alone is not novel* — in-hand-pose→insert is established: **TacGraph (2512.23856)** tactile in-hand
pose → peg insert (3 mm, 57.5%); **ViTaSCOPE (2506.12239)** visuo-tactile in-hand pose; **Active Extrinsic
Contact (2110.03555)** active tactile probe → insertion; **TacRefineNet (2509.25746)** corrects in-hand
grasp pose for VLAs w/o retraining (but *re-grasps* — needs a dexterous hand). **What is novel:** the
measurable **precision-budget theory** + causal attribution to the blind in-gripper channel + the
no-retrain compensation **under a binary gripper that forbids re-grasp**. Lead with the theory; position
the fix as an instantiation that *builds on* (cites) this literature.

- **GTP-FA (2606.03385)** concurrently finds grasp quality bounds VLA assembly (real robot) → **cite as
  corroboration**, differentiate (we give a predictive budget + clearance tie + no-retrain fix, not a
  learned failure classifier).
- **FORGE (2408.04587)** / Factory / IndustReal: the "why not a learned force-search policy?" threat —
  show classical peg-blind controllers specifically cannot recover ~9 mm in-gripper error and that the
  budget *explains* the 5 mm basin rather than just achieving it; our fix needs no RL.
- **Compute-efficiency angle** (single-GPU, "where to spend the budget"): a **sub-theme in Intro/Discussion**,
  not the headline (DIRECT 2606.12402 owns "where to spend compute"; don't reuse the phrase). GraspVLA as a
  *foil* (scaling grasp data ≠ insertion precision), not a beaten baseline.

## 4. Experiment / run matrix

- **RUN 0 (code):** re-add `BDASH_GRASP_OFF_MM` (+`_DEG`) test-bed in `scripted_pick.py`; unify
  `tip_correction` + `BDASH_CHEAT_TIP` oracle in `insertion_controller.py`. *(Phase 1, this commit.)*
- **RUN 1 — blind response `R(g)`:** scripted, `BDASH_GRASP_OFF_MM/_DEG` sweep l∈{0,2,4,6,8,10,12} ×
  φ∈{0,2,4,6,8,10}, c∈{1,2,3}, n≥20, batched, no cameras → `blind_response_c*.csv`.
- **RUN 2 — oracle `R_oracle(g)`:** same sweep + `BDASH_CHEAT_TIP` → `oracle_response_c2.0.csv`.
- **RUN 3 — VLA `p_grasp(g)`:** `BDASH_GRASP_DIAG` over `bdash_vla_v6_recovery`, n≥40, ≥3 seeds → parse to
  `grasp_diag_vla_v6.csv` (l,φ,result per episode).
- **RUN 4 — `p_handoff(d)`:** aggregate `handoff_dx/dy/tilt` from the v6 §3.5 JSONL.
- **RUN 5 — extend basin** `B` to the VLA tail (r→8 mm, φ→12°), recover mode.
- **RUN 6 — tighten anchors:** scripted c2.0 l=8 mm: blind 0/20, +cheat ~11/20.
- **RUN 7 — separability (full d×g grid):** test `success(d,g) ≈ B(d)/B(0)·R(g)`; report residual; re-measure
  `B` under non-zero held cock. *(load-bearing for the product claim.)*
- **RUN 8 — multi-checkpoint transfer (strongest held-out):** same `R/B` predict v3/v4/v5/v6 from each
  checkpoint's `p_grasp`.
- **RUN 9 — multi-seed VLA** (≥5) end-to-end.
- **RUN 10 — interface-widening prototype:** implement the bore-edge probe; measure `σ_ε` on RUN 1's
  known-g test-bed; predict `E_g[R_widened(σ_ε)]`; then run widened VLA, compare. *(the fix.)*
- **RUN 11 — generalization:** generate the c=3.0 socket (forward-predict >90%); 2nd peg geometry (different
  grip_offset → different `R`).

## 5. Figures
interface_partition (concept) · correctable_basin (B + VLA handoff overlay) · blind_response (R heatmap) ·
convolution_predicts_20 (R⊛p_grasp → predicted vs observed 4/20) · boundary_moved (R vs R_oracle vs
R_widened) · clearance_prediction (model vs E2 cliff + c=3.0 forward) · calibration (predicted-vs-observed
y=x money plot) · success_tail_separation (per-episode g colored by outcome).

## 6. Falsifiable predictions
Convolution predicts VLA success in [0.08,0.42] (weak; tighten via held-out). Boundary motion: blind 0/20,
oracle ~55%, no non-tip tweak recovers. Cliff is a *prediction* (c≤1→0, c=3→>90%). Multi-checkpoint: one
R/B predicts v3/v4/v5/v6. Tail separation: the 4 successes have low `e(g)`. Interface-widening: σ_ε≈2 mm →
20%→~50%. Separability: joint grid matches the product within CI.

## 7. Risks & mitigations (from adversarial review)
- **Probe may not recover lateral at 9–17 mm** → lead with bore-edge bisection (not cone), run at REAL
  magnitudes first, report `e_xy` RMSE vs `BDASH_GRASP_DIAG`; scope to c≥2.0.
- **Sim-only / single task** → lead with the THEORY as a controlled sim study (sim enables the oracle);
  add no-hardware generalization (c=3.0, 2nd peg, multi-checkpoint).
- **Single-seed, wide CI** → multi-seed; shift falsifiable weight to held-out tests.
- **Separability physically suspect** → full d×g grid + report residual; fall back to measured 2D surface
  if it fails.
- **§2.1 leak-by-statistics** (magnitude priors, GT-tuned thresholds) → data-adaptive search, freeze
  hyperparameters before eval, ship a grep-audit; `BDASH_GRASP_DIAG` is scoring-only.
- **Contact SNR** (cocked grasp → asymmetric finger baseline) → per-probe baseline subtraction + persistence
  + low-speed gate; report touch SNR / false-trigger rate.
- **sim idealization:** peg-body net contact ≠ real wrist F/T → stated sim-to-real caveat.

## Results (2026-06-27, scripted test-bed unless noted)

All §2.1-gated probes default OFF (headline path unchanged). Figures: `figs/bdash/budget_*.png`
(`plot_budget_figs.py`); data `results/bdash/repro/{sweeps,grasp_diag_vla_v6,probe_*,dc_*}.csv`.

1. **Theory predicts the VLA (non-circular).** From independent measurements (R from the test-bed,
   p_grasp from BDASH_GRASP_DIAG — neither sees the insert), P_pred = 5.4% (effective err e=lat+60·sinθ)
   to 12.9% (lateral only); OBSERVED VLA 10% (2/20), CI95 [2.8%, 30.1%]. Both bracket the observed.
2. **Blind response R(l):** 100% to ~2.4mm, 35% at 4.4mm, 0% beyond 6mm (sharp cliff).
   **R_widened(σ):** 55/45/20% at σ=0/2/3mm → the estimator must reach σ≲2mm lateral.
3. **Lateral channel opened (bore-edge tactile probe):** off≤8mm blind 0/20 → probe **20/20 = 100%**,
   exceeding the per-step true-tip oracle (55–68%). Insight: a FROZEN once-estimate beats the jittery
   oracle (the controller chases per-step contact jitter) — so M11's "55% physical ceiling" was a
   jittery-cheat artifact. Gap: localization fails for lat ≥10mm (bore at the ±12mm grid edge).
4. **Tilt channel observable §2.1 (NEW; contradicts "force-only can't see tilt"):** per-finger contact
   force z-asymmetry dz = f0_z−f1_z ∝ in-gripper cock (net wrench cancels, per-finger moment does not).
   Calib gain ≈0.75 N/° (clean ≥6°). EE de-cant (SIGN=+1) straightens handoff_tilt 10°→2° (rigid grasp:
   EE rotation = peg rotation; no re-grasp). Observable axis ⊥ the finger line; the other axis needs a
   90° wrist-yaw re-measure (future).
5. **Combined fix (de-cant + lateral probe) — the headline test-bed result:** off_deg=6 (tilt 5.7°/lat
   5mm) blind 6/20 → **full 19/20 = 95%**, beating oracle 17/23 = 74% and de-cant-only 12/22. Both blind
   channels opened §2.1-legally, exceeding the true-tip oracle. At off_deg=10/15 (lat 9.5/15mm) full
   collapses (3/21, 1/21): lateral exceeds the ±12mm grid (a wider 7×7 grid over-pressed/under-localized
   → 0/6, needs a 2-stage probe — future).
6. **VLA application:** lateral-only probe = 3/20 (unchanged — VLA grasps are tilt 5–22°/lat 6–25mm,
   beyond a lateral-only fix). Combined fix (de-cant + ±12 probe) on the VLA = [running]. Honest
   expectation: lifts the moderate grasps (tilt≲8°, lat≲8mm after de-cant, ~6/14 handoffs) toward ~95%;
   the extreme grasps stay beyond reach → residual = front-end grasp quality, exactly as the budget
   predicts.

**Paper takeaway:** the precision-budget theory predicts end-to-end success; BOTH blind channels
(lateral, tilt) are openable §2.1-legally and the combined fix exceeds the true-tip oracle at moderate
error; the residual for this VLA is its extreme grasp quality (front-end), which the budget quantifies.
