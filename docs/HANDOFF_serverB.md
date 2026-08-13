# B-DASH — Server B continuation handoff

You (Server B: GPU + Isaac Sim + GR00T + fine-tune capable) are taking over the B-DASH compute work
while another machine writes the paper. This briefs you on the current state and the next lever.

## 0. Update the repo first
Repo: `momoiorg-repository/IsaacLab-Arena-Workshop`, branch `an/bdash-vla-training`.
```
git fetch origin && git checkout an/bdash-vla-training && git pull --ff-only origin an/bdash-vla-training
git lfs pull   # figs/bdash/*.png are LFS
```
Your local copy was old — after this you have all the B-DASH code + docs (commits be154b29, 66775820,
f2c15d09, 896ffe85, 9acdcf4f).

## 1. Where the project is
- **M10** found+fixed the frozen-camera bug (a bare `sim.render()` doesn't refresh RTX sensors on the
  recording box → the VLA trained on a single static frame). Fix = render-coupled `sim.step(render=True)`
  in `record_vla_demos.py` + `policy_runner.py`. First non-zero result followed.
- **v6 model** (`umegan/vdash-gr00t-n1-7-pick-handoff-v6-recovery`, N1.7, frozen ViT) = **20% insert**
  (4/20) at c2.0/L1. Funnel: 85% grasp → 75% clean handoff → 20% insert.
- **M11** localized the residual to **in-gripper grasp quality**: the VLA holds the peg cocked ~8°/~9mm;
  the §2.1 peg-pose-blind insertion controller cannot correct it (cheat ablation 0→55%).
- **This session** built a **precision-budget theory + §2.1 interface-widening probes** (all env-gated,
  default OFF → headline path unchanged):
  - Theory: `P(success) ≈ E_d[B(d)]·E_g[R(g)]` (correctable basin × blind grasp channel). The
    convolution, from independent measurements, predicts the VLA (5–13% vs observed 10%).
  - **Lateral channel openable §2.1**: bore-edge tactile probe → 8mm offset 0/20 → **20/20**, beating
    the true-tip oracle (frozen estimate > jittery per-step true tip).
  - **Tilt channel observable §2.1** (new): per-finger contact-force z-asymmetry `dz` ∝ in-gripper cock
    → EE de-cant straightens it (10°→2°). Combined fix = **95%** at moderate error (tilt5.7°/lat5mm),
    beating oracle 74%.
  - **VLA application: the fix did NOT transfer** (1–3/20). v6 grasps are tilt 5–22° / lat 6–25mm —
    beyond the range-limited, single-axis §2.1 estimators. Residual = **front-end grasp quality**, as
    the budget predicts.

## 2. THE decisive next lever (this is your job — you can fine-tune)
**Retrain the VLA front-end for grasp QUALITY** (not rate — rate is already 85%). Target the
budget-quantified spec: in-gripper **lateral ≲ 2–3 mm AND tilt ≲ 4°** (from `R_widened(σ)` + the
de-cant ceiling; see `figs/bdash/budget_*.png`). v6 was trained on the camera-fixed v5-recovery set
whose scripted grasps are dead-straight (ingrip ~0.1°), yet it reproduces ~8° at eval — a closed-loop
grasp-precision imitation gap. Candidate levers:
- more grasp-precision / recovery (DAgger-style) data emphasizing straight, centered grasps;
- a grasp-quality filter/reward, or relabel from the actual VLA's visited states;
- re-record with the perturbation hook (`record_vla_demos.py --perturb_*`) + the camera fix; retrain
  (frozen ViT, `train_vla.sh` / `train_and_push_v6_recovery.sh`); eval each ckpt at c2.0/L1.
The budget gives the acceptance bar: drive measured in-gripper tilt/lat (BDASH_GRASP_DIAG) under the
spec and the existing classical controller should convert most handoffs.

## 3. Key files
- Docs: `docs/bdash_precision_budget_plan.md` (theory + run matrix + Results), `docs/milestones/bdash_status.md`
  (single-page overview), `M10.md`, `M11.md`, `share/bdash_paper_claims_audit.md`, `docs/run_policy.md`
  (run cheatsheet), `docs/B-DASH_dev_brief_v2.md` (design + §2.1 contract).
- Code (env-gated): `isaaclab_arena/controllers/insertion_controller.py` (CALIBRATE bore-edge probe,
  `_apply_decant`, `tip_correction`, `BDASH_CHEAT_TIP`, `BDASH_TIP_NOISE_MM`),
  `controllers/scripted_pick.py` (`BDASH_GRASP_OFF_MM`/`_DEG` test-bed),
  `policy/bdash_scripted_policy.py` (`BDASH_GRASP_DIAG`/`BDASH_FINGER_DIAG`).
- Scripts: `scripts/bdash/{plot_budget_figs.py, budget_run_*.sh, run_headline_repro.sh, run_eval_grid.py,
  run_convergence_zone.py, record_vla_demos.py, train_vla.sh, train_and_push_v6_recovery.sh}`.

## 4. HF assets (pull on B)
- Model: `umegan/vdash-gr00t-n1-7-pick-handoff-v6-recovery` (~12GB, current best).
- Dataset: `umegan/vdash-pick-handoff-v5-recovery` (1000 demos, camera-fixed / moving frames).

## 5. Gotchas / caveats
- **§2.1**: the insertion controller may use ONLY ee_frame proprioception, the fixed socket pose, wrist
  F/T, and per-finger contact forces — NEVER the peg's true pose. `BDASH_CHEAT_TIP` violates this
  (diagnostic/oracle only).
- **Frozen-camera bug** is environment-specific (RTX 5080 / Isaac Sim 5.1) — verify camera frames move
  on B's GPU before trusting any new dataset/eval (per-step framediff > 0).
- All headline numbers are **single-seed (existence-level)**; multi-seed is pending.
- Training: 3B fp32 fine-tune OOMs a 16GB card → use B's A100/H100-class GPU; gn17 container = N1.7
  (transformers 4.57.3).
- Reproduce the frozen headline: `scripts/bdash/run_headline_repro.sh` (~1h, gn17).

## 6. Full chronological detail
The complete v3→v6 + diagnosis + this-session log lives in the paper machine's Claude working memory
(`bdash-vla-training-pipeline.md`). If you need more than this brief, ask the user to paste it (or it may
be appended below this file).
