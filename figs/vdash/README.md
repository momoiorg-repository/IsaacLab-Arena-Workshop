# V-DASH insertion-diagnosis figures (M11)

Regenerate all: `/isaac-sim/python.sh scripts/vdash/plot_paper_figs.py` (gn17 container).
Every number traces to a CSV/JSONL on disk (see the paper-claims audit `share/vdash_paper_claims_audit.md`).
**All single-seed unless noted; use L1 as the scripted headline (L0 is unstable run-to-run).**

| figure | claim it supports | source data |
|---|---|---|
| `ingripper.png` | The VLA holds the peg cocked **in the gripper** (8°/9 mm) vs scripted 0.2°/0.8 mm — the root cause of the 20% insert. | `logs/vdash/graspdiag_{scripted,vla}.out` (`VDASH_GRASP_DIAG`) |
| `cheat.png` | The failure **is** the tip-estimate error: nominal 98% → 8 mm in-gripper offset 0% → give the controller the true tip 55%. Even ideal tip knowledge caps ~55% (peg still held off-axis). | `logs/vdash/cock8_{test,cheat}.out` |
| `e4_basin.png` | The controller is **robust to handoff pose** — recovery basin ~84% (offset to 5 mm, tilt to ~8°) vs held ~39%. So neither handoff offset nor tilt explains 20%. (L0 scene is fixed → the basin is **deterministic**; each cell = mean over tilt azimuths.) | `results/vdash/e4_{held,recover}_c2.0_s*.csv` |
| `e2_clearance.png` | Classical insertion success vs radial clearance (scripted, n=50, Wilson 95%), L0 & L1. | `results/vdash/e2_clearance_final.csv` |

## Headline numbers (c=2.0, L1, seed 0)

- Scripted insert (clean handoff): **49/50 = 98%**, CI95 [0.895, 0.997].
- VLA v6-recovery: **4/20 = 20%**, CI95 [0.081, 0.416]; funnel 17 grasp → 15 handoff → 4 insert.
- Convergence basin: held **37%** vs recovery **83%** (CI95 [0.760, 0.894]).
- In-gripper at handoff: scripted **0.2° / 0.8 mm** vs VLA **~8° / ~9 mm**.
- VLA handoff (logger, n=15): tilt mean 5.6° / median 4.9° / max 15.7°; \|dxy\| median 6.5 mm.
- **E2 clearance (scripted, n=50): c2.0 = 100% (L0) / 98% (L1); c≤1.0 = 0% at every level.** A sharp
  cliff, not a gentle curve. **Precision-limited, not a controller gap:** c=1.0 with `use_hop_search`
  ON is still 0/50 (f_ins 31 N) — the ~4 mm scripted handoff lateral exceeds what fits a ≤1 mm hole, so
  the classical baseline has a **~2 mm clearance floor** with the current handoff precision.

## The one-line result

Frozen-frames fix (M10) → the VLA perceives & grasps (85%) and hands off (75%); classical control
inserts a clean handoff at 98% with a wide basin; but the VLA's **in-gripper grasp** is cocked ~8°/9 mm,
and a §2.1 peg-pose-blind controller **cannot** correct it → 20% end-to-end. The residual is grasp
quality (front-end), architecturally unfixable downstream.
