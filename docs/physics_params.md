# Physics Parameters — Factory / FORGE PegInsert (M0)

Extracted from Isaac Lab **2.3.2** (`submodules/IsaacLab`), the source-of-truth for B-DASH
contact-rich insertion physics. Per the brief §4.2, **FORGE** is the chosen base (adds
force/torque sensing + noise). FORGE inherits all simulation/asset physics from Factory and
only overrides the controller, observations (adds `ft_force`, `force_threshold`), and
domain-randomization events.

Sources:
- `submodules/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py`
- `.../direct/factory/factory_tasks_cfg.py`
- `.../direct/factory/factory_control.py`
- `.../direct/forge/forge_env_cfg.py`
- `.../direct/forge/forge_env.py`

---

## 1. Simulation (`SimulationCfg` / `PhysxCfg`) — Factory base, used by FORGE

| Param | Value | Notes |
|---|---|---|
| `dt` | **1/120 s** (120 Hz) | physics step |
| `gravity` | (0, 0, −9.81) | |
| `decimation` | **8** (Factory) | env control step = 8 physics steps → 15 Hz policy. FORGE same base. |
| `solver_type` | **1** (TGS) | |
| `max_position_iteration_count` | **192** | "Important to avoid interpenetration" |
| `max_velocity_iteration_count` | 1 | |
| `bounce_threshold_velocity` | 0.2 | |
| `friction_offset_threshold` | 0.01 | |
| `friction_correlation_distance` | 0.00625 | |
| `gpu_max_rigid_contact_count` | 2**23 | |
| `gpu_max_rigid_patch_count` | 2**23 | |
| `gpu_collision_stack_size` | 2**28 | |
| `gpu_max_num_partitions` | **1** | "Important for stable simulation" |
| global `physics_material` | static=1.0, dynamic=1.0 | `RigidBodyMaterialCfg` |

## 2. Per-asset rigid/collision props (peg, hole, robot)

| Param | Value | Applies to |
|---|---|---|
| `solver_position_iteration_count` | **192** | robot, fixed (hole), held (peg) |
| `solver_velocity_iteration_count` | 1 | all |
| `max_depenetration_velocity` | 5.0 | all |
| `max_contact_impulse` | 1e32 | all |
| `collision_props.contact_offset` | **0.0002** (B-DASH; Factory ref 0.005) | all |
| `collision_props.rest_offset` | **0.0** | all |
| `disable_gravity` | **held(peg)=True**, fixed(hole)=False, robot=True | held floats in gripper |
| asset surface friction | **0.75** | peg, hole, robot fingerpad (`*AssetCfg.friction`) |
| robot `franka_fingerpad_length` | 0.017608 m | |

> **Collision representation**: the hole interior requires **SDF mesh collision** (convex hull
> cannot represent a bore). Factory/FORGE ship pre-authored USDs (`factory_hole_8mm.usd`) with
> SDF; our generated sockets (M1) must set `MeshCollisionPropertiesCfg` SDF via
> `isaaclab.sim.converters.MeshConverter` + `sim/schemas/schemas_cfg.py`.

## 3. Task-space controller (`factory_control.py` + `CtrlCfg`)

Operational-space impedance controller (the reusable core for `insertion_controller.py`).

| Param | Factory | FORGE |
|---|---|---|
| `default_task_prop_gains` (kp, xyz/rpy) | [100,100,100, 30,30,30] | **[565,565,565, 28,28,28]** |
| `reset_task_prop_gains` | [300,300,300, 20,20,20] | — |
| `ema_factor` | 0.2 | `ema_factor_range` [0.025, 0.1] |
| `pos_action_bounds` | [0.05,0.05,0.05] | |
| `rot_action_bounds` | [1.0,1.0,1.0] | |
| `pos_action_threshold` | [0.02,0.02,0.02] | |
| `rot_action_threshold` | [0.097,0.097,0.097] | |
| null-space `kp_null` / `kd_null` | 10.0 / 6.3246 | |
| `default_dead_zone` (F/T) | — | **[5,5,5, 1,1,1]** N·/N·m |
| `action_space` | 6 (Δpose) | **7** (Δpose + force-threshold action) |

## 4. FORGE force/torque sensing  ← key for `force_violation` + insertion force control

- Force read from a dedicated **`force_sensor` articulation link** via
  `self._robot.root_physx_view.get_link_incoming_joint_force()[:, force_sensor_body_idx]`
  (`forge_env.py:36,95`). This is a **joint reaction-force sensor**, 6-DoF (F + τ).
- `ft_smoothing_factor` = **0.25** (EMA): `F_smooth = α·F + (1−α)·F_smooth`.
- Frame conversion world→fingertip via `forge_utils.change_FT_frame`.
- Observation additions: `ft_force` (3), `force_threshold` (1).
- Noise (`ForgeObsRandCfg`): `ft_force`=**1.0 N**, `fingertip_pos`=0.00025 m, `fingertip_rot_deg`=0.1°.

> **⚠ R1 — `force_sensor` link is NOT present on the Arena Franka** (`embodiments/franka/franka.py`
> uses the standard Franka, not `franka_mimic.usd`). **Decision for M2:** acquire contact force via
> **`ContactSensor`** on the peg body and the two finger bodies (`activate_contact_sensors=True`
> is already set in `FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG`, `mdp/robot_configs.py:20`). The FORGE
> EMA smoothing (α=0.25) and Gaussian F/T noise (σ≈1 N) options are carried over into
> `insertion_controller.py` / the `force_violation` predicate. Adding a `force_sensor` frame to
> the Franka USD is a fallback if ContactSensor proves too noisy.

## 5. PegInsert reference geometry & clearance (Factory `factory_tasks_cfg.py`)

| Asset | Value |
|---|---|
| Peg8mm diameter | **7.986 mm** (0.007986 m), height 50 mm, mass 0.019 kg |
| Hole8mm diameter | **8.1 mm** (0.0081 m), height 25 mm, mass 0.05 kg |
| → **native radial clearance** | **(8.1 − 7.986)/2 ≈ 0.057 mm** |
| `success_threshold` | 0.04 × socket height (frac.) |
| `engage_threshold` | 0.9 |
| hand_init_pos (rel. fixed tip) | [0, 0, 0.047] m, noise [0.02,0.02,0.01] |

> **Clearance implication for the brief's series** (穴径 = 8.0 + 2c, c∈{2.0,1.0,0.5,0.25} mm):
> our **tightest** case c=0.25 mm radial is **~4.4× looser** than Factory's native 0.057 mm,
> which Factory/FORGE simulate stably at this physics config. ⇒ **R2 (c=0.25 instability) is
> unlikely to bind**; physical stability at c=0.25 is expected. The research risk is instead that
> c=0.25 may still be *too easy* — to be assessed in M1/M5, not a physics-stability issue.

## 6. Mapping to our env callback (`mdp/env_callbacks.py`)

The existing `assembly_env_cfg_callback` currently uses **dt=1/60, decimation=2** and lower
iteration counts — **not FORGE-grade**. M1 introduces `bdash_assembly_env_cfg_callback` applying
the §1–§2 values (dt=1/120, pos_iter=192, contact_offset=0.0002, rest_offset=0.0,
friction=1.0/asset-0.75, gpu_max_num_partitions=1), and wires ContactSensors per §4.

> **contact_offset note:** the runtime uses **0.0002 m** (`configs/bdash/peg_insert/assets.yaml:26`), not Factory's
> 0.005 — the 5 mm Factory offset exceeds the tightest clearance (0.25 mm) and blocks bore entry. The
> deviation is deliberate; this doc lists the B-DASH value with the Factory reference for provenance.

---

## 7. M3b — peg `max_depenetration_velocity` change (§2.3 record)

| Param | Before | After | Where |
|---|---|---|---|
| peg `max_depenetration_velocity` | 5.0 m/s (shared `RIGID_BODY_PROPS_HIGH_PRECISION`) | **1.0 m/s** | `isaaclab_arena_environments/mdp/bdash_peg_assets.py` (`_PEG_RIGID_PROPS`, bdash-local deepcopy — shared constant untouched per §2.2) |

**Why:** brief §3 M3b, to soften the M1 known-risk SDF force spike at axis-misaligned initial
conditions. **Stability re-check (§2.3):** scripted pick→insert, L1, seed-fixed.

| Condition | Result | force_max_N (grasp clamp) median / max | insert force median / max | NaN / blow-up |
|---|---|---|---|---|
| c=2.0, capped 1.0 (50 ep) | **50/50** (no M3a regression) | 34.8 / 41.8 | 0.21 / — | 0 |
| c=0.25, capped 1.0 (24 ep) | 0/24 (`insertion_failed`, clearance too tight) | 35.9 / 43.5 | 12.4 / 43.5 | 0 |
| c=0.25, uncapped 5.0 (24 ep, same seed) | 0/24 | 35.7 / 40.5 | 13.4 / 39.7 | 0 |

**Finding:** capped vs uncapped are statistically identical and **neither explodes** — the M1 SDF
spike does **not** trigger in the funnel-guided scripted flow (the low-force servo never drives a
deep one-step penetration). The cap is kept as cheap **insurance for M7's direct-placement harness**
(peg held at controlled offsets/tilts = the real axis-misaligned IC); re-verify there. No regression
to the established c=2.0 success path.

---

### Runtime confirmation
See `docs/milestones/M0.md` for the actual `Isaac-Forge-PegInsert-Direct-v0` launch log on the
current machine.
