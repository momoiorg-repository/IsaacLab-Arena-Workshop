# Demo Recording & Data Generation Pipeline

Full workflow: teleoperate → record → annotate → replay/verify → mimic augment → convert for GR00T.

All commands run **inside the Docker container** using `/isaac-sim/python.sh`.

---

## Overview

```
record_demos.py     →  raw HDF5 (succeeded episodes only)
annotate_demos.py   →  HDF5 + mimic subtask annotations
generate_dataset.py →  large augmented HDF5 via isaaclab_mimic
convert_hdf5_to_lerobot.py  →  LeRobot format for GR00T training
```

---

## Step 1: Record Demonstrations

`isaaclab_arena/scripts/imitation_learning/record_demos.py`

Teleoperate the robot and save only successful episodes to HDF5.

```bash
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --dataset_file ./datasets/crx5ia_pick/source.hdf5 \
  --num_demos 10 \
  --step_hz 30 \
  table_pick_and_place \
  --embodiment crx5ia_robotiq85 \
  --object dex_cube \
  --teleop_device spacemouse"
```

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_file` | *(required)* | Output `.hdf5` path |
| `--num_demos` | 1 | Episodes to record (0 = infinite) |
| `--step_hz` | 30 | Simulation rate |
| `--num_success_steps` | 10 | Consecutive success steps to confirm episode done |
| `--teleop_device` | keyboard | `keyboard`, `spacemouse`, `avp_handtracking` |
| `--enable_cameras` | off | Record wrist/left/right camera RGB |

**Keyboard controls during recording:**

| Key | Action |
|-----|--------|
| `R` | Reset episode (discard current attempt) |
| Ctrl+C | End session |

Only episodes where the success condition holds for `--num_success_steps` consecutive steps are exported. The HDF5 stores actions, joint states, EEF pose, and (optionally) camera images per episode.

---

## Step 2: Annotate Demonstrations (Mimic Subtasks)

`isaaclab_arena/scripts/imitation_learning/annotate_demos.py`

Replays recorded episodes and adds IsaacLab Mimic subtask boundary annotations. Required before `generate_dataset.py`.

The environment must be a `ManagerBasedRLMimicEnv` subclass (e.g., `CRX5iAMimicEnv`).

### Auto mode (recommended)

Uses the environment's `get_subtask_term_signals()` to detect subtask completion automatically:

```bash
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
  --input_file ./datasets/crx5ia_pick/source.hdf5 \
  --output_file ./datasets/crx5ia_pick/source_annotated.hdf5 \
  --auto \
  table_pick_and_place \
  --embodiment crx5ia_robotiq85 \
  --object dex_cube"
```

### Manual mode

Replays each episode visually; press `S` at the moment each subtask completes:

```bash
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
  --input_file ./datasets/crx5ia_pick/source.hdf5 \
  --output_file ./datasets/crx5ia_pick/source_annotated.hdf5 \
  table_pick_and_place \
  --embodiment crx5ia_robotiq85 \
  --object dex_cube"
```

**Manual keyboard controls:**

| Key | Action |
|-----|--------|
| `N` | Play / resume |
| `B` | Pause |
| `S` | Mark subtask boundary at current frame |
| `Q` | Skip episode |

**What gets written:** The output HDF5 gains `obs/datagen_info/subtask_term_signals/<name>` boolean tensors per episode, plus EEF pose, object pose, and target EEF pose at every step (`obs/datagen_info`).

---

## Step 3: Verify with Replay

`isaaclab_arena/scripts/imitation_learning/replay_demos.py`

Plays back recorded episodes without recording — useful to visually confirm quality before augmentation.

```bash
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/replay_demos.py \
  --dataset_file ./datasets/crx5ia_pick/source_annotated.hdf5 \
  table_pick_and_place \
  --embodiment crx5ia_robotiq85 \
  --object dex_cube"
```

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_file` | *(required)* | HDF5 to replay |
| `--select_episodes` | all | Space-separated episode indices |
| `--validate_states` | off | Compare recorded vs replayed states (single env only) |
| `--num_envs` | 1 | Replay across multiple envs in parallel |

**Controls:** `N` = play, `B` = pause.

State validation (`--validate_states --num_envs 1`) prints per-joint mismatches if the replay drifts from the recorded state. Useful to catch USD/actuator config divergence.

---

## Step 4: Generate Augmented Dataset (Mimic)

`isaaclab_arena/scripts/imitation_learning/generate_dataset.py`

Uses IsaacLab Mimic to synthesize many new episodes from a small set of annotated source demos by randomising object poses and replanning EEF trajectories.

```bash
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
  --input_file ./datasets/crx5ia_pick/source_annotated.hdf5 \
  --output_file ./datasets/crx5ia_pick/augmented.hdf5 \
  --generation_num_trials 500 \
  --num_envs 8 \
  table_pick_and_place \
  --embodiment crx5ia_robotiq85 \
  --object dex_cube"
```

**Key arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input_file` | *(required)* | Annotated source HDF5 |
| `--output_file` | `./datasets/output_dataset.hdf5` | Output path |
| `--generation_num_trials` | *(required)* | Target number of generated episodes |
| `--num_envs` | 1 | Parallel environments (higher = faster) |
| `--enable_cameras` | off | Include camera obs in generated data |
| `--pause_subtask` | off | Pause after each subtask (debug/render only) |

**Generation config defaults** (from `isaaclab_arena/tasks/common/mimic_default_params.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `generation_guarantee` | `True` | Retry failures until trial count is met |
| `generation_keep_failed` | `False` | Discard failed attempts |
| `max_num_failures` | 25 | Abort after N consecutive failures |
| `generation_relative` | `False` | Absolute EEF pose interpolation |
| `seed` | 1 | RNG seed |

These can be overridden by setting them in the environment's `datagen_config` field.

The environment must implement the `ManagerBasedRLMimicEnv` API:
- `get_robot_eef_pose(eef_name)` — current 4×4 pose matrix
- `target_eef_pose_to_action(...)` — convert target pose → delta action
- `action_to_target_eef_pose(action)` — inverse (for datagen info recording)
- `actions_to_gripper_actions(actions)` — extract gripper scalar from action
- `get_object_poses()` — scene object poses for datagen info
- `get_subtask_term_signals()` — auto-annotation signals

---

## Step 5: Convert to LeRobot Format (GR00T Training)

`isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py`

Converts the augmented HDF5 to LeRobot format for GR00T N1.5/N1.6 training.

```bash
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --config_path isaaclab_arena_gr00t/policy/config/crx5ia_robotiq85_manip_gr00t_closedloop_config.yaml \
  --dataset_path ./datasets/crx5ia_pick/augmented.hdf5 \
  --output_path ./datasets/crx5ia_pick/lerobot"
```

The converter uses `crx5ia_robotiq85_manip_gr00t_closedloop_config.yaml` to:
- Map simulation joint names to GR00T policy joint names (7-DOF: J1-J6 + left_knuckle)
- Resize and encode camera frames (wrist_cam, left_cam, right_cam)
- Write LeRobot-compatible `data/` + `videos/` directory structure

---

## CRX5iA + Robotiq 85 Specifics

### Action space (7-DOF)

| Index | Joint | Notes |
|-------|-------|-------|
| 0–5 | J1–J6 | Arm joints |
| 6 | `robotiq_85_left_knuckle_joint` | Gripper (0 = open, 0.8 = closed) |

`robotiq_85_right_knuckle_joint` is driven automatically by `CRX5iARobotiq85JointMirrorAction` with `−1×` the left knuckle value. It does **not** appear in recorded actions or GR00T output.

### Teleoperation actions

- **IK mode** (`CRX5iAIKActionCfg`): delta EEF pose (6-DOF) + gripper scalar = 7-DOF total. Used for `record_demos.py`.
- **Joint position mode** (`CRX5iARobotiq85JointPositionActionsCfg`): direct J1-J6 + left_knuckle. Used for `replay_demos.py` and `generate_dataset.py`.

### Mimic subtasks (pick-and-place)

`PickAndPlaceTask` defines two subtasks per EEF:
1. Reach and grasp the object
2. Place at destination

The subtask boundary for step 1 fires when the gripper closes on the object; step 2 fires at task success. Auto-annotation mode detects these automatically.

---

## Troubleshooting

**"No success termination term was found"** — The environment must have `env_cfg.terminations.success` defined. Check that `PickAndPlaceTask` is wired to the environment correctly.

**"The environment should be derived from ManagerBasedRLMimicEnv"** — The embodiment's `mimic_env` field must point to a `ManagerBasedRLMimicEnv` subclass (e.g., `CRX5iAMimicEnv`). Check `crx5ia.py` line 147/797.

**Right gripper finger stays open** — Verify `CRX5iARobotiq85JointMirrorAction` is in the action config and `robotiq_85_right_knuckle_joint` is in the `gripper` actuator group.

**State validation mismatches during replay** — Joint stiffness/damping in `ImplicitActuatorCfg` may differ from USD-baked values. Check `patch_usd_gravity.py` was run after USD conversion and gravity is disabled on all rigid bodies.
