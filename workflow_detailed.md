# Imitation Learning Workflow Reference

This document provides a concise developer reference for the Franka and DROID pick-and-place
imitation learning workflows. For full step-by-step tutorials with explanations, see the RST docs:

- **Franka Pick and Place**: `docs/pages/example_workflows/franka_pick_and_place/`
- **DROID Pick and Place**: `docs/pages/example_workflows/droid_pick_and_place/`

---

## Robot Setup Overview: Standard Franka vs DROID

| | Standard Franka | DROID |
|---|---|---|
| Arm | Franka Panda | Franka Panda |
| Gripper | Parallel finger (panda_finger_joint) | Robotiq 2F-85 (finger_joint + 5 mimic) |
| Cameras | wrist_cam + left_cam + right_cam (256×256) | external_camera + wrist_camera (720×1280) |
| Action space (sim) | 8 DOF: 7 arm + 1 finger | 8 DOF: 7 arm + finger_joint |
| State space (sim) | 9 DOF: 7 arm + 2 fingers | 13 DOF: 8 arm + 5 mimic gripper |
| Teleop embodiment | `franka` | `droid_differential_ik` |
| Eval embodiment | `franka_joint` | `droid_rel_joint_pos` |
| GR00T base model | `nvidia/GR00T-N1.6-3B` (requires fine-tune) | `nvidia/GR00T-N1.6-DROID` (zero-shot or fine-tune) |
| GR00T tag | `NEW_EMBODIMENT` | `OXE_DROID` |
| Action control | Absolute joint position | **Relative** joint position |

> **Note**: The DROID setup uses the same Panda arm but a different gripper and camera rig.
> The `nvidia/GR00T-N1.6-DROID` model supports **zero-shot inference** (steps 1–4 are optional for evaluation-only runs).

---

## DROID Workflow

```bash
export DATASET_DIR="/workspaces/isaaclab_arena/output"
```

### 1. Record Demonstrations (Teleoperation)

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --device cpu \
  --enable_cameras \
  --dataset_file ${DATASET_DIR}/droid_demo.hdf5 \
  --num_demos 10 \
  --num_success_steps 2 \
  kitchen_pick_and_place \
  --embodiment droid_differential_ik \
  --object cracker_box \
  --teleop_device keyboard
```

- `--embodiment droid_differential_ik` — IK controller. Records `processed_actions` (joint targets) automatically.

---

### 2. Replay Demonstrations (Verification)

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
  --device cpu \
  --dataset_file ${DATASET_DIR}/droid_demo.hdf5 \
  kitchen_pick_and_place \
  --object cracker_box \
  --embodiment droid_differential_ik
```

---

### 3. Annotate Demonstrations (Mimic Preparation)

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
  --device cpu \
  --input_file  ${DATASET_DIR}/droid_demo.hdf5 \
  --output_file ${DATASET_DIR}/droid_demo_annotated.hdf5 \
  --mimic \
  --enable_cameras \
  kitchen_pick_and_place \
  --object cracker_box \
  --embodiment droid_differential_ik
```

---

### 4. Generate Dataset (Mimic Generation)

> **Important**: Use `droid_differential_ik` during data generation (not `droid_rel_joint_pos`).

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
  --device cpu \
  --enable_cameras \
  --input_file  ${DATASET_DIR}/droid_demo_annotated.hdf5 \
  --output_file ${DATASET_DIR}/droid_dataset.hdf5 \
  --num_envs 5 \
  --generation_num_trials 100 \
  --mimic \
  kitchen_pick_and_place \
  --object cracker_box \
  --embodiment droid_differential_ik
```

---

### 5. Transform to LeRobot Format

```bash
python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/droid_pick_place_config.yaml
```

Config: `isaaclab_arena_gr00t/lerobot/config/droid_pick_place_config.yaml`
- `pov_cam_name_sim: "wrist_camera_rgb"` → `observation.images.wrist_image`
- `front_cam_name_sim: "external_camera_rgb"` → `observation.images.exterior_image_1_left`

---

### 6. Train / Fine-tune GR00T N1.6-DROID (Optional — zero-shot works)

```bash
CUDA_VISIBLE_DEVICES=0 python \
  submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-DROID \
  --dataset-path ${DATASET_DIR}/droid_dataset/lerobot \
  --embodiment-tag OXE_DROID \
  --tune-projector \
  --tune-diffusion-model \
  --no-tune-llm \
  --no-tune-visual \
  --max-steps 2000 \
  --use-wandb
```

---

### 7. Closed-Loop Inference (Zero-Shot or Fine-tuned)

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --device cpu \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
  --policy_device cuda \
  --enable_cameras \
  --num_steps 2000 \
  kitchen_pick_and_place \
  --embodiment droid_rel_joint_pos \
  --object cracker_box
```

- `--embodiment droid_rel_joint_pos` — relative joint position control. Matches GR00T-DROID output (relative deltas).
- `droid_manip_gr00t_closedloop_config.yaml` — `OXE_DROID` tag, action_horizon=32, 2-camera setup.

---

## Standard Franka Workflow (parallel finger gripper)

```bash
export DATASET_DIR="/workspaces/isaaclab_arena/output"
```

### 1. Record

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --device cpu --enable_cameras \
  --dataset_file ${DATASET_DIR}/franka_demo.hdf5 \
  --num_demos 5 --num_success_steps 2 \
  table_pick_and_place \
  --embodiment franka \
  --object dex_cube \
  --teleop_device keyboard
```

### 2. Annotate

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
  --device cpu \
  --input_file  ${DATASET_DIR}/franka_demo.hdf5 \
  --output_file ${DATASET_DIR}/franka_demo_annotated.hdf5 \
  --mimic --enable_cameras \
  table_pick_and_place --object dex_cube --embodiment franka
```

### 3. Generate Dataset

```bash
LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
  --device cpu --enable_cameras \
  --input_file  ${DATASET_DIR}/franka_demo_annotated.hdf5 \
  --output_file ${DATASET_DIR}/franka_dataset.hdf5 \
  --num_envs 5 --generation_num_trials 20 --mimic \
  table_pick_and_place --object dex_cube --embodiment franka
```

Headless mode

```bash
python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
  --device cpu --enable_cameras --headless \
  --input_file  ${DATASET_DIR}/franka_demo_annotated.hdf5 \
  --output_file ${DATASET_DIR}/franka_dataset.hdf5 \
  --num_envs 10 --generation_num_trials 1000 --mimic \
  table_pick_and_place --object dex_cube --embodiment franka
```

### 4. Convert to LeRobot

```bash
python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/franka_pick_place_config.yaml
```

Config: `isaaclab_arena_gr00t/lerobot/config/franka_pick_place_config.yaml`
- `pov_cam_name_sim: "wrist_cam_rgb"` → `observation.images.ego_view`
- `front_cam_name_sim: "left_cam_rgb"` → `observation.images.left_view`
- `right_cam_name_sim: "right_cam_rgb"` → `observation.images.right_view`

### 5. Train GR00T N1.6

```bash
CUDA_VISIBLE_DEVICES=0 python \
  submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path ${DATASET_DIR}/franka_dataset/lerobot \
  --modality-config-path isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --tune-projector \
  --tune-diffusion-model \
  --no-tune-llm \
  --no-tune-visual \
  --max-steps 20000 \
  --use-wandb
```

### 6. Eval

```bash
python isaaclab_arena/evaluation/policy_runner.py \
  --device cpu \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/franka_manip_gr00t_closedloop_config.yaml \
  --policy_device cuda --enable_cameras --num_steps 200 \
  table_pick_and_place --embodiment franka_joint --object dex_cube
```

- `--embodiment franka_joint` — absolute joint position control. Matches GR00T Franka output (decoded to absolute).

---

## Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATASET_DIR` | `/workspaces/isaaclab_arena/output` | Output files base directory |
| `MODELS_DIR` | `/workspaces/isaaclab_arena/models` | Model checkpoint directory |
| `LIVESTREAM` | `2` | Headless remote visualization via WebRTC (ports 4700–4900) |
| `--device cpu` | — | Physics simulation device. Policy inference always uses `--policy_device cuda` |
