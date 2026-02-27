# Franka Closed-Loop Inference & Mimic Generation Guide

This document explains the standard workflow for generating demonstration datasets and training the GR00T policy for the Franka robot in IsaacLab-Arena. It specifically addresses common misconceptions regarding action spaces and embodiments during the data generation phase.

## The Core Concept: Two Distinct Action Spaces

The IsaacLab imitation learning workflow for Franka relies on two separate representations of the robot's actions:

1.  **Inverse Kinematics (IK) Space (6D EEF Pose + 1D Gripper):** This is intuitive for human teleoperation and for the internal logic of the Mimic data generator. The system commands the robot by telling it *where the hand should go*.
2.  **Joint Space (7D Arm Joints + 1D Gripper):** This is what the GR00T policy predicts. It commands the robot by specifying the *exact angle for each of the 7 arm joints*.

## The Workflow Phases & Embodiment Selection

Because of these two action spaces, you must use different "embodiments" (configurations of the robot) depending on the phase of the workflow.

### Phase 1: Teleoperation & Mimic Data Generation
**Embodiment to use:** `--embodiment franka`

*   **Why?** The `franka` embodiment uses [DifferentialInverseKinematicsActionCfg](file:///home/an/Workspace/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py#259-293).
*   **How it works:**
    *   During teleoperation or Mimic dataset generation (where new trajectories are synthesized by shifting objects), the system outputs End-Effector (target pose) commands.
    *   The [DifferentialIKController](file:///home/an/Workspace/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab/isaaclab/controllers/differential_ik.py#17-241) inside the `franka` embodiment calculates the required 8D joint angles to achieve that pose.
    *   The simulator steps using these joint angles.
    *   **Crucially**, [ActionStateRecorderManager](file:///home/an/Workspace/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab/isaaclab/envs/mdp/recorders/recorders_cfg.py#55-64) logs *both*:
        *   [action](file:///home/an/Workspace/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py#416-420): The 7D EEF pose commanded by the user or Mimic algorithm.
        *   [processed_actions](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py#133-137): The 8D joint targets computed by the IK solver.

### Phase 2: LeRobot Dataset Conversion
**Config to use:** [isaaclab_arena_gr00t/lerobot/config/franka_pick_place_config.yaml](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena_gr00t/lerobot/config/franka_pick_place_config.yaml)

*   **How it works:** The conversion script ([convert_hdf5_to_lerobot.py](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py)) is configured to extract `"processed_actions"` (the 8D joint angles) from the HDF5 file and map them to the `"action"` key in the final LeRobot dataset. This ensures the GR00T policy is trained to predict joint angles, not EEF poses.

### Phase 3: Policy Training & Closed-Loop Inference
**Embodiment to use:** `--embodiment franka_joint`

*   **Why?** The GR00T policy has been trained on the 8D joint angles ([processed_actions](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py#133-137)). Therefore, during evaluation, the policy directly outputs 8D joint position commands.
*   **How it works:** The `franka_joint` embodiment uses [FrankaJointPositionActionsCfg](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena/embodiments/franka/franka.py#199-213) (direct joint control). It bypasses the IK solver entirely, allowing the policy to control the joints directly.

## The "Action Dimension Mismatch" Error Explained

If you attempt to run [generate_dataset.py](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena/scripts/imitation_learning/generate_dataset.py) with `--embodiment franka_joint` (and the `--mimic` flag), the script will crash or hang due to a tensor dimension mismatch.

*   **The Error:** The Mimic algorithm ([FrankaMimicEnv](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena/embodiments/franka/franka.py#304-440)) outputs 7D EEF pose actions. However, the `franka_joint` embodiment is expecting 8D joint position actions.
*   **The Mistaken Fix:** It is tempting to write custom Inverse Kinematics code inside [FrankaMimicEnv](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena/embodiments/franka/franka.py#304-440) to convert the 7D EEF pose into 8D joint angles so that the `franka_joint` embodiment accepts it. **Do not do this.** It adds redundant computational overhead and often causes async execution hangs in PyTorch.
*   **The Correct Fix:** Simply use `--embodiment franka`. The simulation framework natively handles the IK conversion and logs the necessary joint angles into the [processed_actions](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py#133-137) key.

## Summary Checklist

- [ ] Record Demos: `python scripts/record_demos.py --embodiment franka ...`
- [ ] Generate Dataset (Mimic): `python scripts/imitation_learning/generate_dataset.py --embodiment franka --mimic ...`
- [ ] Convert to LeRobot: Extract [processed_actions](file:///home/an/Workspace/IsaacLab-Arena/isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py#133-137) as the target action.
- [ ] Evaluate Policy: `python scripts/eval_policy.py --embodiment franka_joint ...`
