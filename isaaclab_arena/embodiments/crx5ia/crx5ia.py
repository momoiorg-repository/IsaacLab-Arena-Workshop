# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg, TiledCameraCfg  # noqa: F401
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.crx5ia.observations import gripper_pos, gripper_pos_robotiq85
from isaaclab_arena.embodiments.droid.droid import BinaryJointPositionZeroToOneActionCfg
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

# --------------------------------------------------------------------------- #
# USD paths — locally-converted USDs (from URDF via conversion scripts)
# --------------------------------------------------------------------------- #
_CRX5IA_USD_DIR = os.path.join(os.path.dirname(__file__), "usd")
_CRX5IA_USD_PATH = os.path.join(_CRX5IA_USD_DIR, "crx5ia_robotiq140", "crx5ia_robotiq_140.usd")
_CRX5IA_ROBOTIQ85_USD_PATH = os.path.join(_CRX5IA_USD_DIR, "crx5ia_robotiq85", "crx5ia_robotiq85.usd")

# Default wrist camera offset on the Robotiq 140 base link
_DEFAULT_CAMERA_OFFSET = Pose(
    position_xyz=(0.05, 0.0, -0.05),
    rotation_wxyz=(-0.74896, 0.0, 0.0, -0.66262),
)

# Wrist camera offset for the Robotiq 85 base link (slightly shorter than 140)
_ROBOTIQ85_CAMERA_OFFSET = Pose(
    position_xyz=(0.16233, 0.01159, 0.06064),
    rotation_wxyz=(-0.48513, -0.50093, 0.52131, 0.49188),
)


# --------------------------------------------------------------------------- #
# ArticulationCfg for CRX5iA + Robotiq 140
# --------------------------------------------------------------------------- #
_CRX5IA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_CRX5IA_USD_PATH,
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # J1-J6 default home position + finger_joint
        joint_pos={
            ".*J1": 0.0,
            ".*J2": -0.523,
            ".*J3": 0.0,
            ".*J4": 0.0,
            ".*J5": -1.5708,  # -90° pointing down
            ".*J6": 0.0,
            ".*finger_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*J[1-6]"],
            velocity_limit=100.0,
            stiffness=3000.0,
            damping=300.0,
            effort_limit=4000.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*finger_joint"],
            velocity_limit=2.0,
            stiffness=37.52,
            damping=0.00125,
            effort_limit=1000.0,
        ),
    },
)


# =========================================================================== #
#                           EMBODIMENT CLASS                                   #
# =========================================================================== #


@register_asset
class CRX5iAEmbodiment(EmbodimentBase):
    """Embodiment for the Fanuc CRX5iA + Robotiq 2F-140 gripper.

    The CRX5iA is a 6-DOF collaborative robot arm. Combined with the
    Robotiq 2F-140 gripper, the total action space is 7 DOF:
    6 arm joints (J1-J6) + 1 gripper joint (finger_joint).
    """

    name = "crx5ia"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        camera_offset: Pose | None = _DEFAULT_CAMERA_OFFSET,
        is_tiled_camera: bool = False,
    ):
        if initial_pose is None:
            initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = CRX5iASceneCfg()
        self.action_config = CRX5iAActionsCfg()
        self.observation_config = CRX5iAObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = CRX5iAEventCfg()
        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)
        self.reward_config = CRX5iARewardsCfg()
        self.mimic_env = CRX5iAMimicEnv
        self.camera_config = CRX5iACameraCfg()
        self.camera_config._is_tiled_camera = is_tiled_camera
        self.camera_config._camera_offset = camera_offset

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        self.event_config.init_arm_pose.params["default_pose"] = initial_joint_pose

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name


@register_asset
class CRX5iAJointEmbodiment(CRX5iAEmbodiment):
    """CRX5iA embodiment variant using joint position actions for GR00T closed-loop inference."""

    name = "crx5ia_joint"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        camera_offset: Pose | None = _DEFAULT_CAMERA_OFFSET,
        is_tiled_camera: bool = False,
    ):
        if initial_pose is None:
            initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        super().__init__(
            enable_cameras,
            initial_pose,
            initial_joint_pose,
            concatenate_observation_terms,
            arm_mode,
            camera_offset,
            is_tiled_camera,
        )
        self.action_config = CRX5iAJointPositionActionsCfg()

    def get_command_body_name(self) -> str:
        return "J6_link"


# =========================================================================== #
#                           SCENE CONFIGURATION                                #
# =========================================================================== #


@configclass
class CRX5iASceneCfg:
    """Scene configuration for the CRX5iA + Robotiq 140."""

    robot: ArticulationCfg = _CRX5IA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # End-effector frame — tracks the flange (gripper mount point)
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/J6_link",
                name="end_effector",
                offset=OffsetCfg(
                    # Offset from J6 to the Robotiq 140 fingertip centre
                    # robotiq_base_joint (0) + robotiq_140_base (0.0315) + ~fingertip
                    pos=[0.0, 0.0, 0.16],
                ),
            ),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


# =========================================================================== #
#                        ACTION CONFIGURATIONS                                 #
# =========================================================================== #


class CRX5iAIKAction(DifferentialInverseKinematicsAction):
    """DifferentialIK action that records IK-solved joint positions as processed_actions.

    Same pattern as Franka's FrankaIKJointRecordingAction: overrides processed_actions
    to return joint positions computed by the IK solver (for HDF5 recording / GR00T training).

    Also fixes null-space drift: instead of resetting the desired EEF pose to the
    current pose every step (the default DifferentialInverseKinematicsAction behaviour
    with use_relative_mode=True), we maintain a persistent tracked desired pose and
    only accumulate user-commanded deltas into it.  When the delta is zero the IK
    target stays fixed, so near-null-space joints (e.g. J4) cannot drift.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ik_joint_pos = torch.zeros(self.num_envs, len(self._joint_names), device=self.device)
        self._tracked_ee_pos: torch.Tensor | None = None
        self._tracked_ee_quat: torch.Tensor | None = None

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._ik_joint_pos

    def reset(self, env_ids=None):
        super().reset(env_ids)
        # Snapshot current joint positions as the held target so zero-action
        # steps keep the arm exactly where it reset to (full kp stiffness).
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        ids = env_ids if env_ids is not None else slice(None)
        self._ik_joint_pos[ids] = joint_pos[ids].clone()
        # Update the tracked EEF reference so the first non-zero action
        # starts from the actual current pose with no accumulated error.
        ee_pos, ee_quat = self._compute_frame_pose()
        if self._tracked_ee_pos is None:
            self._tracked_ee_pos = ee_pos.clone()
            self._tracked_ee_quat = ee_quat.clone()
        else:
            self._tracked_ee_pos[ids] = ee_pos[ids].clone()
            self._tracked_ee_quat[ids] = ee_quat[ids].clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        # Store raw + scaled actions exactly as the parent does
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # Lazily initialise tracked pose on the first call (before first reset)
        if self._tracked_ee_pos is None:
            ee_pos, ee_quat = self._compute_frame_pose()
            self._tracked_ee_pos = ee_pos.clone()
            self._tracked_ee_quat = ee_quat.clone()
        # When action is zero: re-anchor the desired EEF to wherever the arm
        # actually is so the next non-zero command starts from the true pose.
        # When action is non-zero: accumulate the delta into the tracked pose
        # and let apply_actions() re-run IK.
        if self._processed_actions.abs().max().item() < 1e-7:
            ee_pos, ee_quat = self._compute_frame_pose()
            self._tracked_ee_pos[:] = ee_pos
            self._tracked_ee_quat[:] = ee_quat
        else:
            self._tracked_ee_pos, self._tracked_ee_quat = PoseUtils.apply_delta_pose(
                self._tracked_ee_pos, self._tracked_ee_quat, self._processed_actions
            )
            self._ik_controller.ee_pos_des[:] = self._tracked_ee_pos
            self._ik_controller.ee_quat_des[:] = self._tracked_ee_quat

    def apply_actions(self):
        # When action is zero: re-send the cached joint targets unchanged so
        # the full kp stiffness holds the arm in place (same as a baked USD
        # drive target).  Only run IK when the user is actually commanding
        # a pose delta, which avoids the "target = current → zero stiffness"
        # failure mode that causes gravity sag.
        if self._processed_actions.abs().max().item() > 1e-7:
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
            if ee_quat_curr.norm() != 0:
                jacobian = self._compute_frame_jacobian()
                joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
            else:
                joint_pos_des = joint_pos.clone()
            self._ik_joint_pos[:] = joint_pos_des
        self._asset.set_joint_position_target(self._ik_joint_pos, self._joint_ids)


@configclass
class CRX5iAIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = CRX5iAIKAction


@configclass
class CRX5iAActionsCfg:
    """Action specifications for the CRX5iA MDP.

    Uses DifferentialIK for the 6-DOF arm + a simple joint position action
    for the single Robotiq 140 finger_joint.
    """

    arm_action: ActionTermCfg = CRX5iAIKActionCfg(
        asset_name="robot",
        joint_names=[".*J[1-6]"],
        body_name="J6_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.05,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.16],  # offset to fingertip centre
        ),
    )

    gripper_action: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=[".*finger_joint"],
        open_command_expr={".*finger_joint": 0.0},
        close_command_expr={".*finger_joint": 0.7},
    )


@configclass
class CRX5iAJointPositionActionsCfg:
    """Joint position actions for GR00T closed-loop inference.

    Direct joint position control: 6 arm joints + 1 gripper finger = 7 DOF.
    """

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*J[1-6]", ".*finger_joint"],
        scale=1.0,
        use_default_offset=False,
    )


# =========================================================================== #
#                        CAMERA CONFIGURATION                                  #
# =========================================================================== #


@configclass
class CRX5iACameraCfg:
    """Camera configuration — wrist cam on the gripper, external left/right cams."""

    wrist_cam: CameraCfg | TiledCameraCfg = MISSING
    left_cam: CameraCfg | TiledCameraCfg = MISSING
    right_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        is_tiled_camera = getattr(self, "_is_tiled_camera", False)
        camera_offset = getattr(self, "_camera_offset", _DEFAULT_CAMERA_OFFSET)

        CameraClass = TiledCameraCfg if is_tiled_camera else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        self.wrist_cam = CameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/J6_link/wrist_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8, focus_distance=28, horizontal_aperture=5.376, vertical_aperture=3.024
            ),
            offset=OffsetClass(
                pos=camera_offset.position_xyz,
                rot=camera_offset.rotation_wxyz,
                convention="ros",
            ),
        )

        self.left_cam = CameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/left_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8, focus_distance=28, horizontal_aperture=5.376, vertical_aperture=3.024
            ),
            offset=OffsetClass(pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"),
        )

        self.right_cam = CameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/right_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8, focus_distance=28, horizontal_aperture=5.376, vertical_aperture=3.024
            ),
            offset=OffsetClass(pos=(0.05, -0.57, 0.66), rot=(0.805, 0.399, -0.195, -0.393), convention="opengl"),
        )


# =========================================================================== #
#                      OBSERVATION CONFIGURATION                               #
# =========================================================================== #


@configclass
class CRX5iAObservationsCfg:
    """Observation specifications for the CRX5iA MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        robot_joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# =========================================================================== #
#                          EVENT CONFIGURATION                                 #
# =========================================================================== #


@configclass
class CRX5iAEventCfg:
    """Event configuration for CRX5iA."""

    init_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            # J1-J6 default + 6 robotiq finger/knuckle joints
            "default_pose": [
                0.0,  # J1 (Base pan)
                -0.523,  # J2 (Shoulder lift) - lean forward
                0.0,  # J3 (Elbow)
                0.0,  # J4 (Wrist 1)
                -1.5708,  # J5 (Wrist 2) - point gripper straight down
                0.0,  # J6 (Wrist 3)
                0.0,  # finger_joint
                0.0,  # left_inner_finger_joint
                0.0,  # left_inner_knuckle_joint
                0.0,  # right_inner_knuckle_joint
                0.0,  # right_outer_knuckle_joint
                0.0,  # right_inner_finger_joint
            ],
        },
    )
    randomize_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# =========================================================================== #
#                         REWARD CONFIGURATION                                 #
# =========================================================================== #


@configclass
class CRX5iARewardsCfg:
    """Reward specifications for the CRX5iA MDP."""

    action_rate = RewardTermCfg(func=mdp_isaac_lab.action_rate_l2, weight=-0.0001)
    joint_vel = RewardTermCfg(
        func=mdp_isaac_lab.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")}
    )


# =========================================================================== #
#                       MIMIC ENVIRONMENT                                      #
# =========================================================================== #


class CRX5iAMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic environment for the CRX5iA + Robotiq 140."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos

        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if noise is not None:
            noise = noise * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        state = self.scene.get_state(is_relative=True)
        return get_rigid_and_articulated_object_poses(state, env_ids)


# =========================================================================== #
#                    CRX5iA + ROBOTIQ 85 VARIANT                               #
# =========================================================================== #


# --------------------------------------------------------------------------- #
# ArticulationCfg for CRX5iA + Robotiq 85
# --------------------------------------------------------------------------- #
_CRX5IA_ROBOTIQ85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_CRX5IA_ROBOTIQ85_USD_PATH,
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            ".*J1": 0.0,
            ".*J2": -0.523,
            ".*J3": 0.0,
            ".*J4": 0.0,
            ".*J5": -1.5708,  # -90° pointing down
            ".*J6": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*J[1-6]"],
            velocity_limit=100.0,
            stiffness=3000.0,
            damping=300.0,
            effort_limit=4000.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint", "right_outer_knuckle_joint"],
            velocity_limit=2.0,
            stiffness=37.52,
            damping=0.00125,
            effort_limit=1000.0,
        ),
    },
)


@configclass
class CRX5iARobotiq85SceneCfg:
    """Scene configuration for the CRX5iA + Robotiq 85."""

    robot: ArticulationCfg = _CRX5IA_ROBOTIQ85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/J6_link",
                name="end_effector",
                offset=OffsetCfg(
                    # Offset from J6 to Robotiq 85 fingertip centre (~14 cm)
                    pos=[0.0, 0.0, 0.14],
                ),
            ),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


class CRX5iARobotiq85GripperRecordingAction(BinaryJointPositionAction):
    """Binary gripper action that drives both knuckle joints but records only left (1 DOF).

    The Robotiq 85 USD has no mimic-joint tendon, so right_knuckle must be driven
    explicitly with the negated left_knuckle value. processed_actions returns only
    the left_knuckle target, keeping the recorded action at 7 DOF to match the
    robotiq85_7dof_action_space.yaml.
    """

    @property
    def processed_actions(self) -> torch.Tensor:
        # _processed_actions: (num_envs, 2) → return only left column
        return self._processed_actions[:, :1]


@configclass
class CRX5iARobotiq85GripperRecordingActionCfg(BinaryJointPositionZeroToOneActionCfg):
    """Config for CRX5iARobotiq85GripperRecordingAction."""

    class_type: type = CRX5iARobotiq85GripperRecordingAction


class CRX5iARobotiq85JointMirrorAction(JointPositionAction):
    """Joint position action for GR00T inference that mirrors right_knuckle.

    GR00T outputs 7 DOF (J1-J6 + left_knuckle). Since the Robotiq 85 USD has no
    mimic-joint tendon, right_knuckle_joint is additionally driven with the negated
    left_knuckle value (multiplier=-1 per URDF).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._right_knuckle_ids, _ = self._asset.find_joints(["right_outer_knuckle_joint"])

    def apply_actions(self):
        super().apply_actions()
        # right_knuckle mirrors left_knuckle with multiplier -1
        left_target = self._processed_actions[:, -1:]
        self._asset.set_joint_position_target(-left_target, joint_ids=self._right_knuckle_ids)


@configclass
class CRX5iARobotiq85JointMirrorActionCfg(JointPositionActionCfg):
    """Config for CRX5iARobotiq85JointMirrorAction."""

    class_type: type = CRX5iARobotiq85JointMirrorAction


@configclass
class CRX5iARobotiq85ActionsCfg:
    """Action specifications for the CRX5iA + Robotiq 85 MDP.

    Primary gripper joint: finger_joint (range 0.0 open → 0.8 closed).
    Right outer knuckle is driven with the negated value (multiplier=-1 per URDF).
    """

    arm_action: ActionTermCfg = CRX5iAIKActionCfg(
        asset_name="robot",
        joint_names=[".*J[1-6]"],
        body_name="J6_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.05,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.14],
        ),
    )

    gripper_action: ActionTermCfg = CRX5iARobotiq85GripperRecordingActionCfg(
        asset_name="robot",
        joint_names=["finger_joint", "right_outer_knuckle_joint"],
        open_command_expr={
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
        },
        close_command_expr={
            "finger_joint": 0.8,
            "right_outer_knuckle_joint": -0.8,
        },
    )


@configclass
class CRX5iARobotiq85JointPositionActionsCfg:
    """Joint position actions for GR00T closed-loop inference (CRX5iA + Robotiq 85).

    7 DOF: J1-J6 + left_knuckle. CRX5iARobotiq85JointMirrorAction additionally drives
    right_knuckle with the negated value, since the USD has no mimic-joint tendon.
    """

    joint_pos = CRX5iARobotiq85JointMirrorActionCfg(
        asset_name="robot",
        joint_names=[".*J[1-6]", "finger_joint"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class CRX5iARobotiq85ObservationsCfg:
    """Observation specifications for the CRX5iA + Robotiq 85 MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        robot_joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos_robotiq85)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class CRX5iARobotiq85EventCfg:
    """Event configuration for CRX5iA + Robotiq 85."""

    init_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            # J1-J6 + 6 gripper joints (must match USD DOF order exactly)
            "default_pose": [
                0.0,  # J1
                -0.523,  # J2
                0.0,  # J3
                0.0,  # J4
                -1.5708,  # J5
                0.0,  # J6
                0.0,  # finger_joint (primary drive)
                0.0,  # right_outer_knuckle_joint (mimic)
                0.0,  # left_inner_finger_joint (passive)
                0.0,  # right_inner_finger_joint (passive)
                0.0,  # left_inner_finger_knuckle_joint (passive)
                0.0,  # right_inner_finger_knuckle_joint (passive)
            ],
        },
    )
    randomize_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@register_asset
class CRX5iARobotiq85Embodiment(EmbodimentBase):
    """Embodiment for the Fanuc CRX5iA + Robotiq 2F-85 gripper.

    7 DOF: 6 arm joints (J1-J6) + 1 gripper joint (finger_joint, range 0.0-0.8).
    USD: isaaclab_arena/embodiments/crx5ia/usd/crx5ia_robotiq85/crx5ia_robotiq85.usd
    Generate with: tools/convert_crx5ia_robotiq85_urdf_to_usd.sh
    """

    name = "crx5ia_robotiq85"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        camera_offset: Pose | None = _ROBOTIQ85_CAMERA_OFFSET,
        is_tiled_camera: bool = False,
    ):
        if initial_pose is None:
            initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = CRX5iARobotiq85SceneCfg()
        self.action_config = CRX5iARobotiq85ActionsCfg()
        self.observation_config = CRX5iARobotiq85ObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = CRX5iARobotiq85EventCfg()
        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)
        self.reward_config = CRX5iARewardsCfg()
        self.mimic_env = CRX5iAMimicEnv
        self.camera_config = CRX5iACameraCfg()
        self.camera_config._is_tiled_camera = is_tiled_camera
        self.camera_config._camera_offset = camera_offset

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        self.event_config.init_arm_pose.params["default_pose"] = initial_joint_pose

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name


@register_asset
class CRX5iARobotiq85JointEmbodiment(CRX5iARobotiq85Embodiment):
    """CRX5iA + Robotiq 85 variant using joint position actions for GR00T closed-loop inference."""

    name = "crx5ia_robotiq85_joint"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        camera_offset: Pose | None = _ROBOTIQ85_CAMERA_OFFSET,
        is_tiled_camera: bool = False,
    ):
        if initial_pose is None:
            initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        super().__init__(
            enable_cameras,
            initial_pose,
            initial_joint_pose,
            concatenate_observation_terms,
            arm_mode,
            camera_offset,
            is_tiled_camera,
        )
        self.action_config = CRX5iARobotiq85JointPositionActionsCfg()

    def get_command_body_name(self) -> str:
        return "J6_link"
