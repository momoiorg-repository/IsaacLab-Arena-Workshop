# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import torch
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
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
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaaclab_arena.assets.object_library import ISAACLAB_STAGING_NUCLEUS_DIR
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.observations import gripper_pos
from isaaclab_arena.utils.pose import Pose

_DEFAULT_CAMERA_OFFSET = Pose(position_xyz=(0.11, -0.031, -0.074), rotation_wxyz=(-0.74896, 0.0, 0.0, -0.66262))


# The reason to use our internal panda USD is to combine the panda and the stand within one USD.
# This is not ideal but currently required by the ObjectPlacementSolver to handle the robot placement correctly.
# TODO(cvolk): Move to the IsaacLab supported FRANKA_CFG and handle the handling of the stand internally.
_FRANKA_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
_FRANKA_CFG.spawn.usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/robot_library/franka_panda_hand_on_stand.usd"


@register_asset
class FrankaEmbodiment(EmbodimentBase):
    """Embodiment for the Franka robot."""

    name = "franka"
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
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = FrankaSceneCfg()
        self.action_config = FrankaActionsCfg()
        self.observation_config = FrankaObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = FrankaEventCfg()
        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)
        self.reward_config = FrankaRewardsCfg()
        self.mimic_env = FrankaMimicEnv
        self.camera_config = FrankaCameraCfg()
        self.camera_config._is_tiled_camera = is_tiled_camera
        self.camera_config._camera_offset = camera_offset

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        self.event_config.init_franka_arm_pose.params["default_pose"] = initial_joint_pose

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name


@register_asset
class FrankaJointEmbodiment(FrankaEmbodiment):
    """Franka embodiment variant using joint position actions for GR00T closed-loop inference."""

    name = "franka_joint"

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
        super().__init__(
            enable_cameras,
            initial_pose,
            initial_joint_pose,
            concatenate_observation_terms,
            arm_mode,
            camera_offset,
            is_tiled_camera,
        )
        self.action_config = FrankaJointPositionActionsCfg()

    def get_command_body_name(self) -> str:
        return "panda_hand"


@configclass
class FrankaSceneCfg:
    """Additions to the scene configuration coming from the Franka embodiment."""

    # The robot (combined USD includes both the panda and the stand)
    robot: ArticulationCfg = _FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The end-effector frame marker
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )

    def __post_init__(self):
        # Add a marker to the end-effector frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


class FrankaIKJointRecordingAction(DifferentialInverseKinematicsAction):
    """DifferentialIK action that records IK-solved joint positions as processed_actions.

    The standard DifferentialInverseKinematicsAction stores the scaled EEF delta in
    processed_actions, which is what PostStepProcessedActionsRecorder saves to HDF5.
    GR00T training needs joint position targets, not EEF deltas. This subclass overrides
    processed_actions to return the joint positions computed by the IK solver.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ik_joint_pos = torch.zeros(self.num_envs, len(self._joint_names), device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        """IK-solved joint position targets (recorded into HDF5 processed_actions)."""
        return self._ik_joint_pos

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._ik_joint_pos[:] = joint_pos_des
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)


@configclass
class FrankaIKJointRecordingActionCfg(DifferentialInverseKinematicsActionCfg):
    """Config for FrankaIKJointRecordingAction."""

    class_type: type = FrankaIKJointRecordingAction


class FrankaGripperRecordingAction(BinaryJointPositionAction):
    """Binary gripper action that drives both finger joints but records only 1 DOF.

    The Franka USD does not have a mimic joint constraint, so panda_finger_joint2
    must be driven explicitly. This class controls both fingers symmetrically while
    returning only the first finger's target in processed_actions, keeping the
    total recorded action at 8 DOF (7 arm + 1 gripper).
    """

    @property
    def processed_actions(self) -> torch.Tensor:
        # _processed_actions shape: (num_envs, 2). Return only column 0 so the
        # HDF5 recorder sees 1 gripper DOF, matching 8dof_action_space.yaml.
        return self._processed_actions[:, :1]


@configclass
class FrankaGripperRecordingActionCfg(BinaryJointPositionActionCfg):
    """Config for FrankaGripperRecordingAction."""

    class_type: type = FrankaGripperRecordingAction


class FrankaJointMirrorAction(JointPositionAction):
    """Joint position action that mirrors finger_joint1 to finger_joint2 at inference.

    GR00T outputs 8 DOF (7 arm + 1 gripper). The 8-DOF action is applied normally
    to [panda_joint1..7, panda_finger_joint1]. Since the Franka USD has no mimic
    joint constraint, panda_finger_joint2 is additionally driven with the same
    gripper value via a separate set_joint_position_target call.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._finger2_ids, _ = self._asset.find_joints(["panda_finger_joint2"])

    def apply_actions(self):
        super().apply_actions()
        # Mirror the last (gripper) DOF to panda_finger_joint2
        finger_target = self._processed_actions[:, -1:]
        self._asset.set_joint_position_target(finger_target, joint_ids=self._finger2_ids)


@configclass
class FrankaJointMirrorActionCfg(JointPositionActionCfg):
    """Config for FrankaJointMirrorAction."""

    class_type: type = FrankaJointMirrorAction


@configclass
class FrankaActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = FrankaIKJointRecordingActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    # Both fingers driven symmetrically. FrankaGripperRecordingAction returns only
    # finger_joint1's target in processed_actions, keeping the recorded action at
    # 8 DOF (7 arm + 1 gripper) to match 8dof_action_space.yaml.
    # The Franka USD has no mimic joint constraint, so both fingers must be driven.
    gripper_action: ActionTermCfg = FrankaGripperRecordingActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
        close_command_expr={"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
    )


@configclass
class FrankaJointPositionActionsCfg:
    """Joint position action config for GR00T closed-loop inference.

    Uses direct joint position control (7 arm joints + 1 gripper finger = 8 DOF).
    FrankaJointMirrorAction additionally drives panda_finger_joint2 with the same
    value, since the Franka USD has no mimic joint constraint.
    """

    joint_pos = FrankaJointMirrorActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*", "panda_finger_joint1"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class FrankaCameraCfg:
    """Configuration for cameras.

    Mirrors DROID's DroidCameraCfg pattern: all cameras live on the robot prim so
    they travel with the robot and are auto-wired as observations by make_camera_observation_cfg.
    Field names determine the observation keys (field_name + "_rgb"), e.g.:
      wrist_cam  -> camera_obs["wrist_cam_rgb"]
      left_cam   -> camera_obs["left_cam_rgb"]
      right_cam  -> camera_obs["right_cam_rgb"]
    """

    wrist_cam: CameraCfg | TiledCameraCfg = MISSING
    left_cam: CameraCfg | TiledCameraCfg = MISSING
    right_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        is_tiled_camera = getattr(self, "_is_tiled_camera", False)
        camera_offset = getattr(self, "_camera_offset", _DEFAULT_CAMERA_OFFSET)

        CameraClass = TiledCameraCfg if is_tiled_camera else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        self.wrist_cam = CameraClass(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
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

        # Left external camera — positioned to the left of the workspace (positive Y),
        # mirroring DROID's external_camera layout.
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

        # Right external camera — positioned to the right of the workspace (negative Y),
        # mirroring DROID's external_camera_2 layout.
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


@configclass
class FrankaObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        # Full joint state (absolute) used by the GR00T closed-loop policy.
        # Key name matches DROID's DroidObservationsCfg and gr00t_closedloop_policy.py line 242.
        robot_joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        # joint_pos recorded into HDF5 as training state — must be absolute to match
        # robot_joint_pos used at inference (gr00t_closedloop_policy.py reads robot_joint_pos).
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaEventCfg:
    """Configuration for Franka."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, -0.785, -0.1107, -1.1775, 0.0, 0.785, 0.785, 0.0400, 0.0400],
        },
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FrankaRewardsCfg:
    """Reward specifications for the MDP."""

    action_rate = RewardTermCfg(func=mdp_isaac_lab.action_rate_l2, weight=-0.0001)
    joint_vel = RewardTermCfg(
        func=mdp_isaac_lab.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")}
    )


# This is copied from FrankaCubeStackIKAbsMimicEnv in isaaclab_mimic.
# We copy it as we only need a few methods from it.
# The remaining ones belong to the task.
class FrankaMimicEnv(ManagerBasedRLMimicEnv):
    """Configuration for Franka Mimic."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.
        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.
        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.
        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.
        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # normalized delta position action
        delta_position = target_pos - curr_pos

        # normalized delta rotation action
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if noise is not None:
            noise = noise * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.
        Args:
            action: Environment action. Shape is (num_envs, action_dim)
        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get pose target
        target_pos = curr_pos + delta_position

        # Convert delta_rotation to axis angle form
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        # Handle invalid division for the case when delta_rotation_angle is close to zero
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).
        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).
        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    # Implemented this to consider articulated objects as well
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object(rigid and articulated) in the current scene.
        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.
        Returns:
            A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        state = self.scene.get_state(is_relative=True)

        object_pose_matrix = get_rigid_and_articulated_object_poses(state, env_ids)

        return object_pose_matrix
