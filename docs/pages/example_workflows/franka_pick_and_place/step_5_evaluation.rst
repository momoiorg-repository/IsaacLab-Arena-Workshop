Closed-Loop Policy Inference and Evaluation
--------------------------------------------

**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/franka_pick_and_place
    export MODELS_DIR=/models/isaaclab_arena/franka_pick_and_place

This step demonstrates running the trained GR00T N1.6 policy in closed-loop and evaluating it
on the Franka pick-and-place task.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_4_policy_training>`.


Step 1: Configure the Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GR00T closed-loop policy is configured via a YAML file.
The config at ``isaaclab_arena_gr00t/policy/config/franka_manip_gr00t_closedloop_config.yaml``
should point to your trained checkpoint:

.. dropdown:: Configuration file (``franka_manip_gr00t_closedloop_config.yaml``)
   :animate: fade-in

   .. code-block:: yaml

      model_path: /models/isaaclab_arena/franka_pick_and_place/checkpoint-20000

      language_instruction: "Pick up the cube and place it on the container."
      action_horizon: 16
      embodiment_tag: NEW_EMBODIMENT
      video_backend: decord
      modality_config_path: isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py

      # Action policy joints (8 DOF: 7 arm + 1 gripper finger)
      policy_joints_config_path: isaaclab_arena_gr00t/embodiments/franka/gr00t_8dof_action_space.yaml
      # State policy joints (9 DOF: 7 arm + 2 gripper fingers)
      state_policy_joints_config_path: isaaclab_arena_gr00t/embodiments/franka/gr00t_9dof_joint_space.yaml
      action_joints_config_path: isaaclab_arena_gr00t/embodiments/franka/8dof_action_space.yaml
      state_joints_config_path: isaaclab_arena_gr00t/embodiments/franka/9dof_joint_space.yaml

      action_chunk_length: 16
      task_mode_name: franka_tabletop_manipulation

      pov_cam_name_sim: ["wrist_cam_rgb", "left_cam_rgb", "right_cam_rgb"]

      original_image_size: [256, 256, 3]
      target_image_size: [256, 256, 3]

Update ``model_path`` to point to your trained checkpoint directory.


Step 2: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the policy in a single environment with GUI visualization:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --device cpu \
     --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
     --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/franka_manip_gr00t_closedloop_config.yaml \
     --policy_device cuda \
     --enable_cameras \
     --num_steps 200 \
     table_pick_and_place \
     --embodiment franka_joint \
     --object dex_cube

Key arguments:

- ``--embodiment franka_joint`` — joint position control for closed-loop inference. **Not** ``franka``
  (IK-controlled, used only for teleop and data generation).
- ``--num_steps 200`` — with ``action_chunk_length=16``, this runs approximately 12 full inference rollouts.
- ``--policy_device cuda`` — runs GR00T inference on GPU while physics runs on CPU.

.. note::

   The embodiment used in closed-loop inference is ``franka_joint`` (direct joint position control),
   which is different from ``franka`` used in teleoperation and data generation (IK-controlled).
   GR00T N1.6 is trained on joint positions, so we use ``franka_joint`` for closed-loop inference.


Step 3: Run Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation across multiple environments is also supported:

.. tabs::

   .. tab:: Single GPU Evaluation

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --device cpu \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/franka_manip_gr00t_closedloop_config.yaml \
           --policy_device cuda \
           --enable_cameras \
           --num_steps 2000 \
           --num_envs 10 \
           table_pick_and_place \
           --embodiment franka_joint \
           --object dex_cube

   .. tab:: Distributed Multi-GPU Evaluation

      .. code-block:: bash

         python -m torch.distributed.run --nnode=1 --nproc_per_node=2 \
           isaaclab_arena/evaluation/policy_runner.py \
           --device cpu \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/franka_manip_gr00t_closedloop_config.yaml \
           --policy_device cuda \
           --enable_cameras \
           --num_steps 2000 \
           --num_envs 10 \
           --distributed \
           --headless \
           table_pick_and_place \
           --embodiment franka_joint \
           --object dex_cube

During evaluation, the console will report which environments are terminated (task succeeded)
or truncated (episode timeout):

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([2, 7], device='cuda:0') and truncated env_ids: tensor([], device='cuda:0', dtype=torch.int64)

At the end of evaluation, you will see success metrics:

.. code-block:: text

   Metrics: {'success_rate': 0.85, 'num_episodes': 20}
