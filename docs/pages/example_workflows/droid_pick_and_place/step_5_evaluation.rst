Closed-Loop Policy Inference and Evaluation
--------------------------------------------

**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/droid_pick_and_place
    export MODELS_DIR=/models/isaaclab_arena/droid_pick_and_place

This step demonstrates running the GR00T N1.6-DROID policy in closed-loop and evaluating it
on the DROID pick-and-place task.

Since ``nvidia/GR00T-N1.6-DROID`` supports **zero-shot inference**, you can run this step
directly without completing the preceding data collection and training steps.


Step 1: Download the GR00T N1.6-DROID Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Download Pre-trained GR00T-N1.6-DROID Model (zero-shot evaluation)
   :animate: fade-in

   Download the pre-trained DROID model for zero-shot evaluation:

   .. code-block:: bash

      hf download nvidia/GR00T-N1.6-DROID \
        --local-dir ${MODELS_DIR}/GR00T-N1.6-DROID

   Alternatively, the config can reference the Hugging Face model ID directly:

   .. code-block:: yaml

      model_path: nvidia/GR00T-N1.6-DROID

If you completed :doc:`step_4_policy_training`, point ``model_path`` to your checkpoint instead:

.. code-block:: bash

   # Example: use your fine-tuned checkpoint
   model_path: ${MODELS_DIR}/checkpoint-2000


Step 2: Configure the Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GR00T closed-loop policy is configured via a YAML file at
``isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml``:

.. dropdown:: Configuration file (``droid_manip_gr00t_closedloop_config.yaml``)
   :animate: fade-in

   .. code-block:: yaml

      model_path: /models/GR00T-N1.6-DROID

      language_instruction: "Move the cube to the pink container on the table."
      action_horizon: 32
      embodiment_tag: OXE_DROID
      video_backend: decord

      policy_joints_config_path: isaaclab_arena_gr00t/embodiments/droid/gr00t_8dof_joint_space.yaml
      action_joints_config_path: isaaclab_arena_gr00t/embodiments/droid/8dof_joint_space.yaml
      state_joints_config_path: isaaclab_arena_gr00t/embodiments/droid/13dof_joint_space.yaml

      action_chunk_length: 32
      task_mode_name: droid_manipulation

      pov_cam_name_sim: ["external_camera_rgb", "wrist_camera_rgb"]

      original_image_size: [720, 1280, 3]
      target_image_size: [180, 320, 3]

Key configuration notes:

- ``pov_cam_name_sim[0]`` maps to the GR00T ``exterior_image_1_left`` video key (external camera).
- ``pov_cam_name_sim[1]`` maps to the GR00T ``wrist_image`` video key (wrist camera).
- ``action_chunk_length: 32`` — DROID executes 32 actions per inference rollout (vs. 16 for Franka).


Step 3: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the policy in a single environment with GUI visualization:

.. code-block:: bash

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

Key arguments:

- ``--embodiment droid_rel_joint_pos`` — relative joint position control for closed-loop inference.
  This matches the GR00T-DROID model output (relative joint deltas). **Not** ``droid_differential_ik``
  (used only for teleop and data generation).
- ``--num_steps 2000`` — with ``action_chunk_length=32``, this runs approximately 62 full inference rollouts.
- ``--policy_device cuda`` — runs GR00T inference on GPU while physics runs on CPU.

.. note::

   The embodiment used in closed-loop inference is ``droid_rel_joint_pos`` (relative joint position control),
   which is different from ``droid_differential_ik`` used in teleoperation and data generation.
   GR00T N1.6-DROID outputs relative joint position deltas, so we use ``droid_rel_joint_pos``
   for closed-loop inference.


Step 4: Run Parallel Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation across multiple environments is also supported:

.. tabs::

   .. tab:: Single GPU Evaluation

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --device cpu \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
           --policy_device cuda \
           --enable_cameras \
           --num_steps 2000 \
           --num_envs 10 \
           kitchen_pick_and_place \
           --embodiment droid_rel_joint_pos \
           --object cracker_box

   .. tab:: Distributed Multi-GPU Evaluation

      .. code-block:: bash

         python -m torch.distributed.run --nnode=1 --nproc_per_node=2 \
           isaaclab_arena/evaluation/policy_runner.py \
           --device cpu \
           --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
           --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
           --policy_device cuda \
           --enable_cameras \
           --num_steps 2000 \
           --num_envs 10 \
           --distributed \
           --headless \
           kitchen_pick_and_place \
           --embodiment droid_rel_joint_pos \
           --object cracker_box

During evaluation, the console will report which environments are terminated (task succeeded)
or truncated (episode timeout):

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([3, 8], device='cuda:0') and truncated env_ids: tensor([], device='cuda:0', dtype=torch.int64)

At the end of evaluation, you will see success metrics:

.. code-block:: text

   Metrics: {'success_rate': 0.75, 'num_episodes': 20}
