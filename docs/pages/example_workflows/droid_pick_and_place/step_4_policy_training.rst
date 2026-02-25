Policy Post-training
---------------------

**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/droid_pick_and_place
    export MODELS_DIR=/models/isaaclab_arena/droid_pick_and_place

.. note::

   Since ``nvidia/GR00T-N1.6-DROID`` supports **zero-shot inference**, this step is optional.
   You can skip directly to :doc:`step_5_evaluation` to evaluate the pre-trained model without fine-tuning.

This step covers post-training the GR00T N1.6-DROID model on your custom DROID pick-and-place dataset.
Note that this tutorial assumes that you've completed the
:doc:`preceding step (Data Generation) <step_3_data_generation>`.


Step 1: Convert to LeRobot Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GR00T N1.6 requires the dataset to be in LeRobot format.
We provide a script to convert from the Isaac Lab Mimic generated HDF5 dataset.

.. code-block:: bash

   python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
     --yaml_file isaaclab_arena_gr00t/lerobot/config/droid_pick_place_config.yaml

This creates a folder ``${DATASET_DIR}/droid_dataset/lerobot`` containing parquet files with
states and actions, MP4 camera recordings (external and wrist views), and dataset metadata.
The converter is controlled by ``isaaclab_arena_gr00t/lerobot/config/droid_pick_place_config.yaml``.

.. dropdown:: Configuration file (``droid_pick_place_config.yaml``)
   :animate: fade-in

   .. code-block:: yaml

      data_root: "/datasets/isaaclab_arena/droid_pick_and_place"
      language_instruction: "Move the cube to the pink container on the table."
      task_index: 0
      hdf5_name: "droid_dataset.hdf5"

      # HDF5 field names
      state_name_sim: "joint_pos"
      action_name_sim: "processed_actions"
      pov_cam_name_sim: "wrist_camera_rgb"
      front_cam_name_sim: "external_camera_rgb"

      # LeRobot field names (matches OXE_DROID modality keys)
      state_name_lerobot: "observation.state"
      action_name_lerobot: "action"
      video_name_lerobot: "observation.images.wrist_image"
      front_video_name_lerobot: "observation.images.exterior_image_1_left"
      task_description_lerobot: "annotation.human.task_description"

      # Joint configs
      policy_joints_config_path: "isaaclab_arena_gr00t/embodiments/droid/gr00t_8dof_joint_space.yaml"
      policy_action_joints_config_path: "isaaclab_arena_gr00t/embodiments/droid/gr00t_8dof_joint_space.yaml"
      action_joints_config_path: "isaaclab_arena_gr00t/embodiments/droid/8dof_joint_space.yaml"
      state_joints_config_path: "isaaclab_arena_gr00t/embodiments/droid/13dof_joint_space.yaml"

      fps: 30
      original_image_size: [720, 1280, 3]
      target_image_size: [180, 320, 3]


Step 2: Post-train GR00T N1.6-DROID
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GR00T N1.6 policy has 3 billion parameters. We provide two post-training options:

- **Best Quality**: 8 GPUs with 48 GB memory
- **Low Hardware Requirements**: 1 GPU with 24 GB memory

.. tabs::

   .. tab:: Best Quality

      Training takes approximately 4–8 hours on 8× L40s GPUs.

      Training Configuration:

      - **Base Model:** GR00T-N1.6-DROID
      - **Tuned Modules:** Projector, diffusion model
      - **Frozen Modules:** LLM, visual backbone
      - **Batch Size:** 24 (adjust based on GPU memory)
      - **Training Steps:** 2,000
      - **GPUs:** 8 (multi-GPU training)

      .. code-block:: bash

         python -m torch.distributed.run --nproc_per_node=8 --standalone \
           submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
           --base-model-path nvidia/GR00T-N1.6-DROID \
           --dataset-path ${DATASET_DIR}/droid_dataset/lerobot \
           --output-dir ${MODELS_DIR} \
           --embodiment-tag OXE_DROID \
           --tune-projector \
           --tune-diffusion-model \
           --no-tune-llm \
           --no-tune-visual \
           --global-batch-size 24 \
           --max-steps 2000 \
           --num-gpus 8 \
           --save-steps 500 \
           --save-total-limit 5 \
           --dataloader-num-workers 16 \
           --use-wandb

   .. tab:: Low Hardware Requirements

      Training takes approximately 2–3 hours on 1× Ada6000 GPU.

      Training Configuration:

      - **Base Model:** GR00T-N1.6-DROID
      - **Tuned Modules:** Projector, diffusion model
      - **Frozen Modules:** LLM, visual backbone
      - **Batch Size:** 16 (adjust based on GPU memory)
      - **Training Steps:** 2,000
      - **GPUs:** 1 (single-GPU training)

      .. code-block:: bash

         CUDA_VISIBLE_DEVICES=0 python \
           submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
           --base-model-path nvidia/GR00T-N1.6-DROID \
           --dataset-path ${DATASET_DIR}/droid_dataset/lerobot \
           --output-dir ${MODELS_DIR} \
           --embodiment-tag OXE_DROID \
           --tune-projector \
           --tune-diffusion-model \
           --no-tune-llm \
           --no-tune-visual \
           --global-batch-size 16 \
           --max-steps 2000 \
           --num-gpus 1 \
           --save-steps 500 \
           --save-total-limit 5 \
           --dataloader-num-workers 16 \
           --use-wandb

Key arguments:

- ``--base-model-path nvidia/GR00T-N1.6-DROID`` — downloads the DROID-pretrained foundation model from
  Hugging Face automatically. This model supports zero-shot inference on DROID-like tasks.
- ``--embodiment-tag OXE_DROID`` — uses the DROID dataset's relative joint position control schema.
- ``--no-tune-llm``, ``--no-tune-visual`` — freeze the LLM and visual backbone to prevent catastrophic forgetting.

See the `GR00T fine-tuning guidelines <https://github.com/NVIDIA/Isaac-GR00T#3-fine-tuning>`_
for information on how to adjust the training configuration to your hardware.

.. note::

   The DROID modality uses ``ActionRepresentation.RELATIVE`` **without** ``state_key``,
   meaning the model outputs **relative** joint position deltas directly. The closed-loop policy
   runner uses ``RelativeJointPositionActionCfg`` (``droid_rel_joint_pos`` embodiment) to match this.
   This is different from the Franka workflow which uses absolute joint positions.
