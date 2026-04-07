Policy Post-training
---------------------

**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/franka_pick_and_place
    export MODELS_DIR=/models/isaaclab_arena/franka_pick_and_place

This step covers post-training the GR00T N1.6 foundation model on the Franka pick-and-place dataset.
Note that this tutorial assumes that you've completed the
:doc:`preceding step (Data Generation) <step_3_data_generation>` or downloaded the
pre-generated dataset from Hugging Face as described below.

.. dropdown:: Download Pre-generated Dataset (skip preceding steps)
   :animate: fade-in

   To skip teleoperation and data generation, download the Mimic-generated HDF5 dataset:

   .. code-block:: bash

      hf download \
         umegan/isaaclab-arena-franka-dataset \
         franka_dataset.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR

.. dropdown:: Download Pre-converted LeRobot Dataset (skip conversion step)
   :animate: fade-in

   To skip the LeRobot conversion step, download the pre-converted dataset:

   .. code-block:: bash

      hf download \
         umegan/isaaclab-arena-franka-dataset \
         --include "franka_dataset/lerobot/*" \
         --repo-type dataset \
         --local-dir $DATASET_DIR


Step 1: Convert to LeRobot Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GR00T N1.6 requires the dataset to be in LeRobot format.
We provide a script to convert from the Isaac Lab Mimic generated HDF5 dataset.

.. code-block:: bash

   python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
     --yaml_file isaaclab_arena_gr00t/lerobot/config/franka_pick_place_config.yaml

This creates a folder ``${DATASET_DIR}/franka_dataset/lerobot`` containing parquet files with
states and actions, MP4 camera recordings (wrist and front views), and dataset metadata.
The converter is controlled by ``isaaclab_arena_gr00t/lerobot/config/franka_pick_place_config.yaml``.

.. dropdown:: Configuration file (``franka_pick_place_config.yaml``)
   :animate: fade-in

   .. code-block:: yaml

      data_root: "/datasets/isaaclab_arena/franka_pick_and_place"
      language_instruction: "Pick up the cube and place it in the container."
      task_index: 0
      hdf5_name: "franka_dataset.hdf5"

      # HDF5 field names
      state_name_sim: "joint_pos"
      action_name_sim: "processed_actions"
      pov_cam_name_sim: "wrist_cam_rgb"
      front_cam_name_sim: "left_cam_rgb"
      right_cam_name_sim: "right_cam_rgb"

      # LeRobot field names
      state_name_lerobot: "observation.state"
      action_name_lerobot: "action"
      video_name_lerobot: "observation.images.ego_view"
      front_video_name_lerobot: "observation.images.left_view"
      right_video_name_lerobot: "observation.images.right_view"
      task_description_lerobot: "annotation.human.task_description"

      # Joint configs
      policy_joints_config_path: "isaaclab_arena_gr00t/embodiments/franka/gr00t_9dof_joint_space.yaml"
      policy_action_joints_config_path: "isaaclab_arena_gr00t/embodiments/franka/gr00t_8dof_action_space.yaml"
      action_joints_config_path: "isaaclab_arena_gr00t/embodiments/franka/8dof_action_space.yaml"
      state_joints_config_path: "isaaclab_arena_gr00t/embodiments/franka/9dof_joint_space.yaml"

      fps: 30
      original_image_size: [256, 256, 3]
      target_image_size: [256, 256, 3]


Step 2: Post-train GR00T N1.6
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GR00T N1.6 policy has 3 billion parameters. We provide two post-training options:

- **Best Quality**: 8 GPUs with 48 GB memory
- **Low Hardware Requirements**: 1 GPU with 24 GB memory

.. tabs::

   .. tab:: Best Quality

      Training takes approximately 4–8 hours on 8× L40s GPUs.

      Training Configuration:

      - **Base Model:** GR00T-N1.6-3B (foundation model)
      - **Tuned Modules:** Visual backbone, projector, diffusion model
      - **Frozen Modules:** LLM
      - **Batch Size:** 24 (adjust based on GPU memory)
      - **Training Steps:** 20,000
      - **GPUs:** 8 (multi-GPU training)

      .. code-block:: bash

         python -m torch.distributed.run --nproc_per_node=8 --standalone submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
         --dataset_path=$DATASET_DIR/franka_dataset/lerobot \
         --output_dir=$MODELS_DIR \
         --modality_config_path=isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py \
         --global_batch_size=24 \
         --max_steps=20000 \
         --num_gpus=8 \
         --save_steps=5000 \
         --save_total_limit=5 \
         --base_model_path=nvidia/GR00T-N1.6-3B \
         --no_tune_llm \
         --tune_visual \
         --tune_projector \
         --tune_diffusion_model \
         --no-resume \
         --dataloader_num_workers=16 \
         --use-wandb \
         --embodiment_tag=NEW_EMBODIMENT \
         --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

   .. tab:: Low Hardware Requirements

      Training takes approximately 2–3 hours on 1× Ada6000 GPU.

      Training Configuration:

      - **Base Model:** GR00T-N1.6-3B (foundation model)
      - **Tuned Modules:** Visual backbone, projector, diffusion model
      - **Frozen Modules:** LLM
      - **Batch Size:** 16 (adjust based on GPU memory)
      - **Training Steps:** 30,000
      - **GPUs:** 1 (single-GPU training)

      .. code-block:: bash

         CUDA_VISIBLE_DEVICES=0 python submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
         --dataset_path=$DATASET_DIR/franka_dataset/lerobot \
         --output_dir=$MODELS_DIR \
         --modality_config_path=isaaclab_arena_gr00t/embodiments/franka/franka_modality_config.py \
         --global_batch_size=16 \
         --max_steps=30000 \
         --num_gpus=1 \
         --save_steps=5000 \
         --base_model_path=nvidia/GR00T-N1.6-3B \
         --no_tune_llm \
         --tune_visual \
         --tune_projector \
         --tune_diffusion_model \
         --dataloader_num_workers=16 \
         --use-wandb \
         --embodiment_tag=NEW_EMBODIMENT \
         --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
         --save_total_limit=5

Key arguments:

- ``--base_model_path=nvidia/GR00T-N1.6-3B`` — downloads the foundation model from Hugging Face automatically.
- ``--embodiment_tag=NEW_EMBODIMENT`` — used for custom embodiments not in the base model.
- ``--modality_config_path`` — Franka-specific modality config defining video/state/action keys.
- ``--tune_visual``, ``--tune_projector``, ``--tune_diffusion_model`` — tune visual backbone, projector and diffusion model.
- ``--no_tune_llm`` — freeze the LLM to prevent catastrophic forgetting.
- ``--color_jitter_params`` — data augmentation for better real-world generalization.

See the `GR00T fine-tuning guidelines <https://github.com/NVIDIA/Isaac-GR00T#3-fine-tuning>`_
for information on how to adjust the training configuration to your hardware.

.. note::

   The Franka modality config uses ``ActionRepresentation.RELATIVE`` with ``state_key="single_arm"``,
   which means the model learns **relative** joint deltas that are decoded back to **absolute** positions
   by the GR00T decode pipeline. The closed-loop policy runner uses ``JointPositionActionCfg``
   (absolute joint position control) to match this.
