Data Generation
----------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

This step covers generating a large dataset using
`Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_,
which augments a small set of hand-recorded demonstrations into hundreds of varied episodes.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Teleoperation Data Collection) <step_2_teleoperation>`.


.. dropdown:: Download Pre-annotated Dataset (skip annotation step)
   :animate: fade-in

   To skip the annotation step, download the pre-annotated dataset from Hugging Face:

   .. code-block:: bash

      hf download \
         umegan/isaaclab-arena-franka-dataset \
         franka_demo_annotated.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR


.. _franka_step_1_annotate_demonstrations:

Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotate the recorded demonstrations with subtask boundaries so that Isaac Lab Mimic can generate
augmented variations. The pick-and-place task has two subtasks: **reach** and **place**.

For more details on mimic annotation, refer to the
`Isaac Lab Mimic documentation <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#annotate-the-demonstrations>`_.

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
     --device cpu \
     --input_file  ${DATASET_DIR}/franka_demo.hdf5 \
     --output_file ${DATASET_DIR}/franka_demo_annotated.hdf5 \
     --mimic \
     --enable_cameras \
     franka_pick_and_place \
     --object dex_cube \
     --embodiment franka

Follow the on-screen CLI instructions to mark subtask boundaries for each recorded demonstration:

1. **Reach:** Robot reaches toward the cube
2. **Place:** Robot places the cube into the container


.. dropdown:: Download Pre-generated Dataset (skip data generation step)
   :animate: fade-in

   To skip data generation, download the pre-generated dataset from Hugging Face:

   .. code-block:: bash

      hf download \
         umegan/isaaclab-arena-franka-dataset \
         franka_dataset.hdf5 \
         --repo-type dataset \
         --local-dir $DATASET_DIR


.. _franka_step_2_generate_augmented_dataset:

Step 2: Generate Augmented Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic generates additional demonstrations by applying rigid body transformations to
introduce object position variations.

.. important::

   Use ``--embodiment franka`` (IK-controlled) during data generation, **not** ``franka_joint``.
   The IK controller is what records ``processed_actions`` (joint targets) that GR00T trains on.
   Using ``franka_joint`` here will cause a crash.

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
     --device cpu \
     --enable_cameras \
     --input_file  ${DATASET_DIR}/franka_demo_annotated.hdf5 \
     --output_file ${DATASET_DIR}/franka_dataset.hdf5 \
     --num_envs 5 \
     --generation_num_trials 100 \
     --mimic \
     franka_pick_and_place \
     --object dex_cube \
     --embodiment franka

Key arguments:

- ``--num_envs 5`` — run 5 parallel environments to speed up generation.
- ``--generation_num_trials 100`` — attempt 100 variations; successful ones are kept.

Data generation takes 30–60 minutes depending on hardware.


Step 3: Validate Generated Data (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the generated dataset to verify it visually:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --device cpu \
     --enable_cameras \
     --dataset_file ${DATASET_DIR}/franka_dataset.hdf5 \
     franka_pick_and_place \
     --embodiment franka \
     --object dex_cube

You should see the Franka arm performing varied pick-and-place demonstrations across different
cube positions.

.. note::

   The dataset was generated using CPU device physics, so the replay uses ``--device cpu``
   to ensure reproducibility.
