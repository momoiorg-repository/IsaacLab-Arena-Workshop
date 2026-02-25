Data Generation
----------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

.. note::

   Since ``nvidia/GR00T-N1.6-DROID`` supports **zero-shot inference**, this step is optional.
   You can skip directly to :doc:`step_5_evaluation` to evaluate the pre-trained model.

This step covers generating a large dataset using
`Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_,
which augments a small set of hand-recorded demonstrations into hundreds of varied episodes.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Teleoperation Data Collection) <step_2_teleoperation>`.


.. _droid_step_1_annotate_demonstrations:

Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotate the recorded demonstrations with subtask boundaries so that Isaac Lab Mimic can generate
augmented variations. The pick-and-place task has two subtasks: **reach** and **place**.

For more details on mimic annotation, refer to the
`Isaac Lab Mimic documentation <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#annotate-the-demonstrations>`_.

.. code-block:: bash

   LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
     --device cpu \
     --input_file  ${DATASET_DIR}/droid_demo.hdf5 \
     --output_file ${DATASET_DIR}/droid_demo_annotated.hdf5 \
     --mimic \
     --enable_cameras \
     kitchen_pick_and_place \
     --object cracker_box \
     --embodiment droid_differential_ik

Follow the on-screen CLI instructions to mark subtask boundaries for each recorded demonstration:

1. **Reach:** Robot reaches toward the cracker box
2. **Place:** Robot places the cracker box on the target


.. _droid_step_2_generate_augmented_dataset:

Step 2: Generate Augmented Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic generates additional demonstrations by applying rigid body transformations to
introduce object position variations.

.. important::

   Use ``--embodiment droid_differential_ik`` during data generation.
   This is the IK-controlled embodiment that records ``processed_actions`` (joint targets) that GR00T trains on.
   Do **not** use ``droid_rel_joint_pos`` or ``droid_abs_joint_pos`` here — those are for inference only.

.. code-block:: bash

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

Key arguments:

- ``--num_envs 5`` — run 5 parallel environments to speed up generation.
- ``--generation_num_trials 100`` — attempt 100 variations; successful ones are kept.

Data generation takes 30–60 minutes depending on hardware.
Remove ``--headless`` to visualize the generation process.


Step 3: Validate Generated Data (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the generated dataset to verify it visually:

.. code-block:: bash

   LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --device cpu \
     --enable_cameras \
     --dataset_file ${DATASET_DIR}/droid_dataset.hdf5 \
     kitchen_pick_and_place \
     --embodiment droid_differential_ik \
     --object cracker_box

You should see the DROID arm performing varied pick-and-place demonstrations across different
object positions.

.. note::

   The dataset was generated using CPU device physics, so the replay uses ``--device cpu``
   to ensure reproducibility.
