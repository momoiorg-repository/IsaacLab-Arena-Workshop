Teleoperation Data Collection
------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

.. note::

   Since ``nvidia/GR00T-N1.6-DROID`` supports **zero-shot inference**, this step is optional.
   You can skip directly to :doc:`step_5_evaluation` to evaluate the pre-trained model.

This step covers collecting DROID demonstrations using Isaac Lab Teleop with a keyboard or SpaceMouse.
The DROID arm is controlled via IK Differential (``droid_differential_ik``) — you control end-effector
velocity, and the IK controller tracks it. The resulting joint targets (``processed_actions``) are
automatically recorded to the HDF5 file.


Step 1: Set the Dataset Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   export DATASET_DIR=/datasets/isaaclab_arena/droid_pick_and_place


Step 2: Record Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

   .. tab:: Keyboard

      .. code-block:: bash

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

   .. tab:: SpaceMouse

      .. code-block:: bash

         LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/record_demos.py \
           --device cpu \
           --enable_cameras \
           --dataset_file ${DATASET_DIR}/droid_demo.hdf5 \
           --num_demos 10 \
           --num_success_steps 2 \
           kitchen_pick_and_place \
           --embodiment droid_differential_ik \
           --object cracker_box \
           --teleop_device spacemouse

Key arguments:

- ``--embodiment droid_differential_ik`` — differential IK controller. Records ``processed_actions``
  (joint targets) automatically. This is the correct embodiment for data collection.
- ``--num_demos 10`` — collect 10 successful demonstrations.
- ``--num_success_steps 2`` — require 2 consecutive success steps to count a demo as successful.
- ``LIVESTREAM=2`` — enables remote visualization via WebRTC (ports 4700–4900).

The script saves successful demonstrations to ``${DATASET_DIR}/droid_demo.hdf5``.

.. hint::

   For best results during the recording session:

   - Move slowly and smoothly — the IK controller tracks end-effector velocity
   - Collect at least 10 successful demonstrations for Mimic to work well
   - The DROID gripper uses Robotiq 2F-85; map keyboard/SpaceMouse keys to open/close accordingly


Step 3: Verify Recorded Demonstrations (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the recorded demonstrations to confirm they were captured correctly:

.. code-block:: bash

   LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --device cpu \
     --dataset_file ${DATASET_DIR}/droid_demo.hdf5 \
     kitchen_pick_and_place \
     --embodiment droid_differential_ik \
     --object cracker_box

You should see the DROID arm replaying your demonstrations.
