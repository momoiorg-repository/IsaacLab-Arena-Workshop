Teleoperation Data Collection
------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

This step covers collecting demonstrations using Isaac Lab Teleop with a keyboard or SpaceMouse.
The Franka arm is controlled via Inverse Kinematics (IK) — you control the end-effector pose,
and the IK controller solves for joint angles. The resulting joint targets (``processed_actions``)
are automatically recorded to the HDF5 file.


Step 1: Set the Dataset Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   export DATASET_DIR=/datasets/isaaclab_arena/franka_pick_and_place


Step 2: Record Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

   .. tab:: Keyboard

      .. code-block:: bash

         LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/record_demos.py \
           --device cpu \
           --enable_cameras \
           --dataset_file ${DATASET_DIR}/franka_demo.hdf5 \
           --num_demos 10 \
           --num_success_steps 2 \
           table_pick_and_place \
           --embodiment franka \
           --object dex_cube \
           --teleop_device keyboard

   .. tab:: SpaceMouse

      .. code-block:: bash

         LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/record_demos.py \
           --device cpu \
           --enable_cameras \
           --dataset_file ${DATASET_DIR}/franka_demo.hdf5 \
           --num_demos 10 \
           --num_success_steps 2 \
           table_pick_and_place \
           --embodiment franka \
           --object dex_cube \
           --teleop_device spacemouse

Key arguments:

- ``--embodiment franka`` — IK-controlled Franka. Records ``processed_actions`` (joint targets) automatically.
- ``--num_demos 10`` — collect 10 successful demonstrations.
- ``--num_success_steps 2`` — require 2 consecutive success steps to count a demo as successful.
- ``LIVESTREAM=2`` — enables remote visualization via WebRTC (ports 4700–4900).

The script saves successful demonstrations to ``${DATASET_DIR}/franka_demo.hdf5``.

.. hint::

   For best results during the recording session:

   - Move slowly and smoothly — the IK controller tracks end-effector targets
   - Complete each pick-and-place in a single continuous motion
   - Collect at least 10 successful demonstrations for Mimic to work well


Step 3: Verify Recorded Demonstrations (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the recorded demonstrations to confirm they were captured correctly:

.. code-block:: bash

   LIVESTREAM=2 python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --device cpu \
     --dataset_file ${DATASET_DIR}/franka_demo.hdf5 \
     table_pick_and_place \
     --embodiment franka \
     --object dex_cube

You should see the Franka arm replaying your demonstrations.
