Franka Pick and Place Task
==========================

.. image:: ../../../images/franka_pick_and_place.gif
   :width: 80%
   :align: center

|

This example demonstrates the complete workflow for the **Franka pick and place task** in Isaac Lab - Arena,
covering environment setup and validation, teleoperation data collection, data generation with Isaac Lab Mimic,
policy post-training, and closed-loop evaluation.

Task Overview
-------------

**Task ID:** ``franka_pick_and_place``

**Task Description:** The Franka Panda arm picks up a cube from a table and places it in a container.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation
   * - **Skills**
     - Reach, Grasp, Place
   * - **Embodiment**
     - Franka Panda (7 DOF arm + parallel finger gripper)
   * - **Interop**
     - Isaac Lab Teleop, Isaac Lab Mimic
   * - **Scene**
     - Galileo table environment
   * - **Objects**
     - Dex cube, red container
   * - **Policy**
     - GR00T N1.6 (vision-language foundation model)
   * - **Post-training**
     - Imitation Learning
   * - **Physics**
     - PhysX (200Hz @ 4 decimation)
   * - **Closed-loop**
     - Yes (50Hz control, absolute joint position)
   * - **Action space**
     - 8 DOF: 7 arm joints + 1 gripper finger
   * - **State space**
     - 9 DOF: 7 arm joints + 2 gripper fingers


Workflow
--------

This tutorial covers the pipeline between creating an environment, generating training data,
fine-tuning a policy (GR00T N1.6), and evaluating the policy in closed-loop.
A user can follow the whole pipeline, or can start at any intermediate step.

Prerequisites
^^^^^^^^^^^^^

Start the isaaclab docker container:

:docker_run_default:

We store data on Hugging Face, so you'll need to log in to Hugging Face if you haven't already:

.. code-block:: bash

    hf auth login

Create the folders for the data and models:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/franka_pick_and_place
    mkdir -p $DATASET_DIR
    export MODELS_DIR=/models/isaaclab_arena/franka_pick_and_place
    mkdir -p $MODELS_DIR

Workflow Steps
^^^^^^^^^^^^^^

Follow the following steps to complete the workflow:

- :doc:`step_1_environment_setup`
- :doc:`step_2_teleoperation`
- :doc:`step_3_data_generation`
- :doc:`step_4_policy_training`
- :doc:`step_5_evaluation`


.. toctree::
   :maxdepth: 1
   :hidden:

   step_1_environment_setup
   step_2_teleoperation
   step_3_data_generation
   step_4_policy_training
   step_5_evaluation
