Environment Setup and Validation
---------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

On this page we briefly describe the DROID environment and validate that it loads correctly.

The DROID robot is configured as the Franka Panda arm with a **Robotiq 2F-85 gripper** and a
**two-camera rig**: an external overview camera (``external_camera``) and a wrist-mounted camera
(``wrist_camera``).


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: The Kitchen Pick and Place Environment (DROID)
   :animate: fade-in

   .. code-block:: python

      class KitchenPickAndPlaceEnvironment(ExampleEnvironmentBase):

          name: str = "kitchen_pick_and_place"

          def get_env(self, args_cli: argparse.Namespace):
              from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
              from isaaclab_arena.scene.scene import Scene
              from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
              from isaaclab_arena.utils.pose import Pose, PoseRange

              background = self.asset_registry.get_asset_by_name("kitchen")()
              pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
              destination_container = self.asset_registry.get_asset_by_name("pink_container")()
              embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
                  enable_cameras=args_cli.enable_cameras
              )

              pick_up_object.set_initial_pose(
                  PoseRange(
                      position_xyz_min=(0.30, -0.05, 0.80),
                      position_xyz_max=(0.40, 0.05, 0.80),
                  )
              )

              scene = Scene(assets=[background, pick_up_object, destination_container])

              isaaclab_arena_environment = IsaacLabArenaEnvironment(
                  name=self.name,
                  embodiment=embodiment,
                  scene=scene,
                  task=PickAndPlaceTask(pick_up_object, destination_container, background),
              )
              return isaaclab_arena_environment


Robot Comparison: Standard Franka vs DROID
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 35 32 33
   :header-rows: 1

   * - Property
     - Standard Franka
     - DROID
   * - **Arm**
     - Franka Panda
     - Franka Panda
   * - **Gripper**
     - Parallel finger (2 DOF)
     - Robotiq 2F-85 (1 + 5 mimic DOF)
   * - **Cameras**
     - wrist + front (256×256)
     - external + wrist (720×1280)
   * - **Action space**
     - 8 DOF (7 arm + 1 finger)
     - 8 DOF (7 arm + finger_joint)
   * - **State space**
     - 9 DOF (7 arm + 2 fingers)
     - 13 DOF (7 arm + 5 mimic + finger)
   * - **Teleop embodiment**
     - ``franka``
     - ``droid_differential_ik``
   * - **Eval embodiment**
     - ``franka_joint``
     - ``droid_rel_joint_pos``
   * - **GR00T base model**
     - ``nvidia/GR00T-N1.6-3B``
     - ``nvidia/GR00T-N1.6-DROID``
   * - **GR00T tag**
     - ``NEW_EMBODIMENT``
     - ``OXE_DROID``
   * - **Action control**
     - Absolute joint position
     - Relative joint position


Step 1: Validate the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before collecting data, validate that the environment loads and the DROID robot is visible:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --device cpu \
     kitchen_pick_and_place \
     --embodiment droid_differential_ik \
     --object cracker_box

You should see the DROID robot in the kitchen environment with the cracker box on the table.
