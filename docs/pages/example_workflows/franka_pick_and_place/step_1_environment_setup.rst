Environment Setup and Validation
---------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:

On this page we briefly describe the environment used in this example workflow
and validate that we can load it in Isaac Lab.


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../../images/franka_pick_and_place.png
   :width: 100%
   :alt: Franka pick and place task view


.. dropdown:: The Table Pick and Place Environment
   :animate: fade-in

   .. code-block:: python

      class FrankaPickAndPlaceEnvironment(ExampleEnvironmentBase):

          name: str = "franka_pick_and_place"

          def get_env(self, args_cli: argparse.Namespace):
              from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
              from isaaclab_arena.scene.scene import Scene
              from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
              from isaaclab_arena.utils.pose import Pose, PoseRange
              from isaaclab_arena.assets.object_base import ObjectType
              from isaaclab_arena.assets.object_reference import ObjectReference

              background = self.asset_registry.get_asset_by_name("galileo")()
              pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
              destination_container = self.asset_registry.get_asset_by_name("red_container")()
              embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
                  enable_cameras=args_cli.enable_cameras
              )

              destination_location = ObjectReference(
                  name="destination_location",
                  prim_path="{ENV_REGEX_NS}/red_container",
                  parent_asset=destination_container,
                  object_type=ObjectType.RIGID,
              )

              pick_up_object.set_initial_pose(
                  PoseRange(
                      position_xyz_min=(0.53, -0.15, 0.30),
                      position_xyz_max=(0.65, 0.15, 0.30),
                  )
              )

              destination_container.set_initial_pose(
                  Pose(
                      position_xyz=(0.68, 0.31, 0.23),
                      rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                  )
              )

              scene = Scene(assets=[background, pick_up_object, destination_container, destination_location])

              isaaclab_arena_environment = IsaacLabArenaEnvironment(
                  name=self.name,
                  embodiment=embodiment,
                  scene=scene,
                  task=PickAndPlaceTask(
                      pick_up_object,
                      destination_location,
                      background,
                      destination_object=destination_container,
                  ),
              )
              return isaaclab_arena_environment


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Interact with the Asset Registry**

.. code-block:: python

   background = self.asset_registry.get_asset_by_name("galileo")()
   pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
   destination_container = self.asset_registry.get_asset_by_name("red_container")()
   embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

Here we select the components for the task: the Galileo room as the background, a dex cube as the
pick-up object, and a red container as the placement target. The ``AssetRegistry`` and ``DeviceRegistry``
have been initialized in the ``ExampleEnvironmentBase`` class.
See :doc:`../../concepts/concept_assets_design` for details on asset architecture.

**2. Position the Objects**

.. code-block:: python

   pick_up_object.set_initial_pose(
       PoseRange(
           position_xyz_min=(0.53, -0.15, 0.30),
           position_xyz_max=(0.65, 0.15, 0.30),
       )
   )
   destination_container.set_initial_pose(
       Pose(position_xyz=(0.68, 0.31, 0.23), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
   )

The cube uses ``PoseRange`` to introduce position randomization over episodes, while the container
is fixed. Both are placed on the white table in the Galileo room.

**3. Compose the Scene**

.. code-block:: python

   scene = Scene(assets=[background, pick_up_object, destination_container, destination_location])

All assets are assembled into an Isaac Lab - Arena scene.
See :doc:`../../concepts/concept_scene_design` for scene composition details.

**4. Create the IsaacLab Arena Environment**

.. code-block:: python

   isaaclab_arena_environment = IsaacLabArenaEnvironment(
       name=self.name,
       embodiment=embodiment,
       scene=scene,
       task=PickAndPlaceTask(pick_up_object, destination_location, background, destination_object=destination_container),
   )

The ``PickAndPlaceTask`` encapsulates the pick-and-place objective. ``IsaacLabArenaEnvironment`` is the
top-level container connecting the embodiment, scene, and task.
See :doc:`../../concepts/concept_environment_design` for environment composition details.


Step 1: Validate the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before collecting data, validate that the environment loads and the Franka robot is visible:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --device cpu \
     franka_pick_and_place \
     --embodiment franka \
     --object dex_cube

You should see the Franka Panda arm in the Galileo table environment with the cube and red container
on the table. The environment will reset automatically.
