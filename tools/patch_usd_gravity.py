# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from pxr import PhysxSchema, Usd, UsdPhysics


def disable_gravity_on_stage(usd_path):
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"Failed to open {usd_path}")
        return

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # Add PhysxRigidBodyAPI if not present
            if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rb = PhysxSchema.PhysxRigidBodyAPI(prim)
            physx_rb.CreateDisableGravityAttr(True)
            print(f"Disabled gravity on: {prim.GetPath()}")

    stage.Save()
    print(f"Successfully saved {usd_path}")


if __name__ == "__main__":
    usd_path = "/workspaces/isaaclab_arena/isaaclab_arena/embodiments/crx5ia/usd/crx5ia_robotiq85/crx5ia_robotiq85.usd"
    disable_gravity_on_stage(usd_path)
    app.close()
