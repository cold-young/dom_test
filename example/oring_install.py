from omni.isaac.kit import SimulationApp
import numpy as np
import os
import sys
import time
import torch

simulation_app = SimulationApp({"headless": False})
from omni.isaac.core import World
 
from omni.isaac.core.prims import RigidPrimView 
from omni.isaac.core.articulations import ArticulationView
from omni.physx.scripts.physicsUtils import *
from omni.isaac.core.physics_context.physics_context import PhysicsContext
import omni.usd
 

# from omni.physx.scripts import deformableUtils, physicsUtils
import omni.kit.commands
from omni.isaac.core.utils.stage import open_stage
from pxr import Gf, Sdf, UsdGeom, Usd, UsdLux, PhysxSchema
from dom_test.example.utils.pcd_util import PointCloudUtil

class Test():
    def __init__(self):
        self._device = "cuda:0"        
        self.gripper_close = False
        self._path = os.getcwd()
        self.obs = None
        
    def init_simulation(self):
        # self.scene.init_simulation()
        self._scene = PhysicsContext()
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.enable_ccd(flag=False)
        # self._scene.set_physics_dt(dt=1.0 / 300.0)

    def main(self):        
        self._path = os.getcwd()
        self.asset_path = self._path + "/dom_test/example/assets/" 
        object_usd_path = self.asset_path + "install_scene_test.usd"
        # object_usd_path = self.asset_path + "test_bed.usd"
        open_stage(usd_path=object_usd_path)
        
        world = World(stage_units_in_meters=1)
        
        world._physics_context.enable_gpu_dynamics(flag=True)
        self.stage = world.stage
        self.init_simulation()

        self.franka = ArticulationView(
            prim_paths_expr="/World/franka_install/franka_gripper", name="robotiq")
        world.scene.add(self.franka)
        self.pcd = PointCloudUtil("/World/franka_install/oring/oring")
        world.reset()
        for _ in range(10):
            world.step(render=True)
        i=0
        while simulation_app.is_running():
            world.step(render=True)
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()

            i+=1
            if i == 10:
                print("Inside pcds/Total pcds:", self.pcd.get_number_of_inside_points(),"/",self.pcd.get_number_of_points())
                i = 0
            # # # ### PD position control##
            djv = self.franka.get_joint_positions(clone=False)
            
            # # djv*= 0
            # djv[:, :6] = 0
            # djv[:, 6] += 0.01
            # djv[:, 7] = djv[:, 6]
            # djv[:, 9] = djv[:, 6]
            # djv[:, 8] = -djv[:, 6]
            # djv[:, 10] = -djv[:, 6]
            # djv[:, 11] = -djv[:, 6]
            # self.robotiq.set_joint_position_targets(djv)  # pd controls
 
        
if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
