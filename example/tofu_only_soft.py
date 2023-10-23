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

class Test():
    def __init__(self):
        self._device = "cuda:0"        
        self.gripper_close = False
        self._path = os.getcwd()
        self.obs = None
        
    def init_simulation(self):
        print("sibal")
        # self.scene.init_simulation()
        self._scene = PhysicsContext()
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.enable_ccd(flag=False)
        # self._scene.set_physics_dt(dt=1.0 / 300.0)

    def main(self):        
        self._path = os.getcwd()
        self.asset_path = self._path + "/dom_test/example/assets/" 
        object_usd_path = self.asset_path + "tofu_gripper_2.usd"
        open_stage(usd_path=object_usd_path)
        
        world = World(stage_units_in_meters=1)
        
        world._physics_context.enable_gpu_dynamics(flag=True)
        self.stage = world.stage
        self.init_simulation()

        # self._pointer = PointMarker("/Visuals/ee_goal", 1, radius=0.1)
        self.robotiq = ArticulationView(
            prim_paths_expr="/World/robotiq/robotiq", name="robotiq")
        world.scene.add(self.robotiq)
        
        # self.mesh_prim = self.stage.GetPrimAtPath("/tofu/liquid")
        # self.sampling_api = PhysxSchema.PhysxParticleSamplingAPI(self.mesh_prim)
        # self.pointTargets = self.sampling_api.GetParticlesRel().GetTargets()
        # particlePath = self.pointTargets[0]
        # pointsPrim = self.stage.GetPrimAtPath(particlePath)
        # points = UsdGeom.Points(pointsPrim)
        
        world.reset()
        # points = UsdGeom.Points(pointsPrim)
        for _ in range(10):
            world.step(render=True)
        i=0
        while simulation_app.is_running():
            world.step(render=True)
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()

            # positions = points.GetPointsAttr().Get()
            i+=1
            # self._pointer.set_world_poses(positions=np.array([positions[0]]))
            if i == 10:
                # print("Number of total particles:", self.particle.get_number_of_points())
                # print("Number of Outside particles:", self.particle.get_number_of_outside_points())
                # print("Outside particles/Total particles:", self.particle.get_number_of_outside_points(),"/",self.particle.get_number_of_points())
                i = 0
            # # # ### PD position control##
            djv = self.robotiq.get_joint_positions(clone=False)
            # djv*= 0
            djv[:, :6] = 0
            djv[:, 6] = 0.5
            djv[:, 7] = djv[:, 6]
            djv[:, 9] = djv[:, 6]
            djv[:, 8] = -djv[:, 6]
            djv[:, 10] = -djv[:, 6]
            djv[:, 11] = -djv[:, 6]
            self.robotiq.set_joint_position_targets(djv)  # pd controls
 
        
if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
