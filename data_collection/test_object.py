###
# 23.7.21
# Grasp pose stability prediction model with groud truth pcds and tactile sensor
# To adjust parameter for generate variable "depth" pose
###

import argparse
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--object", type=str, default="mm_4", help="Object types: tofu, lemon, strawberry, peach, mm_1")
parser.add_argument("--data_num", type=int, default=10, help="Number of data to collect")
parser.add_argument("--rand_ori", action="store_true", default=False, help="Randomization when creating object (orientation")

args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
import torch
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.prims import RigidPrimView
from pxr import UsdGeom, PhysxSchema, Gf, Usd
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.transformations import get_relative_transform, tf_matrix_from_pose, pose_from_tf_matrix

from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.prims as prim_utils
from omni.usd.commands import DeletePrimsCommand
from omni.isaac.core.articulations import ArticulationView

import random
import datetime
from tqdm import tqdm

class Test():
    def __init__(self):
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_num = args_cli.data_num

    def init_simulation(self):
        self._scene = PhysicsContext(sim_params={"use_gpu_pipeline": False, 
                                                 "use_gpu": True, 
                                                 "device": "cpu", 
                                                 "use_flatcache": False})
        
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.set_gravity(value=-9.8)
        
        # setting init stage 
        prim_path = '/World/ground'
        prim_utils.create_prim(prim_path, 
                        usd_path=os.path.join(self.current_directory, 'usd', 'grid_ground.usd'), 
                        translation=(0.0, 0.0, 0.0))
    
    def create_object(self, rand_ori=True):
        # Set Object
        if rand_ori:
            prim_utils.create_prim("/World/Object",
                usd_path=os.path.join(self.current_directory, 'usd','foods', f'{args_cli.object}.usd'), 
                translation=(0.0, 0.0, 1.0),
                orientation=self.random_yaw_orientation(1, "cpu").squeeze().tolist())
            self.deformable_body = PhysxSchema.PhysxDeformableBodyAPI(get_prim_at_path("/World/Object/mesh"))
        else: 
            prim_utils.create_prim("/World/Object",
                        usd_path=os.path.join(self.current_directory, 'usd','foods', f'{args_cli.object}.usd'), 
                        translation=(0.0, 0.0, 1.0))
                        # orientation=(0.0, 0.0, 0.0, 1.0))
            self.deformable_body = PhysxSchema.PhysxDeformableBodyAPI(get_prim_at_path("/World/Object/mesh"))

    def create_gripper(self, i):
        
        # Set gripper
        prim_utils.create_prim("/World/Robot",
            usd_path=os.path.join(self.current_directory, 'usd', 'robotiq.usd'), 
            translation=(0.0, i, np.round(random.uniform(3.25, 3.4), 4)))  

        self.gripper = ArticulationView(prim_paths_expr="/World/Robot/robotiq")

    def get_test_poses(self):
        slice_poses = np.around(np.linspace(-1., 1.2, self.data_num),4)
        a = torch.tensor([0, 0, 2.2]).expand(size=[len(slice_poses),3]).clone()
        for i, x in enumerate(slice_poses):
            a[i,1] = x
        return a

    def random_yaw_orientation(self, num: int, device: str) -> torch.Tensor:
        roll = torch.zeros(num, dtype=torch.float, device=device)
        pitch = torch.zeros(num, dtype=torch.float, device=device)
        yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
        return self.quat_from_euler_xyz(roll, pitch, yaw)

    def random_orientation(self, num: int, device: str) -> torch.Tensor:
        quat = torch.randn((num, 4), dtype=torch.float, device=device)
        return torch.nn.functional.normalize(quat, p=2.0, dim=-1, eps=1e-12)
    
    def quat_from_euler_xyz(self, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        # compute quaternion
        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp
        return torch.stack([qw, qx, qy, qz], dim=-1)

    def main(self):
        self.init_simulation()
        world = World(stage_units_in_meters=1, 
                      set_defaults=False)
        set_camera_view(eye=[0, 9, 10], target=[0.01, 0.01, 0.01], camera_prim_path="/OmniverseKit_Persp")
        self.stage = world.stage
        # Create object
        self.create_object(rand_ori=False)
        # Create gripper
        self.create_gripper(i=0.0)
                    
        world.scene.add(self.gripper)
        world.reset()

        with torch.no_grad():
            # select_grasps = []
            world.step()
            # Data collection 
            positions = self.get_test_poses()

            for position in tqdm(positions):
                world.scene.clear()
                DeletePrimsCommand(paths=["/World/Object"], destructive=True).do() 
                if args_cli.rand_ori:
                    self.create_object(rand_ori=True)
                else: 
                    self.create_object(rand_ori=False)
                self.create_gripper(i = position[1])
                world.scene.add(self.gripper)
                world.reset()    
                
                for i in range(130):
                    init_position = self.gripper.get_joint_positions()

                    if i > 50:
                        init_position[:, :2] = 0.
                        init_position[:, 3:6] = 0.
                        init_position[:, 2] += 0.05                        
                        init_position[:, 6] += 0.01
                        init_position[:, 7] += 0.01
                        init_position[:, 9] += 0.01
                        init_position[:, 8] -= 0.01
                        init_position[:, 10] -= 0.01
                        init_position[:, 11] -= 0.01
                        self.gripper.set_joint_position_targets(init_position)
                    else:
                        init_position[:, :6] = 0.
                        init_position[:, 6] += 0.01
                        init_position[:, 7] += 0.01
                        init_position[:, 9] += 0.01
                        init_position[:, 8] -= 0.01
                        init_position[:, 10] -= 0.01
                        init_position[:, 11] -= 0.01
                        self.gripper.set_joint_position_targets(init_position)
                    world.step(render=True) 
                        
                

    
if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
