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
# from deftouchnet.utils.soft_prim_view import SoftBodyView

from omni.physx.scripts.physicsUtils import *

from omni.isaac.core.physics_context.physics_context import PhysicsContext
import omni.usd
 

# from omni.physx.scripts import deformableUtils, physicsUtils
import omni.kit.commands
from omni.isaac.sensor import _sensor
from omni.isaac.core.utils.stage import open_stage

# import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt

# from pxr import UsdGeom, Sdf, Gf, Vt, PhysicsSchemaTools    

class Test():
    def __init__(self):
        self._device = "cuda:0"        
        # self.gripper_close = False
        self._path = os.getcwd()
        self.obs = None
 
        
    def init_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.set_friction_offset_threshold(0.01)
        self._scene.set_friction_correlation_distance(0.005)
        self._scene.enable_ccd(flag=False)

    def update(self):
        # 새로운 z값 생성
        scale = 100
        reading1 = self.object.get_net_contact_forces() *scale
        reading2 = self.object2.get_net_contact_forces() *scale
        reading3 = self.object3.get_net_contact_forces() *scale
        reading4 = self.object4.get_net_contact_forces()*scale

        x_force = []
        y_force = []
        z_force = []
        # from IPython import embed; embed()
        
        for i in range(4):
            a = 'reading'+str(i+1)
            var = eval(a)
            x_force.append(float(var[:,0]*0.01))
            y_force.append(float(var[:,1]*0.01))
            z_force.append(float(var[:,2]))
        x_force, y_force, z_force = np.array(x_force), np.array(y_force), np.array(z_force)
        
        print( "##################Normal Force################# \n"
                "{:.2f}".format(float(reading1[:,0])),"{:.2f}".format(float(reading2[:,0])),"\n",
                "{:.2f}".format(float(reading3[:,0])),"{:.2f}".format(float(reading4[:,0])),"\n",
                )
        # print(reading1,"\n",
        #       reading2)

#########################################33
        x_new = np.array(self.x_list) - x_force
        y_new = np.array(self.y_list) - y_force
        # z_force = np.abs(z_force)*100 + 10
        z_force = z_force*100 + 10
        
        
        self.scatter.set_offsets(np.c_[x_new, y_new])
        # from IPython import embed; embed(sss);exit()
        self.scatter.set_sizes(sizes=z_force)
        # self.scatter = self.ax.scatter(x_new, y_new, s=z*10)  
        
    def update_raw(self):
        # 새로운 z값 생성
        scale = 100
        reading1 = self.object.get_net_contact_forces() *scale
        reading2 = self.object2.get_net_contact_forces() *scale
        reading3 = self.object3.get_net_contact_forces() *scale
        reading4 = self.object4.get_net_contact_forces()*scale
        reading5 = self.object5.get_net_contact_forces() *scale
        reading6 = self.object6.get_net_contact_forces()*scale
        
        print( "################## Normal Force ################# \n"
                "{:.2f}".format(abs(float(reading1[:,0]))),"{:.2f}".format(abs(float(reading2[:,0]))),"\n",
                "{:.2f}".format(abs(float(reading3[:,0]))),"{:.2f}".format(abs(float(reading4[:,0]))),"\n",
                "{:.2f}".format(abs(float(reading5[:,0]))),"{:.2f}".format(abs(float(reading6[:,0]))),"\n",
                )

    def main(self):        
        self._path = os.getcwd()
        self.asset_path = self._path + "/dom_test/example/assets/" 
        object_usd_path = self.asset_path + "test_scene_2.usd"
        open_stage(usd_path=object_usd_path)
        
        world = World(stage_units_in_meters=1)
        world._physics_context.enable_gpu_dynamics(flag=True)
        self.stage = world.stage
        self.init_simulation()

        
        # world.reset()
        self.object = RigidPrimView(prim_paths_expr="/grid_ground/tactile_robotiq/tactile/sensors_2/sensor_2_0000", name="object_view",
                                    track_contact_forces=True)
        self.object2 = RigidPrimView(prim_paths_expr="/grid_ground/tactile_robotiq/tactile/sensors_2/sensor_2_0001", name="object_view",
                                    track_contact_forces=True)
        self.object3 = RigidPrimView(prim_paths_expr="/grid_ground/tactile_robotiq/tactile/sensors_2/sensor_2_0002", name="object_view",
                                    track_contact_forces=True)
        self.object4 = RigidPrimView(prim_paths_expr="/grid_ground/tactile_robotiq/tactile/sensors_2/sensor_2_0003", name="object_view",
                                    track_contact_forces=True)
        self.object5 = RigidPrimView(prim_paths_expr="/grid_ground/tactile_robotiq/tactile/sensors_2/sensor_2_0004", name="object_view",
                                    track_contact_forces=True)
        self.object6 = RigidPrimView(prim_paths_expr="/grid_ground/tactile_robotiq/tactile/sensors_2/sensor_2_0005", name="object_view",
                                    track_contact_forces=True)        

        world.reset()
 
        # ani = animation.FuncAnimation(self.fig, self.update, frames=1, interval=100)
        
        self.object.initialize()
        self.object2.initialize()
        self.object3.initialize()
        self.object4.initialize()
        self.object5.initialize()
        self.object6.initialize()

        # self.set_vis()
        # self.get_vis()

        while simulation_app.is_running():
            world.step(render=True)
 
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            self.update_raw()
            
            # self.fig.canvas.draw()  # 그래프 업데이트
            # self.fig.canvas.flush_events()
            # djv = self.robotiq.get_joint_velocities(clone=False)
            # djv *= 0.0
            # djv[:, 6] = 5
            # djv[:, 7] = djv[:, 6]
            # djv[:, 9] = djv[:, 6]
            # djv[:, 8] = -djv[:, 6]
            # djv[:, 10] = -djv[:, 6]
            # djv[:, 11] = -djv[:, 6]
            # # tjv = self.robotiq.get_joint_velocities(clone=False).clone()
            
        
if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
