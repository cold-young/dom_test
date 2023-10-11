import torch
import numpy as np

import carb
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.scenes.scene_registry import SceneRegistry
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.utils.prims import (
    is_prim_path_valid,
)
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.prims.xform_prim import XFormPrim
import os

class SceneCustom(object):
    def __init__(self) -> None:
        self._scene_registry = SceneRegistry()
        self.scene = Scene()

        self._path = os.getcwd()
        self.asset_path = self._path + "/example/assets" 
        # define_prim("/World/envs/env_0")
        return
    
    def init_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.set_friction_offset_threshold(0.01)
        self._scene.set_friction_correlation_distance(0.005)
        self._scene.enable_ccd(flag=False)
        # from IPython import embed; embed(); exit()
        
        # from Orbit env cfg
        self._scene.set_gpu_total_aggregate_pairs_capacity(1024 * 1024 * 2) #PxgDynamicsMemoryConfig:
        self._scene.set_gpu_max_rigid_contact_count(1024 * 1024 * 2)
        self._scene.set_gpu_max_rigid_patch_count(80 * 1024 * 2 * 2)
        self._scene.set_gpu_found_lost_pairs_capacity(1024 * 1024 * 2)
        self._scene.set_gpu_found_lost_aggregate_pairs_capacity(1024 * 1024 * 32)
        self._scene.set_gpu_max_soft_body_contacts(1024 * 1024)
        self._scene.set_gpu_heap_capacity(64 * 1024 * 1024)
        self._scene.set_gpu_temp_buffer_capacity(16 * 1024 * 1024)
        self._scene.set_gpu_max_num_partitions(8)
        
        self._scene.set_physics_dt(dt=1/100, substeps=20)

    # gpu_max_rigid_contact_count: 524288
    # gpu_max_rigid_patch_count: 81920
    # gpu_found_lost_pairs_capacity: 8192
    # gpu_found_lost_aggregate_pairs_capacity: 262144
    # gpu_total_aggregate_pairs_capacity: 8192
    # gpu_max_soft_body_contacts: 1048576
    # gpu_max_particle_contacts: 1048576
    # gpu_heap_capacity: 67108864
    # gpu_temp_buffer_capacity: 16777216
    # gpu_max_num_partitions: 8

    def add_default_ground_plane(
        self,
        z_position: float = 0,
        name="default_ground_plane",
        prim_path: str = "/World/defaultGroundPlane",
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.8,
        usd_path: str = None,
    ) -> None:
        """[summary]

        Args:
            z_position (float, optional): [description]. Defaults to 0.
            name (str, optional): [description]. Defaults to "default_ground_plane".
            prim_path (str, optional): [description]. Defaults to "/World/defaultGroundPlane".
            static_friction (float, optional): [description]. Defaults to 0.5.
            dynamic_friction (float, optional): [description]. Defaults to 0.5.
            restitution (float, optional): [description]. Defaults to 0.8.
            usd_path (str, optional): [description]. defaults is simple version. if you want wall version, use "vis"

        Returns:
            [type]: [description]
        """
        if self.scene.object_exists(name=name):
            carb.log_info("ground floor already created with name {}.".format(name))
            return self.scene.get_object(self, name=name)
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
        if usd_path is None:
            usd_path = self.asset_path + "/environment/default_environment_simple.usd"
        elif usd_path == "vis":
            usd_path = self.asset_path + "/environment/default_environment.usd"
            # usd_path = "omniverse://localhost/Users/chanyoung/Asset/sensor/grid/default_environment.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        physics_material_path = find_unique_string_name(
            initial_name="/World/Physics_Materials/physics_material", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        physics_material = PhysicsMaterial(
            prim_path=physics_material_path,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        plane = GroundPlane(prim_path=prim_path, name=name, z_position=z_position, physics_material=physics_material)
        self.scene.add(plane)
        return plane

    def set_camera_init_view(self, eye, target):
        if eye is None:
            eye = [-0.7185, 16.68674, 13.17014]
        if target is None: 
            target=[0.01, 0.01, 0.01]
            
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
 
    def add(self, obj: XFormPrim) -> XFormPrim:
        return self.scene.add(obj)