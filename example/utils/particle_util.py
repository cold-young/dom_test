#
# Utils for collision detection


from omni.isaac.core.articulations.articulation_view import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.stage import open_stage, add_reference_to_stage, get_current_stage
from omni.physx import get_physx_scene_query_interface 
from omni.isaac.core.utils.prims import create_prim, delete_prim, get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform, tf_matrix_from_pose, pose_from_tf_matrix
from pxr import UsdGeom, Usd, Gf, PhysicsSchemaTools, Sdf, PhysxSchema

import numpy as np
import math
import trimesh as t
import open3d as o3d
import torch

class ParticleUtil():
    def __init__(
        self,
        stage,
        mesh_path,
        ):
        self._stage = stage
        self.mesh_path = mesh_path
        self.initalize_particles(mesh_path)
        return
    
    def initalize_particles(self, mesh_path):
        mesh_prim = self._stage.GetPrimAtPath(mesh_path)
        samplingApi = PhysxSchema.PhysxParticleSamplingAPI(mesh_prim)
        pointTargets = samplingApi.GetParticlesRel().GetTargets()

        particlePath = pointTargets[0]
        pointsPrim = self._stage.GetPrimAtPath(particlePath)
        self.points = UsdGeom.Points(pointsPrim)
        
    def get_number_of_points(self):
        positions = self.get_position_array()
        return len(positions)
    
    def get_position_array(self):
        positions = self.points.GetPointsAttr().Get()
        return positions
    
    def get_number_of_outside_points(self):
        trigger = UsdGeom.Mesh(get_current_stage().GetPrimAtPath("/tofu/checker"))
        trig_vert = np.array(trigger.GetPointsAttr().Get())
        ###
        vertices = trig_vert
        vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
        relative_tf_column_major = get_relative_transform(get_prim_at_path("/tofu/checker"), 
                                                        get_prim_at_path("/tofu"))
        relative_tf_row_major = np.transpose(relative_tf_column_major)

        points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
        points_in_meters = points_in_relative_coord[:, :-1]
        ###
        trig_bbox = t.PointCloud(points_in_meters).bounding_box
        # contact_check = np.array([item for item in trig_bbox.contains(self.get_position_array())])
        contact_check = trig_bbox.contains(self.get_position_array())
        
        return len(contact_check[contact_check == False])
        # from IPython import embed; embed()
    
    
    def get_oring_final_check(self, deform_points):
        """Need to fix for general case"""
        """get collision check between oring and trigger and set grasp"""
        self.coll_check = []
        self.oring_no_fall = []
        self.pole_check = []
        for i in range(self.num_envs):
              
            twist_trig_prim = get_current_stage().GetPrimAtPath(self.twist_trig_path[i])
            twist_trig_mesh = UsdGeom.Mesh(twist_trig_prim)
            twist_trig_local_collision_point = (np.array(twist_trig_mesh.GetPointsAttr().Get()))
            
            vertices = np.array(twist_trig_local_collision_point)
            vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
            relative_tf_column_major = get_relative_transform(get_prim_at_path(self.twist_trig_path[i]), 
                                                            get_prim_at_path("/World/envs/env_{}".format(i)))
            relative_tf_row_major = np.transpose(relative_tf_column_major)

            points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
            points_in_meters = points_in_relative_coord[:, :-1]


            # make bounding box
            twist_trig_pc = t.PointCloud(points_in_meters)
            twist_trig_boundingbox = twist_trig_pc.bounding_box


            goal_trig_prim = get_current_stage().GetPrimAtPath(self.pole_trig_path[i])
            goal_trig_mesh = UsdGeom.Mesh(goal_trig_prim)
            goal_trig_local_collision_point = (np.array(goal_trig_mesh.GetPointsAttr().Get()))
            
            vertices = np.array(goal_trig_local_collision_point)
            vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
            relative_tf_column_major = get_relative_transform(get_prim_at_path(self.pole_trig_path[i]), 
                                                            get_prim_at_path("/World/envs/env_{}".format(i)))
            relative_tf_row_major = np.transpose(relative_tf_column_major)

            points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
            points_in_meters = points_in_relative_coord[:, :-1]


            # make bounding box
            goal_trig_pc = t.PointCloud(points_in_meters)
            goal_trig_boundingbox = goal_trig_pc.bounding_box
            
            contact_check = not any(item == True for item in twist_trig_boundingbox.contains(deform_points[i]))
            self.coll_check.append(torch.as_tensor(contact_check))

            pole_check = any(item == True for item in goal_trig_boundingbox.contains(deform_points[i]))
            self.pole_check.append(torch.as_tensor(pole_check))
            
            # check oring center in trigger 
            raw_pcds_array = np.stack(deform_points[i])  # Convert list to numpy array
            # raw_pcds_tensor = torch.from_numpy(raw_pcds_array)  # Convert numpy array to tensor
            oring_mean_point = np.mean(raw_pcds_array, axis=0) 

            oring_inner_check = twist_trig_boundingbox.contains([oring_mean_point])
            self.oring_no_fall.append(torch.as_tensor(oring_inner_check))

        return torch.as_tensor(self.coll_check, dtype=float), torch.as_tensor(self.pole_check, dtype=float), torch.as_tensor(self.oring_no_fall, dtype=float).reshape(self.num_envs)