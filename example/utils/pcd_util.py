# Pointcloud utils
# 23.10.12

import numpy as np
import math
import trimesh as t
import open3d as o3d
import torch
from pxr import UsdGeom, Usd, Gf, PhysicsSchemaTools, Sdf, PhysxSchema


class PointCloudUtil():
    def __init__(self):
        return
    
    def initalize_particles(self, mesh_path):
        mesh_prim = self._stage.GetPrimAtPath(mesh_path)
        samplingApi = PhysxSchema.PhysxParticleSamplingAPI(mesh_prim)
        pointTargets = samplingApi.GetParticlesRel().GetTargets()

        particlePath = pointTargets[0]
        pointsPrim = self._stage.GetPrimAtPath(particlePath)
        self.points = UsdGeom.Points(pointsPrim)

    def get_chamfer_distance(self, raw_pcds, target_pcds):
        """
        Check distance(chamfer distanse) between object node and target_node. 
        Args:

        INPUT
            object_node [num_envs*N*3](npy, np.array): 
            target_nodes [num_envs*N*3](npy, np.array): 
            
        OUTPUT
            chamfer_dist [num_envs*1](npy, np.array): each node's chamfer distances.    
        """
        chamfer_dists = []
        for i, raw_pcd in enumerate(raw_pcds):    
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(raw_pcd)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_pcds[i])
            # o3d.visualization.draw_geometries([object_pcd, target_pcd])

            cham_dist = np.asarray(object_pcd.compute_point_cloud_distance(target_pcd)).sum() # chamfer distance
            chamfer_dists.append(cham_dist)
 
        return chamfer_dists