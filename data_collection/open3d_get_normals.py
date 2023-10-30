# http://www.open3d.org/docs/latest/tutorial/geometry/surface_reconstruction.html#Ball-pivoting

#%%

import open3d as o3d
import numpy as np
# import matplotlib.pyplot as plt

front = [-1,0,0]
lookat = [0,0,0]
up = [0, 0, 1]
#%%
print("Testing mesh in Open3D...")
armadillo_mesh = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)

knot_mesh = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))

#%%
## Visualize PC

print("Load a ply point cloud, print it, and render it")
# ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud("./my.xyz", format='xyz')
print(pcd)
print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=front,
#                                   lookat=lookat,
#                                   up=up)

# %%

## Vortex downsampling
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
# o3d.visualization.draw_geometries([downpcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])
# %%

## Vortex normal estimation
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

# o3d.visualization.draw_geometries([downpcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024],
#                                   point_show_normal=True)
                                  
# %%
## Access estimated vertex normal

# Get object Center
downpcd_points = np.array(downpcd.points) # N X 3 array
downpcd_center = np.array(downpcd.points).mean(axis=0)
downpcd_vector_from_center_to_point = downpcd_points - downpcd_center

downpcd_vectors = np.asarray(downpcd.normals)
# np.asarray(downpcd.normals) * 
# downpcd.normals = o3d.utility.Vector3dVector(downpcd_vector_from_center_to_point)
# downpcd.orient_normals_to_align_with_direction()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

angles = np.array([angle_between(v1,v2) for (v1,v2) in zip(downpcd_vectors,downpcd_vector_from_center_to_point )])
downpcd_vectors[np.where(angles>np.pi/2),:] = -downpcd_vectors[np.where(angles>np.pi/2),:]
downpcd.normals = o3d.utility.Vector3dVector(downpcd_vectors)
# downpcd.normals = o3d.utility.Vector3dVector(i for i np.array(downpcd.normals))
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                #   front=[0.4257, -0.2125, -0.8795],
                                front = downpcd_center + np.array([0,-100,0]),
                                #   lookat=[2.6172, 2.0475, 1.532],
                                    lookat = downpcd_center,
                                #   up=[-0.0694, -0.9768, 0.2024],
                                up = downpcd_center,
                                  point_show_normal=True)
#%% 
## Save files
o3d.io.write_point_cloud(filename="./with_normals.xyzn", pointcloud=downpcd)
