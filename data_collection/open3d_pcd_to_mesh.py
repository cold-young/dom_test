# https://engineer-mole.tistory.com/248

#%%
import numpy as np
import open3d as o3d

#%%
# ## Mesh generation
# pcd = o3d.io.read_point_cloud('./my1.xyz')
# pcd.estimate_normals()

# # to obtain a consistent normal orientation
# pcd.orient_normals_towards_camera_location(pcd.get_center())

# # or you might want to flip the normals to make them point outward, not mandatory
# pcd.normals = o3d.utility.Vector3dVector( - np.asarray(pcd.normals))

# # surface reconstruction using Poisson reconstruction
# mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# # paint uniform color to better visualize, not mandatory
# mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))

# o3d.io.write_triangle_mesh('a.ply', mesh)

#%%
# IMPORT MESH
mesh = o3d.io.read_triangle_mesh("./a.ply")

# %%
# ## AVerage filter
# vertices = np.asarray(mesh.vertices)
# noise = 0.1
# vertices += np.random.uniform(0, noise, size=vertices.shape)
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])

# print('filter with average with 1 iteration')
# mesh_out = mesh.filter_smooth_simple(number_of_iterations=1)
# mesh_out.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_out])

# print('filter with average with 5 iterations')
# mesh_out = mesh.filter_smooth_simple(number_of_iterations=5)
# mesh_out.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_out])

#%%
# Laplacian

#%% 
# Taubin filter
# # Average & Laplacian Filter -> they lead to a shrinkage of the triangle mesh.
# print('filter with Taubin with 10 iterations')
# mesh_out = mesh.filter_smooth_taubin(number_of_iterations=10)
# mesh_out.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_out])

# print('filter with Taubin with 100 iterations')
# mesh_out = mesh.filter_smooth_taubin(number_of_iterations=100)
# mesh_out.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh_out])

#%% 
# SAMPLING
# pcd = mesh.sample_points_uniformly(number_of_points=5000)
# o3d.visualization.draw_geometries([pcd])

#%%
# Mesh subdivision
o3d.visualization.draw_geometries([mesh], point_show_normal=True, mesh_show_wireframe=True )
# o3d.visualization.draw_geometries([mesh], zoom=0.8, point_show_normal=True, mesh_show_wireframe=True)
mesh = mesh.subdivide_loop(number_of_iterations=2)
print(
    f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], point_show_normal=True, mesh_show_wireframe=True)

#%%
# Non-blocking visualization
# http://www.open3d.org/docs/0.8.0/tutorial/Advanced/non_blocking_visualization.html