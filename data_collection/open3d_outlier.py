#%%
import open3d as o3d
import numpy as np

#%%

## Load files
pcd = o3d.io.read_point_cloud("./my1.xyz", format='xyz')
#%%
## Visualize original pcd
print("render pcd")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#%% 
## Voxel down sample 

print("Downsample the point cloud with a voxel of 0.02")
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
o3d.visualization.draw_geometries([voxel_down_pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#%%
##  Uniform down sample
print("Every 5th points are selected")
uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
o3d.visualization.draw_geometries([uni_down_pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
# %%
## statistical outlier removal
downpcd_center = np.array(pcd.points).mean(axis=0)
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                    #   front=[0.4257, -0.2125, -0.8795],
                                    #   lookat=[2.6172, 2.0475, 1.532],
                                    #   up=[-0.0694, -0.9768, 0.2024])
                                                                    front = downpcd_center + np.array([0,-100,0]),
                                #   lookat=[2.6172, 2.0475, 1.532],
                                    lookat = downpcd_center,
                                #   up=[-0.0694, -0.9768, 0.2024],
                                up = downpcd_center)

print("Statistical oulier removal")
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10,
                                                    std_ratio=1)
display_inlier_outlier(pcd, ind)
# %%
## Radius outlier removal
print("Radius oulier removal")
cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.5)
display_inlier_outlier(pcd, ind)
# %%
