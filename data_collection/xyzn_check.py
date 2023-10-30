import numpy as np
import open3d as o3d

import sys
import os
sys.setrecursionlimit(10**4) # RecursionError: maximum recursion depth exceeded while calling a Python object

current_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_directory, 'data', 'lemon')
pcd = o3d.io.read_point_cloud(path+"/lemon_pcds_0.xyzn", format='xyzn')

point_cloud_handle = o3d.geometry.PointCloud()
point_cloud_handle.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
point_cloud_handle.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(point_cloud_handle)

def time_gen():
    count = 0
    while True:
        count+=1
        yield count

time_ = time_gen()
def key_callback(vis):
    time = next(time_)
    print(str(time))
    pcd = o3d.io.read_point_cloud(path + "/lemon_pcds_"+str(time)+".xyzn", format='xyzn')
    point_cloud_handle.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    point_cloud_handle.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))
    vis.update_geometry(point_cloud_handle)
    vis.update_renderer()
    vis.poll_events()

vis.register_key_callback(65, key_callback)

vis.run()
vis.destroy_window()