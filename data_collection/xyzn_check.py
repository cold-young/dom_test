import numpy as np
import open3d as o3d

import sys
sys.setrecursionlimit(10**4) # RecursionError: maximum recursion depth exceeded while calling a Python object

data = np.load('./Oring.npy')
index = np.load('./Oring_Indices.npy')

(TIMELENGTH, NUMPOINTS, _) = data.shape

pcd = o3d.io.read_point_cloud("./oring/0.xyzn", format='xyzn')

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
    pcd = o3d.io.read_point_cloud("./oring/"+str(time)+".xyzn", format='xyzn')
    point_cloud_handle.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    point_cloud_handle.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))
    vis.update_geometry(point_cloud_handle)
    vis.update_renderer()
    vis.poll_events()

vis.register_key_callback(65, key_callback)

vis.run()
vis.destroy_window()