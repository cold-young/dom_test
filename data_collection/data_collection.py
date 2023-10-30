# Data collection module 
# This module is for collecting data for training srl models

# Chanyoung Ahn
# 23.10.30
# Can extract normal vector 
# Dont use headless (when we use headless, we can't get deformed object's pcd)
import argparse
import os

from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--data_num", type=int, default=10000, help="Number of data to collect")
parser.add_argument("--norm_pcd", action="store_true", default=False, help="Normalize pcd")
parser.add_argument("--object", type=str, default="lemon", help="Object types: tofu, lemon, strawberry, peach")
parser.add_argument("--gravity", action="store_true", default=True, help="Option of gravity on/off")
# parser.add_argument("--extract_indices", action="store_true", default=True, help="Save init indices of the object (Object_indices.npy)")
parser.add_argument("--vis_pcd", action="store_true", default=False, help="Visualize pcds of objects (Object.npy)")

args_cli = parser.parse_args()

config = {"headless": args_cli.headless}
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.render.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
# launch the simulator
simulation_app = SimulationApp(config, experience=app_experience)

"""Rest everything follows."""

if args_cli.headless:
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.isaac")
    enable_extension("omni.kit.window.toolbar")
    enable_extension("omni.kit.viewport.rtx")
    enable_extension("omni.kit.viewport.pxr")
    enable_extension("omni.kit.viewport.bundle")
    enable_extension("omni.kit.window.status_bar")

from omni.isaac.core.physics_context.physics_context import PhysicsContext
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
from omni.usd.commands import  DeletePrimsCommand
import omni.usd
from pxr import UsdGeom, PhysxSchema, Gf, Usd, UsdShade
import open3d as o3d
import trimesh as tr
import numpy as np
import torch 
import random
from tqdm import tqdm

class DataCollection():
    def __init__(self):
        self.current_directory = os.path.dirname(os.path.abspath(__file__))

    def init_simulation(self):
        # use gpu_pipline=False
        # usd gpu=True
        # device = cpu
        # use_flat_cache=False
        self._scene = PhysicsContext(sim_params={"use_gpu_pipeline": False, 
                                                 "use_gpu": True, 
                                                 "device": "cpu", 
                                                 "use_flatcache": False})
        
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        # self._scene.enable_ccd(flag=True)
        # self._scene.enable_flatcache(False)
        
        if args_cli.gravity:
            self._scene.set_gravity(value=-9.8)
        else:
            self._scene.set_gravity(value=0.0)
        
        # setting init stage 
        prim_path = '/World/ground'
        prim_utils.create_prim(prim_path, 
                        usd_path=os.path.join(self.current_directory, 'usd', 'grid_ground.usd'), 
                        translation=(0.0, 0.0, 0.0))
    
    def create_object(self, rand_ori=True):
        # Set Object
        if rand_ori:
            if args_cli.gravity:
                prim_utils.create_prim("/World/Object",
                    usd_path=os.path.join(self.current_directory, 'usd','foods', f'{args_cli.object}.usd'), 
                    translation=(0.0, 0.0, 1.0),
                    orientation=self.random_yaw_orientation(1, "cpu").squeeze().tolist())
                self.deformable_body = PhysxSchema.PhysxDeformableBodyAPI(get_prim_at_path("/World/Object/mesh"))
            else:
                prim_utils.create_prim("/World/Object",
                            usd_path=os.path.join(self.current_directory, 'usd','foods', f'{args_cli.object}.usd'), 
                            translation=(0.0, 0.0, 3.0),
                            orientation=self.random_orientation(1, "cpu").squeeze().tolist())
                self.deformable_body = PhysxSchema.PhysxDeformableBodyAPI(get_prim_at_path("/World/Object/mesh"))
        else:
            if args_cli.gravity:
                prim_utils.create_prim("/World/Object",
                            usd_path=os.path.join(self.current_directory, 'usd','foods', f'{args_cli.object}.usd'), 
                            translation=(0.0, 0.0, 1.0),
                            orientation=(0.0, 0.0, 0.0, 1.0))
                self.deformable_body = PhysxSchema.PhysxDeformableBodyAPI(get_prim_at_path("/World/Object/mesh"))
            else:
                prim_utils.create_prim("/World/Object",
                            usd_path=os.path.join(self.current_directory, 'usd','foods', f'{args_cli.object}.usd'), 
                            translation=(0.0, 0.0, 3.0),
                            orientation=(0.0, 0.0, 0.0, 1.0))
                self.deformable_body = PhysxSchema.PhysxDeformableBodyAPI(get_prim_at_path("/World/Object/mesh"))
                     
    def create_gripper(self):
        # Set gripper
        if args_cli.gravity:
            if args_cli.object == 'tofu':
                prim_utils.create_prim("/World/Robot",
                    usd_path=os.path.join(self.current_directory, 'usd', 'robotiq.usd'), 
                    translation=(0.0, np.round(random.uniform(-0.35, 0.35), 4), np.round(random.uniform(3.25, 3.4), 4)))  
            else: 
                prim_utils.create_prim("/World/Robot",
                    usd_path=os.path.join(self.current_directory, 'usd', 'robotiq.usd'), 
                    translation=(0.0, np.round(random.uniform(-0.35, 0.35), 4), np.round(random.uniform(3.35, 3.7), 4))) 
        else:
            prim_utils.create_prim("/World/Robot",
                        usd_path=os.path.join(self.current_directory, 'usd', 'robotiq.usd'), 
                        translation=(0.0, np.round(random.uniform(-0.35, 0.35), 4), np.round(random.uniform(5.5, 5.8), 4)))
        
        self.gripper = ArticulationView(prim_paths_expr="/World/Robot/robotiq")

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

    def create_pcd(self, pcd_num:int, pcd_path:str="/World/vis_pcd"):
        color = (1,0,0)
        size = 0.2

        stage = omni.usd.get_context().get_stage()
        point_list = np.zeros([pcd_num,3])
        sizes = size * np.ones(pcd_num)
        stage = omni.usd.get_context().get_stage()
        points = UsdGeom.Points.Define(stage, pcd_path)
        points.CreatePointsAttr().Set(point_list)
        points.CreateWidthsAttr().Set(sizes)
        points.CreateDisplayColorPrimvar("constant").Set([color])
        return points
    
    def _convert_poly_to_tri(self,
        vertices: np.ndarray, faces_indices: np.ndarray, face_vertex_counts: np.ndarray
    ) -> np.ndarray:
        """Converts the input mesh into a triangle mesh.

        Args:
            vertices: A 2D array of shape (V, 3) containing the vertices that define the mesh.
            faces_indices: A 1D array representing the indices of the vertices for the corresponding faces.
            face_vertex_counts: A 1D array containing the number of vertices defined for each face.

        Returns:
            An array containing the faces of the triangle mesh.
        """
        mask = face_vertex_counts > 3
        faces = np.empty(
            (face_vertex_counts.shape[0] + np.sum(face_vertex_counts[np.nonzero(mask)]) - 3 * np.sum(mask), 3),
            dtype=np.int64,
        )
        faces_idx = 0
        poly_faces_idx = 0

        for vertex_count in face_vertex_counts:
            if vertex_count == 3:
                faces[faces_idx, :] = faces_indices[poly_faces_idx : poly_faces_idx + 3]
            else:  # if face is not a triangle, then break it up into multiple triangles
                faces[faces_idx : faces_idx + vertex_count - 2, 0] = faces_indices[poly_faces_idx]
                # sub-divide the polygon into several triangles by creating lines from the first vertex
                for i in range(poly_faces_idx, poly_faces_idx + vertex_count - 2):
                    faces[faces_idx + i - poly_faces_idx, 1:] = faces_indices[i + 1 : i + 3]
            faces_idx += vertex_count - 2
            poly_faces_idx += vertex_count

        return faces


    def get_deform_point(self, normalize: bool = False):
        # local_collision_point = np.array(self.deformable_body.GetCollisionPointsAttr().Get())
        local_collision_point = np.array(get_prim_at_path("/World/Object/mesh").GetAttribute("points").Get())   
        vertices = np.array(local_collision_point)
        vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
        relative_tf_column_major = get_relative_transform(get_prim_at_path("/World/Object/mesh"), 
                                                          get_prim_at_path("/World"))
        relative_tf_row_major = np.transpose(relative_tf_column_major)
        points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
        pcd = points_in_relative_coord[:, :-1]
        
        face_idx = np.array(get_prim_at_path("/World/Object/mesh").GetAttribute("faceVertexIndices").Get())
        face_vertex_counts = np.array(get_prim_at_path("/World/Object/mesh").GetAttribute("faceVertexCounts").Get())
        
        # convert prim mesh to a triangle mesh
        tris = self._convert_poly_to_tri(pcd, face_idx, face_vertex_counts)
        # get normals
        v1 = pcd[tris[:, 0]] - pcd[tris[:, 1]]
        v2 = pcd[tris[:, 0]] - pcd[tris[:, 2]]
        surface_normals = np.cross(v1, v2)
        areas = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=-1)
        surface_normals /= 2 * areas[:, None]
        
        a = tr.Trimesh(vertices=pcd, faces=tris, face_normals=surface_normals)
        
        pcd = a.vertices
        normal = a.vertex_normals
        
        if normalize:
            mins = np.min(pcd, axis=0)
            maxs = np.max(pcd, axis=0)
            scale_factor = np.max(maxs - mins)/2
            mean = np.mean(pcd, axis=0)
            norm_pcd = (pcd - mean)/scale_factor
            return norm_pcd, normal
        else:
            """only raw deform point cloud"""
            return pcd, normal
    
    def main(self):
        print(f"Data Collection num : {args_cli.data_num}, object : {args_cli.object}, gravity : {args_cli.gravity}")
        self.init_simulation()
        world = World(stage_units_in_meters=1, 
                      set_defaults=False)
        set_camera_view(eye=[0, 9, 10], target=[0.01, 0.01, 0.01], camera_prim_path="/OmniverseKit_Persp")
        
        self.stage = world.stage
        
        # Create object
        self.create_object(rand_ori=False)
        # Create gripper
        self.create_gripper()
        if not os.path.exists(os.path.join(self.current_directory, 'data', f'{args_cli.object}')):
            os.makedirs(os.path.join(self.current_directory, 'data', f'{args_cli.object}'))
        
        world.scene.add(self.gripper)

        if args_cli.vis_pcd:
            pcd_num = len(np.array(self.deformable_body.GetCollisionPointsAttr().Get()))
            points = self.create_pcd(pcd_num)
            
        world.reset()
        if args_cli.norm_pcd:
            pcd, normal = self.get_deform_point(normalize=True)
        else:
            pcd, normal = self.get_deform_point(normalize=False)

        i = 0
        collected_data = 0
        
        dataset = np.empty((args_cli.data_num, len(pcd), 6))
        while args_cli.data_num > collected_data:
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            i += 1
            print(f"Data collection : {collected_data} / {args_cli.data_num}")
            if i == 130:
                world.scene.clear()
                DeletePrimsCommand(paths=["/World/Object"], destructive=True).do() 

                self.create_object(rand_ori=True)
                self.create_gripper()
                world.scene.add(self.gripper)
                world.reset()    
                i = 0
            
            # if i == 100:
            #     # check using open3d
            #     face_idx = np.array(get_prim_at_path("/World/Object/mesh").GetAttribute("faceVertexIndices").Get())
            #     face_vertex_counts = np.array(get_prim_at_path("/World/Object/mesh").GetAttribute("faceVertexCounts").Get())
                
            #     # convert prim mesh to a triangle mesh
            #     tris = self._convert_poly_to_tri(pcd, face_idx, face_vertex_counts)
                
            #     a = tr.Trimesh(vertices=pcd, faces=tris, face_normals=normal)
            #     _pcd = o3d.geometry.PointCloud()
            #     _pcd.points = o3d.utility.Vector3dVector(a.vertices)
            #     _pcd.normals = o3d.utility.Vector3dVector(a.vertex_normals)
            #     o3d.visualization.draw_geometries([_pcd], point_show_normal=True)
            #     print("test")
                
            init_position = self.gripper.get_joint_positions()
            init_position[:, :6] = 0.
            init_position[:, 6] += 0.01
            init_position[:, 7] += 0.01
            init_position[:, 9] += 0.01
            init_position[:, 8] -= 0.01
            init_position[:, 10] -= 0.01
            init_position[:, 11] -= 0.01
            self.gripper.set_joint_position_targets(init_position)
            
            world.step(render=True)
            
            if args_cli.norm_pcd:
                pcd, normal = self.get_deform_point(normalize=True)
            else:
                pcd, normal = self.get_deform_point(normalize=False)

            dataset[collected_data] = np.hstack([pcd, normal])
            collected_data += 1
            
            if args_cli.vis_pcd:
                points.GetPointsAttr().Set(pcd)
        
        if args_cli.norm_pcd:
            for i in range(dataset.shape[0]):
                # np.save(os.path.join(self.current_directory, 'data', f'{args_cli.object}', f'{args_cli.object}_pcds_norm_{i}.npy'), dataset[i,:,:])
                np.savetxt(os.path.join(self.current_directory, 'data', f'{args_cli.object}', f'{args_cli.object}_pcds_norm_{i}.xyzn'), dataset[i,:,:])
        else:
            for i in tqdm(range(dataset.shape[0])):
                # np.save(os.path.join(self.current_directory, 'data', f'{args_cli.object}', f'{args_cli.object}_pcds_{i}.npy'), dataset[i,:,:])
                np.savetxt(os.path.join(self.current_directory, 'data', f'{args_cli.object}', f'{args_cli.object}_pcds_{i}.xyzn'), dataset[i,:,:])
                
        
        print(f"Data Collection done! num : {args_cli.data_num}, object : {args_cli.object}, gravity : {args_cli.gravity}")


if __name__ == "__main__":
    try:
        test = DataCollection()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()