from collections import defaultdict
from typing import Dict, Optional, Tuple
import numpy as np
import open3d as o3d
from src.util.logger import get_logger

VoxelGridType = Dict[Tuple[int, int, int], Dict]


class VoxelGrid:
    """Manages voxelization and pixelation of point clouds."""
    
    def __init__(self, voxel_dict: VoxelGridType):
        self.grid = voxel_dict
    
    @staticmethod
    def from_point_cloud(point_cloud: o3d.geometry.PointCloud) -> "VoxelGrid":
        """Create voxel grid from snapped point cloud."""
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)
        colors = np.asarray(point_cloud.colors)

        voxel_map = defaultdict(list)

        for i, p in enumerate(points):
            base = np.floor(p).astype(int)
            voxel_map[tuple(base)].append(i)

            for dim in range(3):
                if np.isclose(p[dim] % 1, 0.0):
                    neighbor = base.copy()
                    neighbor[dim] -= 1
                    voxel_map[tuple(neighbor)].append(i)

        voxel_grid = {}

        for (x, y, z), indices in voxel_map.items():
            pts = points[indices]
            nmls = normals[indices]
            cols = colors[indices]

            center = np.array([x + 0.5, y + 0.5, z + 0.5])

            rel = pts - center

            px = (rel[:, 0] >= 0) & (nmls[:, 0] > 0.5)
            nx = (rel[:, 0] < 0) & (nmls[:, 0] < -0.5)

            py = (rel[:, 1] >= 0) & (nmls[:, 1] > 0.5)
            ny = (rel[:, 1] < 0) & (nmls[:, 1] < -0.5)

            pz = (rel[:, 2] >= 0) & (nmls[:, 2] > 0.5)
            nz = (rel[:, 2] < 0) & (nmls[:, 2] < -0.5)

            sub_pc = o3d.geometry.PointCloud()
            sub_pc.points = o3d.utility.Vector3dVector(pts)
            sub_pc.colors = o3d.utility.Vector3dVector(cols)
            sub_pc.normals = o3d.utility.Vector3dVector(nmls)

            voxel_grid[(x, y, z)] = {
                "center": center,
                "points": pts,
                "+x": sub_pc.select_by_index(np.nonzero(px)[0]),
                "-x": sub_pc.select_by_index(np.nonzero(nx)[0]),
                "+y": sub_pc.select_by_index(np.nonzero(py)[0]),
                "-y": sub_pc.select_by_index(np.nonzero(ny)[0]),
                "+z": sub_pc.select_by_index(np.nonzero(pz)[0]),
                "-z": sub_pc.select_by_index(np.nonzero(nz)[0]),
            }

        return VoxelGrid(voxel_grid)

    def pixelate_faces(
        self,
        resolution: Optional[int] = None,
        logger_instance=None
    ) -> None:
        """Pixelate all voxel faces."""
        resolution = resolution or 16
        logger_inst = logger_instance or get_logger(__name__)
        
        for (vx, vy, vz), voxel in self.grid.items():
            voxel_min = np.array([vx, vy, vz])
            for face_key in ["+x", "-x", "+y", "-y", "+z", "-z"]:
                color_grid = self._compute_face_color_grid(
                    voxel[face_key], voxel_min, face_key, resolution=resolution
                )
                self.grid[(vx, vy, vz)][f"{face_key}_color_grid"] = color_grid

    def _compute_face_color_grid(
        self,
        face_pcd: o3d.geometry.PointCloud,
        voxel_min: np.ndarray,
        face_type: str,
        resolution: int = 16
    ) -> Optional[np.ndarray]:
        """Project face points onto 2D grid and aggregate colors."""
        points = np.asarray(face_pcd.points)
        colors = np.asarray(face_pcd.colors)

        if len(points) == 0:
            return None

        if face_type in ["+x", "-x"]:
            u_vals, v_vals = points[:, 1] - voxel_min[1], points[:, 2] - voxel_min[2]
        elif face_type in ["+y", "-y"]:
            u_vals, v_vals = points[:, 0] - voxel_min[0], points[:, 2] - voxel_min[2]
        else:
            u_vals, v_vals = points[:, 0] - voxel_min[0], points[:, 1] - voxel_min[1]

        u_idx = np.clip((u_vals * resolution).astype(int), 0, resolution - 1)
        v_idx = np.clip((v_vals * resolution).astype(int), 0, resolution - 1)

        grid_colors = np.zeros((resolution, resolution, 3))
        grid_counts = np.zeros((resolution, resolution, 1))

        for i in range(len(points)):
            grid_colors[u_idx[i], v_idx[i]] += colors[i]
            grid_counts[u_idx[i], v_idx[i]] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_grid_colors = grid_colors / grid_counts

        return avg_grid_colors

    def get_voxel(self, x: int, y: int, z: int) -> Optional[Dict]:
        """Retrieve voxel data by coordinates."""
        return self.grid.get((x, y, z), None)
    
    def __len__(self):
        return len(self.grid)