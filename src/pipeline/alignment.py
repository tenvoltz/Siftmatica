from typing import Dict, Optional, Tuple
import numpy as np
import open3d as o3d
from src.util.geometry import normalize_vector
from src.util.logger import get_logger
from src.config import AlignmentConfig
from src.pipeline.plane_analyzer import PlaneAnalyzer, PlaneModel
from src.pipeline.color_gradient_analyzer import ColorGradientAnalyzer
from src.pipeline.voxel_grid import VoxelGrid, VoxelGridType

logger = get_logger(__name__)


class PointCloudAlignment:
    """Orchestrates point cloud alignment to canonical grid reference frame."""

    def __init__(self, config: Optional[AlignmentConfig] = None, logger_instance=None):
        self.config = config or AlignmentConfig()
        self.logger = logger_instance or get_logger(__name__)
        self.plane_analyzer = PlaneAnalyzer(self.config, self.logger)
        self.gradient_analyzer = ColorGradientAnalyzer(self.config, self.logger)

    def align_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> VoxelGridType:
        """Fully align point cloud to canonical grid reference frame."""
        # self._draw_point_cloud(point_cloud, title="Original Point Cloud", plot_axis=False, plot_grid=False)
        # pc = self._downsample_point_cloud(point_cloud, voxel_size=0.1)
        # self._draw_point_cloud(pc, title="Downsampled Point Cloud", plot_axis=False, plot_grid=False)
        pc = self._prepare_point_cloud(point_cloud)
        # self._draw_point_cloud(pc, title="Prepared Point Cloud", plot_axis=False, plot_grid=False)
        pc, planes = self._align_to_axes(pc)
        # self.plane_analyzer._draw_planes(pc, planes, title="Detected Planes")
        # self._draw_point_cloud(pc, title="Aligned Point Cloud", plot_axis=False, plot_grid=True)
        scale, phase = self._estimate_grid_parameters(pc)
        pc = self._conform_to_grid_space(pc, scale, phase)
        # self._draw_point_cloud(pc, title="Conformed Point Cloud", plot_axis=False, plot_grid=True)
        voxel_grid = self._snap_and_voxelize(pc)
        self._plot_voxel_grid(pc, voxel_grid)
        return voxel_grid

    def _prepare_point_cloud(self, pc: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Flip vertically and estimate surface normals."""
        pc = self._flip_vertical(pc)
        pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return self._estimate_normals(pc)

    def _align_to_axes(self, pc: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Detect planes and align to estimated axes."""
        # downsample_pc = self._downsample_point_cloud(pc, voxel_size=0.2)
        planes = self.plane_analyzer._detect_planes(pc)
        axes, _ = self.plane_analyzer.estimate_axes(planes)
        # self._draw_point_cloud(pc, title="Point Cloud with Detected Axes", plot_axis=True, estimated_axes=axes.T, plot_grid=True)
        return self._align_point_cloud_to_axes(pc, axes), planes

    def _estimate_grid_parameters(self, pc: o3d.geometry.PointCloud) -> Tuple[float, Dict[str, float]]:
        """Estimate scale and phase from color gradient and plane positions."""
        planes = self.plane_analyzer._detect_planes(pc)
        _, axical_plane_models = self.plane_analyzer.estimate_axes(planes)
        # self.plane_analyzer._draw_planes(pc, axical_plane_models, title="Axial Planes for Phase Estimation")
        estimated_phase = self._estimate_phase_from_plane(axical_plane_models)
        estimated_scale = self.gradient_analyzer.estimate_scale(pc, plot=True)

        return estimated_scale, estimated_phase

    def _conform_to_grid_space(
        self,
        point_cloud: o3d.geometry.PointCloud,
        estimated_scale: float,
        estimated_phase: Dict[str, float]
    ) -> o3d.geometry.PointCloud:
        """Scale and translate to conform to grid space."""
        self.logger.trace(
            "_conform_to_grid_space",
            "Conforming point cloud to estimated grid..."
        )
        translation_vector = np.array([estimated_phase.get(axis, 0) for axis in ("x", "y", "z")])

        point_cloud = point_cloud.scale(1.0 / estimated_scale, center=(0, 0, 0))
        self.logger.trace("_conform_to_grid_space", f"Scaled by {1.0 / estimated_scale:.4f}")
        point_cloud = point_cloud.translate(-translation_vector / estimated_scale)
        self.logger.trace("_conform_to_grid_space", f"Translated by {-translation_vector / estimated_scale}")

        return point_cloud

    def _snap_and_voxelize(self, point_cloud: o3d.geometry.PointCloud) -> VoxelGridType:
        """Snap points to grid and voxelize."""
        snapped_pcd = self._snap_to_grid(point_cloud)
        # self._draw_point_cloud(snapped_pcd, title="Snapped Point Cloud", plot_axis=False, plot_grid=True)
        
        voxel_grid_obj = VoxelGrid.from_point_cloud(snapped_pcd)
        voxel_grid_obj.pixelate_faces(resolution=self.config.voxel_resolution, logger_instance=self.logger)
        return voxel_grid_obj.grid

    def _estimate_phase_from_plane(
        self, axical_plane_models: Optional[list]
    ) -> Dict[str, float]:
        """Estimate phase (offset) from plane models."""
        if axical_plane_models is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        self.logger.trace(
            "._estimate_phase_from_plane",
            "Estimating phase (offset) from plane models..."
        )
        translations = []
        for idx, (plane_model, inliers) in enumerate(axical_plane_models):
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normalize_vector(normal)
            distance = -d / np.linalg.norm(normal)
            translations.append(distance)

        self.logger.trace(
            "._estimate_phase_from_plane",
            f"Estimated translations from planes: {translations}"
        )
        return {
            "x": translations[0],
            "y": translations[1],
            "z": translations[2],
        }

    def _snap_to_grid(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Snap points to nearest grid points along normals."""
        self.logger.trace("._snap_to_grid", "Snapping point cloud to nearest grid points...")
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(
            normals, norms,
            out=np.zeros_like(normals),
            where=norms > self.config.normal_epsilon
        )

        targets = np.round(points)

        with np.errstate(divide='ignore', invalid='ignore'):
            t = (targets - points) / normals

        t[np.abs(normals) < self.config.normal_epsilon] = np.nan

        abs_t = np.abs(t)
        best_axis = np.nanargmin(abs_t, axis=1)

        t_best = t[np.arange(len(points)), best_axis]

        invalid = np.isnan(t_best)
        t_best[invalid] = 0.0

        snapped_points = points + normals * t_best[:, np.newaxis]
        snapped_points[invalid] = targets[invalid]

        point_cloud.points = o3d.utility.Vector3dVector(snapped_points)
        return point_cloud

    def _downsample_point_cloud(self, point_cloud: o3d.geometry.PointCloud, voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
        """Downsample point cloud using voxeling."""
        self.logger.trace(
            "._downsample_point_cloud",
            f"Downsampling point cloud with voxel size: {voxel_size}..."
        )
        return point_cloud.voxel_down_sample(voxel_size=voxel_size)

    def _estimate_normals(self, point_cloud: o3d.geometry.PointCloud, radius: float = 0.1, max_nn: int = 30) -> o3d.geometry.PointCloud:
        """Estimate surface normals."""
        self.logger.trace(
            "._estimate_normals",
            f"Estimating normals with radius: {radius} and max_nn: {max_nn}..."
        )
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        return point_cloud

    def _flip_vertical(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Flip point cloud vertically."""
        self.logger.trace("._flip_vertical", "Flipping point cloud vertically...")
        return point_cloud.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0)),
            center=point_cloud.get_center()
        )

    def _plot_voxel_grid(
        self,
        point_cloud: o3d.geometry.PointCloud,
        voxel_grid: VoxelGridType,
        resolution: Optional[int] = None,
        gap: Optional[float] = None
    ) -> None:
        """Visualize voxel grid with pixelated faces."""
        resolution = resolution or self.config.voxel_resolution
        gap = gap or self.config.voxel_gap

        geometries = []

        step = 1.0 / resolution
        scale = 1.0 - gap
        for (vx, vy, vz), voxel in voxel_grid.items():
            voxel_min = np.array([vx, vy, vz])

            all_vertices = []
            all_triangles = []
            all_colors = []
            v_offset = 0

            for face_key in ["+x", "-x", "+y", "-y", "+z", "-z"]:
                color_grid = voxel.get(f"{face_key}_color_grid", None)

                if color_grid is None:
                    continue

                for i in range(resolution):
                    for j in range(resolution):
                        if np.isnan(color_grid[i, j]).any():
                            continue

                        pixel_color = color_grid[i, j]

                        u0, v0 = i * step, j * step
                        u1, v1 = (i + 1) * step, (j + 1) * step

                        if face_key == "+x":
                            p, q, r, s = [1, u0, v0], [1, u1, v0], [1, u1, v1], [1, u0, v1]
                        elif face_key == "-x":
                            p, q, r, s = [0, u0, v1], [0, u1, v1], [0, u1, v0], [0, u0, v0]
                        elif face_key == "+y":
                            p, q, r, s = [u1, 1, v0], [u0, 1, v0], [u0, 1, v1], [u1, 1, v1]
                        elif face_key == "-y":
                            p, q, r, s = [u0, 0, v0], [u1, 0, v0], [u1, 0, v1], [u0, 0, v1]
                        elif face_key == "+z":
                            p, q, r, s = [u0, v0, 1], [u1, v0, 1], [u1, v1, 1], [u0, v1, 1]
                        elif face_key == "-z":
                            p, q, r, s = [u1, v0, 0], [u0, v0, 0], [u0, v1, 0], [u1, v1, 0]

                        p = np.array(p)
                        q = np.array(q)
                        r = np.array(r)
                        s = np.array(s)
                        p = (p - 0.5) * scale + 0.5
                        q = (q - 0.5) * scale + 0.5
                        r = (r - 0.5) * scale + 0.5
                        s = (s - 0.5) * scale + 0.5

                        all_vertices.extend([voxel_min + p, voxel_min + q, voxel_min + r, voxel_min + s])
                        all_colors.extend([pixel_color] * 4)

                        all_triangles.append([v_offset, v_offset + 1, v_offset + 2])
                        all_triangles.append([v_offset, v_offset + 2, v_offset + 3])
                        v_offset += 4

            if all_vertices:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(np.array(all_vertices))
                mesh.triangles = o3d.utility.Vector3iVector(np.array(all_triangles))
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(all_colors))
                mesh.compute_vertex_normals()
                geometries.append(mesh)

        o3d.visualization.draw_geometries(geometries)

    def _save_point_cloud(self, point_cloud: o3d.geometry.PointCloud, filename: str) -> None:
        """Save point cloud to file."""
        self.logger.trace("._save_point_cloud", f"Saving point cloud to {filename}...")
        o3d.io.write_point_cloud(filename, point_cloud)

    def _draw_point_cloud(
        self,
        point_cloud: o3d.geometry.PointCloud,
        title: str = "Point Cloud Visualization",
        plot_axis: bool = True,
        estimated_axes: Optional[np.ndarray] = None,
        plot_grid: bool = True,
        num_grid_lines: int = 10
    ) -> None:
        """Visualize point cloud with optional axes and grid."""
        self.logger.trace("._draw_point_cloud", "Visualizing point cloud with Open3D...")
        geometries = [point_cloud]
        if plot_axis and estimated_axes is not None:
            axis_length = 1.0
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)
            axes.rotate(estimated_axes.T, center=(0, 0, 0))
            geometries.append(axes)

        if plot_grid:
            grid_lines = []
            for i in range(-num_grid_lines, num_grid_lines + 1):
                line_x = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector([
                        [i, 0, -num_grid_lines],
                        [i, 0, num_grid_lines]
                    ]),
                    lines=o3d.utility.Vector2iVector([[0, 1]])
                )
                line_x.paint_uniform_color([0.8, 0.8, 0.8])
                grid_lines.append(line_x)

                line_z = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector([
                        [-num_grid_lines, 0, i],
                        [num_grid_lines, 0, i]
                    ]),
                    lines=o3d.utility.Vector2iVector([[0, 1]])
                )
                line_z.paint_uniform_color([0.8, 0.8, 0.8])
                grid_lines.append(line_z)
            geometries.extend(grid_lines)

        o3d.visualization.draw_geometries(geometries, window_name=title)

    def _align_point_cloud_to_axes(self, point_cloud: o3d.geometry.PointCloud, axes: np.ndarray) -> o3d.geometry.PointCloud:
        """Rotate point cloud to align with given axes."""
        self.logger.trace(
            "._align_point_cloud_to_axes",
            "Aligning point cloud to estimated axes...",
            {"axes": axes}
        )
        return point_cloud.rotate(axes.T, center=point_cloud.get_center())
