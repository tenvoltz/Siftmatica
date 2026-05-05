import colorsys
from typing import List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from src.util.geometry import normalize_vector, projected_onto_plane
from src.util.logger import get_logger
from src.config import AlignmentConfig
import open3d as o3d

PlaneModel = Tuple[np.ndarray, List[int]]


class PlaneAnalyzer:
    """Analyzes planes in point clouds and estimates coordinate axes from plane normals."""

    def __init__(self, config: Optional[AlignmentConfig] = None, logger_instance=None):
        self.config = config or AlignmentConfig()
        self.logger = logger_instance or get_logger(__name__)

    def _detect_planes(
        self,
        point_cloud: o3d.geometry.PointCloud,
        max_planes: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        ransac_n: Optional[int] = None,
        num_iterations: Optional[int] = None,
        min_inlier_ratio: Optional[float] = None,
    ) -> List[PlaneModel]:
        """Detect planes using RANSAC segmentation."""
        max_planes = max_planes or self.config.max_planes
        distance_threshold = distance_threshold or self.config.plane_distance_threshold
        ransac_n = ransac_n or self.config.plane_ransac_n
        num_iterations = num_iterations or self.config.plane_num_iterations
        min_inlier_ratio = min_inlier_ratio or self.config.plane_min_inlier_ratio

        self.logger.trace("PlaneAnalyzer._detect_planes", "Detecting planes using RANSAC...")
        planes_models = []
        remaining_pcd = point_cloud
        for i in tqdm(range(max_planes)):
            if len(remaining_pcd.points) < ransac_n:
                self.logger.warning("PlaneAnalyzer._detect_planes", "Not enough points left for plane detection.")
                break

            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            inlier_ratio = len(inliers) / len(remaining_pcd.points)
            if inlier_ratio < min_inlier_ratio:
                self.logger.warning(
                    "PlaneAnalyzer._detect_planes",
                    f"Plane {i+1} detected with inlier ratio {inlier_ratio:.4f}, below threshold. Stopping."
                )
                break

            planes_models.append((plane_model, inliers))
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            self.logger.trace("PlaneAnalyzer._detect_planes", f"Plane {i+1} detected with inlier ratio: {inlier_ratio:.4f}")

        return planes_models

    def _estimate_axes_from_normals(
        self, planes_normals: List[Tuple[np.ndarray, float]]
    ) -> Optional[Tuple[np.ndarray, List[int]]]:
        """Estimate coordinate axes from plane normals and weights."""
        self.logger.trace(
            "PlaneAnalyzer._estimate_axes_from_normals",
            "Estimating axes from plane normals..."
        )
        if len(planes_normals) < 2:
            self.logger.warning(
                "PlaneAnalyzer._estimate_axes_from_normals",
                "Not enough planes detected to estimate axes. Returning None."
            )
            return None

        normals, weights = zip(*planes_normals)
        normals = list(normals)
        weights = list(weights)

        # Estimate vertical (Y) axis
        y_candidates = []
        for idx, normal in enumerate(normals):
            vertical_alignment = abs(float(np.dot(normal, [0, 1, 0])))
            if vertical_alignment < self.config.vertical_alignment_threshold:
                continue
            score = weights[idx] * vertical_alignment
            y_candidates.append((score, normal))

        if not y_candidates:
            return None

        best_y = max(y_candidates, key=lambda item: item[0])
        v_y = best_y[1]
        vertical_idx = next(i for i, n in enumerate(normals) if np.allclose(n, best_y[1]))

        # Estimate Z axis
        z_candidates = []
        for idx, normal in enumerate(normals):
            if abs(float(np.dot(normal, v_y))) > self.config.vertical_alignment_threshold:
                continue
            orthogonality = 1.0 - abs(np.dot(normal, v_y))
            if orthogonality < self.config.orthogonality_threshold:
                continue

            projected = projected_onto_plane(normal, v_y)
            proj_norm = np.linalg.norm(projected)
            if proj_norm < self.config.normal_epsilon:
                continue

            z_axis = projected / proj_norm
            score = weights[idx] * orthogonality
            z_candidates.append((score, z_axis, normal))

        if not z_candidates:
            return None

        best_z = max(z_candidates, key=lambda item: item[0])
        v_z = best_z[1]
        z_idx = next(i for i, n in enumerate(normals) if np.allclose(n, best_z[2]))
        v_x = np.cross(v_y, v_z)

        # Estimate X axis
        x_normal_candidates = []
        for idx, normal in enumerate(normals):
            if idx == vertical_idx or idx == z_idx:
                continue
            alignment = abs(np.dot(normal, v_x))
            score = weights[idx] * alignment
            x_normal_candidates.append((score, normal))

        if not x_normal_candidates:
            return None

        best_x = max(x_normal_candidates, key=lambda item: item[0])
        x_idx = next(i for i, n in enumerate(normals) if np.allclose(n, best_x[1]))

        v_x = normalize_vector(v_x)
        v_z = np.cross(v_x, v_y)
        v_z = normalize_vector(v_z)
        orthonormal_axes = np.column_stack((v_x, v_y, v_z))
        best_align_plane_idxes = [x_idx, vertical_idx, z_idx]
        return orthonormal_axes, best_align_plane_idxes

    def estimate_axes(self, planes_models: List[PlaneModel]) -> Tuple[Optional[np.ndarray], Optional[List[PlaneModel]]]:
        """Estimate axes from plane models."""
        if len(planes_models) >= 2:
            plane_normals = [
                (
                    normalize_vector(np.asarray(plane_model[:3], dtype=np.float64)),
                    len(inliers),
                )
                for plane_model, inliers in planes_models
            ]
            result = self._estimate_axes_from_normals(plane_normals)
            if result is not None:
                axes, best_align_plane_idxes = result
                best_align_plane_models = [planes_models[idx] for idx in best_align_plane_idxes]
                return axes, best_align_plane_models
        return None, None

    def _draw_planes(self, point_cloud, planes_models, title="Detected Planes"):
        plane_colors = self._generate_plane_colors(len(planes_models))
        geometries = []

        for idx, (plane_model, inliers) in enumerate(planes_models):
            color = plane_colors[idx]
            
            plane_pcd = point_cloud.select_by_index(inliers)
            plane_pcd.paint_uniform_color(color)
            geometries.append(plane_pcd)

            points = np.asarray(plane_pcd.points)
            center = points.mean(axis=0)
            size = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

            plane_mesh = self._create_plane_mesh(plane_model, center, size=size)
            plane_mesh.paint_uniform_color(color)
            geometries.append(plane_mesh)
            
        geometries.append(point_cloud)

        o3d.visualization.draw_geometries(geometries, window_name=title)

    def _generate_plane_colors(self,n, saturation=0.8, value=0.9):
        colors = []
        for i in range(n):
            hue = i / n  # evenly spaced hues
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append([r, g, b])
        return colors

    def _create_plane_mesh(self,plane_model, center, size=1.0):
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # Create two orthogonal vectors on the plane
        if abs(normal[2]) > 0.9:
            tangent = np.array([1, 0, 0])
        else:
            tangent = np.cross(normal, [0, 0, 1])
        tangent = tangent / np.linalg.norm(tangent)

        bitangent = np.cross(normal, tangent)

        # Create a square plane
        half = size / 2
        corners = [
            center + (-half * tangent - half * bitangent),
            center + ( half * tangent - half * bitangent),
            center + ( half * tangent + half * bitangent),
            center + (-half * tangent + half * bitangent),
        ]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(corners)
        mesh.triangles = o3d.utility.Vector3iVector([
            [0, 1, 2],
            [0, 2, 3]
        ])
        mesh.compute_vertex_normals()

        return mesh
