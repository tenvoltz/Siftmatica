from collections import defaultdict
import cv2
from src.util.geometry import normalize_vector, projected_onto_plane
from src.util.logger import get_logger
import numpy as np
from tqdm import tqdm
import open3d as o3d
from scipy import fftpack, ndimage, signal

logger = get_logger(__name__)

class PointCloudAlignment:
    def __init__(self, logger_instance=logger):
        self.logger = logger_instance

    def align_point_cloud(self, point_cloud):
        # downsampled_pcd = self._downsample_point_cloud(point_cloud)
        point_cloud = self._flip_vertical(point_cloud)
        point_cloud = self._estimate_normals(point_cloud)
        planes_models = self._detect_planes(point_cloud)
        estimated_axes, _ = self.estimate_axes(planes_models)
        aligned_pcd = self.align_point_cloud_to_axes(point_cloud, estimated_axes)
        aligned_planes_models = self._detect_planes(aligned_pcd)
        _, axical_plane_models = self.estimate_axes(aligned_planes_models)
        estimated_phase = self._estimate_phase_from_plane(axical_plane_models)
        estimated_scale = self.estimate_scale_by_color_gradient(aligned_pcd, plot=False)
        aligned_pcd = self._conform_to_grid(aligned_pcd, estimated_scale, estimated_phase)
        self._draw_point_cloud(
            aligned_pcd,
            title="Aligned Point Cloud",
            plot_grid=True,
            num_grid_lines=10,
        )
        snapped_pcd = self._snap_to_grid(aligned_pcd)
        voxel_grid = self._voxelize_point_cloud(snapped_pcd)
        self._pixelate_faces(voxel_grid, resolution=16)
        self._save_point_cloud(snapped_pcd, "snapped_point_cloud.ply")
        self._plot_voxel_grid(snapped_pcd, voxel_grid)
        return voxel_grid

    def _estimate_phase_from_plane(self, axical_plane_models):
        self.logger.trace("._estimate_phase_from_plane", "Estimating phase (offset) from plane models...")
        translations = []
        for idx, (plane_model, inliers) in enumerate(axical_plane_models):
            # Assume 3 plane (x, y, z) are detected and aligned with the axes, we can estimate the phase by looking at the plane equations
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            # The distance from the origin to the plane along its normal is given by -d / ||normal||
            distance = -d / np.linalg.norm(normal)
            translations.append(distance)
        self.logger.trace("._estimate_phase_from_plane", f"Estimated translations from planes: {translations}")
        return {
            "x": translations[0],
            "y": translations[1],
            "z": translations[2],
        }

    def _conform_to_grid(self, point_cloud, estimated_scale, estimated_phase):
        self.logger.trace("_conform_to_grid", "Conforming point cloud to estimated grid...")
        translation_vector = np.array([estimated_phase.get(axis, 0) for axis in ("x", "y", "z")])

        point_cloud = point_cloud.scale(1.0 / estimated_scale, center=(0, 0, 0))
        self.logger.trace("_conform_to_grid", f"Scaled by {1.0 / estimated_scale:.4f}")
        point_cloud = point_cloud.translate(-translation_vector / estimated_scale)
        self.logger.trace("_conform_to_grid", f"Translated by {-translation_vector / estimated_scale}")

        return point_cloud

    def _snap_to_grid(self, point_cloud):
        self.logger.trace("._snap_to_grid", "Snapping point cloud to nearest grid points...")
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-8)
        eps = 1e-8
        # Target integer planes per axis
        targets = np.round(points)

        # Compute t for each axis: shape (N, 3)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (targets - points) / normals

        # Invalidate axes where normal component is ~0
        t[np.abs(normals) < eps] = np.nan

        # Pick t with smallest absolute value per point
        abs_t = np.abs(t)
        best_axis = np.nanargmin(abs_t, axis=1)

        # Gather best t values
        t_best = t[np.arange(len(points)), best_axis]

        # Handle degenerate normals (all NaN case)
        invalid = np.isnan(t_best)
        t_best[invalid] = 0.0

        # Apply displacement along normals
        snapped_points = points + normals * t_best[:, np.newaxis]

        # For points with degenerate normals, snap directly to target grid points
        snapped_points[invalid] = targets[invalid]

        point_cloud.points = o3d.utility.Vector3dVector(snapped_points)
        return point_cloud

    def _voxelize_point_cloud(self, point_cloud):
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

            # Position relative to voxel center
            rel = pts - center

            # Direction masks (position + normal agreement)
            px = (rel[:, 0] >= 0) & (nmls[:, 0] > 0.5)
            nx = (rel[:, 0] <  0) & (nmls[:, 0] < -0.5)

            py = (rel[:, 1] >= 0) & (nmls[:, 1] > 0.5)
            ny = (rel[:, 1] <  0) & (nmls[:, 1] < -0.5)

            pz = (rel[:, 2] >= 0) & (nmls[:, 2] > 0.5)
            nz = (rel[:, 2] <  0) & (nmls[:, 2] < -0.5)

            # Build sub point cloud once
            sub_pc = o3d.geometry.PointCloud()
            sub_pc.points = o3d.utility.Vector3dVector(pts)
            sub_pc.colors = o3d.utility.Vector3dVector(cols)
            sub_pc.normals = o3d.utility.Vector3dVector(nmls)

            voxel_grid[(x, y, z)] = {
                "center": center,
                "points": pts,
                "colors": cols,
                "+x": sub_pc.select_by_index(np.nonzero(px)[0]),
                "-x": sub_pc.select_by_index(np.nonzero(nx)[0]),
                "+y": sub_pc.select_by_index(np.nonzero(py)[0]),
                "-y": sub_pc.select_by_index(np.nonzero(ny)[0]),
                "+z": sub_pc.select_by_index(np.nonzero(pz)[0]),
                "-z": sub_pc.select_by_index(np.nonzero(nz)[0]),
            }

        self.logger.trace(
            "._voxelize_point_cloud",
            f"Voxel grid created with {len(voxel_grid)} voxels."
        )

        return voxel_grid

    def _plot_voxel_grid(self, point_cloud, voxel_grid, resolution=16, gap=0.02):
        geometries = []

        # # Define the 8 corner offsets of a unit cube relative to the min_bound (x, y, z)
        # # Ordering: 0:(0,0,0), 1:(1,0,0), 2:(0,1,0), 3:(1,1,0), 4:(0,0,1), 5:(1,0,1), 6:(0,1,1), 7:(1,1,1)
        # corners_template = np.array([
        #     [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        #     [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        # ], dtype=float)

        # # Define triangles for each face (2 triangles per face)
        # # Each face points to the indices in corners_template
        # faces_indices = {
        #     "-x": [[0, 2, 4], [2, 6, 4]],
        #     "+x": [[1, 5, 3], [5, 7, 3]],
        #     "-y": [[0, 4, 1], [4, 5, 1]],
        #     "+y": [[2, 3, 6], [3, 7, 6]],
        #     "-z": [[0, 1, 2], [1, 3, 2]],
        #     "+z": [[4, 6, 5], [6, 7, 5]]
        # }

        # for (x, y, z), voxel in voxel_grid.items():
        #     # Option 1: Draw a cube for each voxel colored by the mean color of the points in the voxel
        #     cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        #     cube.translate(voxel["center"] - np.array([0.5, 0.5, 0.5]))
        #     colors = np.asarray(voxel["colors"])
        #     mean_color = np.mean(colors, axis=0)
        #     cube.paint_uniform_color(mean_color)
        #     geometries.append(cube)

        #     # Option 2: Draw the points in the voxel as a point cloud with the mean color
        #     points = np.asarray(voxel["points"])
        #     colors = np.asarray(voxel["colors"])
        #     mean_color = np.mean(colors, axis=0)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(points)
        #     pcd.colors = o3d.utility.Vector3dVector(np.tile(mean_color, (len(points), 1)))
        #     geometries.append(pcd)

        # for (x, y, z), voxel in voxel_grid.items():
        #     all_vertices = []
        #     all_triangles = []
        #     all_colors = []
        #     vertex_count = 0

        #     # Base position for this voxel
        #     min_bound = np.array([x, y, z])

        #     for face_key in ["+x", "-x", "+y", "-y", "+z", "-z"]:
        #         face_pcd = voxel[face_key]

        #         # Skip drawing the face if there are no points associated with it
        #         if len(face_pcd.points) == 0:
        #             continue

        #         # Calculate mean color for this face
        #         face_colors = np.asarray(face_pcd.colors)
        #         mean_color = np.mean(face_colors, axis=0)

        #         # To have sharp face colors, we create unique vertices for every face
        #         for tri in faces_indices[face_key]:
        #             for corner_idx in tri:
        #                 all_vertices.append(min_bound + corners_template[corner_idx])
        #                 all_colors.append(mean_color)

        #             # Add the triangle indices relative to the current vertex_count
        #             all_triangles.append([vertex_count, vertex_count + 1, vertex_count + 2])
        #             vertex_count += 3

        #     if all_vertices:
        #         mesh = o3d.geometry.TriangleMesh()
        #         mesh.vertices = o3d.utility.Vector3dVector(np.array(all_vertices))
        #         mesh.triangles = o3d.utility.Vector3iVector(np.array(all_triangles))
        #         mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(all_colors))

        #         # Optional: compute normals for better lighting
        #         mesh.compute_vertex_normals()
        #         geometries.append(mesh)

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

                if color_grid is None: continue

                for i in range(resolution):
                    for j in range(resolution):
                        # Only draw pixel if it has points (not NaN)
                        if np.isnan(color_grid[i, j]).any():
                            continue

                        pixel_color = color_grid[i, j]

                        # Define 4 corners of the pixel based on face orientation
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

                        # Performed a scaling to create a gap between pixels for better visualization
                        p = np.array(p); q = np.array(q); r = np.array(r); s = np.array(s)
                        p = (p - 0.5) * scale + 0.5
                        q = (q - 0.5) * scale + 0.5
                        r = (r - 0.5) * scale + 0.5
                        s = (s - 0.5) * scale + 0.5
                        
                        # Add vertices (offset by voxel_min)
                        all_vertices.extend(
                            [voxel_min + p, voxel_min + q, voxel_min + r, voxel_min + s]
                        )
                        all_colors.extend([pixel_color] * 4)

                        # Add 2 triangles per pixel
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

    def _pixelate_faces(self, voxel_grid, resolution=16):
        for (vx, vy, vz), voxel in voxel_grid.items():
            voxel_min = np.array([vx, vy, vz])
            for face_key in ["+x", "-x", "+y", "-y", "+z", "-z"]:
                color_grid = self._pixelate_faces_helper(
                    voxel[face_key], voxel_min, face_key, resolution=resolution
                )
                voxel_grid[(vx, vy, vz)][f"{face_key}_color_grid"] = color_grid

    def _pixelate_faces_helper(self, face_pcd, voxel_min, face_type, resolution=16):
        """
        Project points onto the 2D face plane and calculate a resolution x resolution color grid.
        """
        points = np.asarray(face_pcd.points)
        colors = np.asarray(face_pcd.colors)

        if len(points) == 0:
            return None

        # 1. Determine which axes to use based on the face direction
        # We map (x, y, z) to (u, v) coordinates relative to the voxel_min
        if face_type in ["+x", "-x"]:
            u_vals, v_vals = points[:, 1] - voxel_min[1], points[:, 2] - voxel_min[2]
        elif face_type in ["+y", "-y"]:
            u_vals, v_vals = points[:, 0] - voxel_min[0], points[:, 2] - voxel_min[2]
        else:  # +z, -z
            u_vals, v_vals = points[:, 0] - voxel_min[0], points[:, 1] - voxel_min[1]

        # 2. Assign points to grid bins
        # Clip to ensure indices stay within [0, resolution-1]
        u_idx = np.clip((u_vals * resolution).astype(int), 0, resolution - 1)
        v_idx = np.clip((v_vals * resolution).astype(int), 0, resolution - 1)

        # 3. Aggregate colors per pixel
        grid_colors = np.zeros((resolution, resolution, 3))
        grid_counts = np.zeros((resolution, resolution, 1))

        for i in range(len(points)):
            grid_colors[u_idx[i], v_idx[i]] += colors[i]
            grid_counts[u_idx[i], v_idx[i]] += 1

        # Avoid division by zero for empty pixels
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_grid_colors = grid_colors / grid_counts

        return avg_grid_colors

    def _downsample_point_cloud(self, point_cloud, voxel_size=0.05):
        self.logger.trace("._downsample_point_cloud", f"Downsampling point cloud with voxel size: {voxel_size}...")
        downpcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        return downpcd

    def _estimate_normals(self, point_cloud, radius=0.1, max_nn=30):
        self.logger.trace("._estimate_normals", f"Estimating normals with radius: {radius} and max_nn: {max_nn}...")
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        return point_cloud

    def _flip_vertical(self, point_cloud):
        self.logger.trace("._flip_vertical", "Flipping point cloud vertically...")
        return point_cloud.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=point_cloud.get_center())

    def _detect_planes(
        self,
        point_cloud,
        max_planes: int = 5,
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000,
        min_inlier_ratio: float = 0.02,
    ):
        self.logger.trace("._detect_planes", "Detecting planes using RANSAC...")
        planes_models = []
        remaining_pcd = point_cloud
        for i in tqdm(range(max_planes)):
            if len(remaining_pcd.points) < ransac_n:
                self.logger.warning("._detect_planes", "Not enough points left for plane detection.")
                break

            plane_model, inliers = remaining_pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
            inlier_ratio = len(inliers) / len(remaining_pcd.points)
            if inlier_ratio < min_inlier_ratio:
                self.logger.warning("._detect_planes", f"Plane {i+1} detected with inlier ratio {inlier_ratio:.4f}, which is below the threshold. Stopping plane detection.")
                break

            planes_models.append((plane_model, inliers))
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            self.logger.trace("._detect_planes", f"Plane {i+1} detected with inlier ratio: {inlier_ratio:.4f}")

        # self._draw_planes_on_open3d(point_cloud, plane_models)
        return planes_models

    def _draw_planes_on_open3d(self, point_cloud, plane_models):
        self.logger.trace("._draw_planes_on_open3d", "Drawing detected planes on Open3D visualization...")
        plane_colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1]
        ]
        geometries = []
        for idx, plane in enumerate(plane_models):
            a, b, c, d = plane[0]
            inliers = plane[1]
            plane_normal = np.array([a, b, c])
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            plane_point = -d * plane_normal / np.dot(plane_normal, plane_normal)

            # Create a plane mesh manually
            size = 2.0
            vertices = np.array([
                [-size/2, -size/2, 0],
                [size/2, -size/2, 0],
                [size/2, size/2, 0],
                [-size/2, size/2, 0]
            ])
            triangles = np.array([[0, 1, 2], [0, 2, 3]])
            plane_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))

            # Translate to plane position
            plane_mesh.translate(plane_point)

            # Rotate to align with plane normal
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, 0))
            if not np.allclose(plane_normal, [0, 0, 1]):
                rotation_axis = np.cross([0, 0, 1], plane_normal)
                rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
                rotation_angle = np.arccos(np.clip(np.dot([0, 0, 1], plane_normal), -1, 1))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            plane_mesh.rotate(rotation_matrix, center=plane_point)

            plane_mesh.paint_uniform_color(plane_colors[idx % len(plane_colors)])
            geometries.append(plane_mesh)

            # Optionally, add inlier points as a separate point cloud for visualization
            inlier_points = np.asarray(point_cloud.points)[inliers]
            inlier_pcd = o3d.geometry.PointCloud()
            inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)
            inlier_pcd.paint_uniform_color(plane_colors[idx % len(plane_colors)])
            geometries.append(inlier_pcd)

        o3d.visualization.draw_geometries(geometries)

    def _estimate_axes_through_svd(self, point_cloud):
        self.logger.trace("._estimate_axes_through_svd", "Estimating and aligning axes...")
        points = np.asarray(point_cloud.points)
        centered_points = points - np.mean(points, axis=0)
        _, _, Vh = np.linalg.svd(centered_points)
        axes = Vh.T 

        world_up = np.array([0, 1, 0])
        vertical_idx = np.argmax(np.abs(axes[1, :])) 
        v_y = axes[:, vertical_idx]

        # Ensure the vertical axis points upwards
        if np.dot(v_y, world_up) < 0: v_y *= -1

        other_indices = [i for i in range(3) if i != vertical_idx]
        v_x = axes[:, other_indices[0]]
        v_z = np.cross(v_x, v_y)
        v_x = np.cross(v_y, v_z)
        orthonormal_axes = np.column_stack((v_x, v_y, v_z))
        return orthonormal_axes

    def _estimate_axes_through_plane_normals(self, planes_normals):
        self.logger.trace("._estimate_axes_through_plane_normals", "Estimating axes from plane normals...")
        if len(planes_normals) < 2:
            self.logger.warning("._estimate_axes_through_plane_normals", "Not enough planes detected to estimate axes. Returning None.")
            return None

        normals, weights = zip(*planes_normals)
        normals = list(normals)
        weights = list(weights)

        # vertical_idx = np.argmax([abs(n[1]) for n in normals])
        # v_y = normals[vertical_idx]
        # if v_y[1] < 0: v_y *= -1
        # v_y = normalize_vector(v_y)
        y_candidates = []
        for idx, normal in enumerate(normals):
            vertical_alignment = abs(float(np.dot(normal, [0, 1, 0])))
            if vertical_alignment < 0.90: continue 
            score = weights[idx] * vertical_alignment
            y_candidates.append((score, normal))
        if not y_candidates: return None
        best_y = max(y_candidates, key=lambda item: item[0])
        v_y = best_y[1]
        vertical_idx = next(i for i, n in enumerate(normals) if np.allclose(n, best_y[1]))

        z_candidates = []
        for idx, normal in enumerate(normals):
            # Skip planes that are nearly parallel to the vertical axis
            if abs(float(np.dot(normal, v_y))) > 0.90: continue 
            orthogonality = 1.0 - abs(np.dot(normal, v_y))
            # Skip planes that are too parallel to the vertical axis
            if orthogonality < 0.15: continue
            projected = projected_onto_plane(normal, v_y)
            proj_norm = np.linalg.norm(projected)
            # Skip planes that are nearly parallel to the vertical axis after projection
            if proj_norm < 1e-8: continue

            z_axis = projected / proj_norm
            score = weights[idx] * orthogonality
            z_candidates.append((score, z_axis, normal))

        if not z_candidates: return None
        best_z = max(z_candidates, key=lambda item: item[0])
        v_z = best_z[1]
        z_idx = next(i for i, n in enumerate(normals) if np.allclose(n, best_z[2]))
        v_x = np.cross(v_y, v_z)

        x_normal_candidates = []
        for idx, normal in enumerate(normals):
            if idx == vertical_idx or idx == z_idx: continue
            alignment = abs(np.dot(normal, v_x))
            score = weights[idx] * alignment
            x_normal_candidates.append((score, normal))

        if not x_normal_candidates: return None
        best_x = max(x_normal_candidates, key=lambda item: item[0])
        x_idx = next(i for i, n in enumerate(normals) if np.allclose(n, best_x[1]))

        v_x = normalize_vector(v_x)
        v_z = np.cross(v_x, v_y)
        v_z = normalize_vector(v_z)
        orthonormal_axes = np.column_stack((v_x, v_y, v_z))
        best_align_plane_idxes = [x_idx, vertical_idx, z_idx] 
        return orthonormal_axes, best_align_plane_idxes

    def estimate_axes(self, planes_models):
        if len(planes_models) >= 2:
            plane_normals = [
                (
                    normalize_vector(np.asarray(plane_model[:3], dtype=np.float64)),
                    len(inliers),
                )
                for plane_model, inliers in planes_models
            ]
            axes, best_align_plane_idxes = self._estimate_axes_through_plane_normals(plane_normals)
            best_align_plane_models = [planes_models[idx] for idx in best_align_plane_idxes]
            if axes is not None: return axes, best_align_plane_models
        return None, None
        # return self._estimate_axes_through_svd(point_cloud)

    def align_point_cloud_to_axes(self, point_cloud, axes):
        self.logger.trace("._align_point_cloud_to_axes", "Aligning point cloud to estimated axes...", {"axes": axes})
        return point_cloud.rotate(axes.T, center=point_cloud.get_center())

    def estimate_scale_by_color_gradient(self, point_cloud, resolution=0.05, axes=("x", "z"), plot=True):
        self.logger.trace("estimate_scale_by_color_gradient", "Estimating scale using color gradient...")
        all_distances = []
        peaks_info = {
            "x": [],
            "y": [],
            "z": []
        }
        for axis in axes:
            result = self._color_gradient_on_projection(point_cloud, collapsed_axis=axis, resolution=resolution)
            title = f"Color Gradient Magnitude (2D projection, collapsed {axis}-axis)"
            if result is None: continue

            gradient_magnitude, axis_resolution = result
            gradient_magnitude = ndimage.gaussian_filter(gradient_magnitude, sigma=1.0)
            if plot: self._plot_2D_gradient(gradient_magnitude, axis_resolution, title=title) 

            axis_results = self._find_2D_gradient_peak_distance(gradient_magnitude, collapsed_axis=axis)
            for label, (distances, peaks) in axis_results.items():
                all_distances.append(distances * axis_resolution)
                peaks_info[label].append(peaks * axis_resolution)

        if not all_distances:
            self.logger.warning("estimate_scale_by_color_gradient", "No valid color gradient distances found across axes.")
            return None

        combined_distances = np.concatenate(all_distances)
        self.logger.trace("estimate_scale_by_color_gradient", f"Combined color gradient distances: {combined_distances}")

        estimated_scale = self._estimate_scale_from_distances(combined_distances)
        # estimated_phase = {}
        # # Find the phase (offset) for each axis and adjust distances accordingly
        # for label in axes:
        #     if peaks_info[label]:
        #         all_peaks = np.concatenate(peaks_info[label])
        #         phase = self._estimate_phase_from_peaks(all_peaks, estimated_scale)
        #         self.logger.trace("estimate_scale_by_color_gradient", f"Estimated phase for {label}-axis: {phase:.4f} units")
        #         estimated_phase[label] = phase

        return estimated_scale #, estimated_phase

    def _color_gradient_on_projection(self, point_cloud, collapsed_axis="y", resolution=0.05):
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        colors_lab = cv2.cvtColor((colors * 255).astype(np.uint8)[:, None, :], cv2.COLOR_RGB2LAB).reshape(-1, 3)

        projection_map = {"x": (1, 2), "y": (2, 0), "z": (1, 0)} 

        if collapsed_axis not in projection_map:
            raise ValueError("collapsed_axis must be 'x', 'y', or 'z'")

        p_idx = projection_map[collapsed_axis]

        mins, maxs = points[:, p_idx].min(axis=0), points[:, p_idx].max(axis=0)
        bins = [np.arange(mins[i], maxs[i] + resolution, resolution) for i in range(2)]
        if any(len(b) < 2 for b in bins): return None

        grid_lab = np.stack([
            np.histogram2d(points[:, p_idx[0]], points[:, p_idx[1]],
                           bins=bins, weights=colors_lab[:, i].astype(np.float64))[0]
            for i in range(3)
        ], axis=-1)

        counts, _, _ = np.histogram2d(points[:, p_idx[0]], points[:, p_idx[1]], bins=bins)
        grid_avg = np.divide(grid_lab, counts[..., None], out=np.zeros_like(grid_lab), where=counts[..., None] > 0)
        grid_smooth = ndimage.gaussian_filter(grid_avg, sigma=(1.0, 1.0, 0))
        grads = np.gradient(grid_smooth)
        # We only care about spatial gradients (0 and 1), not the channel gradient (2)
        gradient_magnitude = np.sqrt(np.sum(np.square(grads[0]) + np.square(grads[1]), axis=-1))
        return gradient_magnitude, resolution

    def _plot_2D_gradient(self, gradient, resolution, title="Color Gradient Magnitude (2D Projection)"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        extent = np.array(gradient.shape) * resolution / 2
        plt.imshow(gradient, extent=(-extent[1], extent[1], -extent[0], extent[0]), origin='lower', cmap='inferno')
        plt.colorbar(label='Gradient Magnitude')
        plt.title(title)
        plt.xlabel('Distance (units)')
        plt.ylabel('Distance (units)')
        plt.grid(True)
        plt.show()

    def _find_2D_gradient_peak_distance(
        self, gradient_magnitude, threshold_percentile=90, collapsed_axis="y"
    ):
        axis_labels = {"x": ("y", "z"), "y": ("z", "x"), "z": ("y", "x")}
        if collapsed_axis not in axis_labels:
            raise ValueError("collapsed_axis must be 'x', 'y', or 'z'")
        v_label, h_label = axis_labels.get(collapsed_axis)

        distances_v, peaks_v = [], []
        distances_h, peaks_h = [], []
        for row in gradient_magnitude:
            peaks = self._find_1D_gradient_peaks(row, threshold_percentile)
            if len(peaks) > 1:
                distances_v.append(np.diff(peaks))
                peaks_v.append(peaks)

        for col in gradient_magnitude.T:
            peaks = self._find_1D_gradient_peaks(col, threshold_percentile)
            if len(peaks) > 1:
                distances_h.append(np.diff(peaks))
                peaks_h.append(peaks)

        distances_h = np.concatenate(distances_h) if distances_h else np.array([])
        peaks_h = np.concatenate(peaks_h) if peaks_h else np.array([])
        distances_v = np.concatenate(distances_v) if distances_v else np.array([])
        peaks_v = np.concatenate(peaks_v) if peaks_v else np.array([])

        # profile_v = np.sum(gradient_magnitude, axis=1)
        # profile_h = np.sum(gradient_magnitude, axis=0)
        # peaks_v = self._find_1D_gradient_peaks(profile_v, threshold_percentile)
        # peaks_h = self._find_1D_gradient_peaks(profile_h, threshold_percentile)
        # distances_v = np.diff(peaks_v) if len(peaks_v) > 1 else np.array([])
        # distances_h = np.diff(peaks_h) if len(peaks_h) > 1 else np.array([])

        return {
            v_label: (distances_v, peaks_v),
            h_label: (distances_h, peaks_h)
        }

    def _estimate_scale_from_distances(self, distances, delta_factor=0.5, max_iters=100, tol=1e-6):
        """
        Estimate the scale from a set of distance measurements using an iterative reweighted least squares approach 
        with weighted Huber loss to mitigate the influence of outliers.
        """
        if len(distances) == 0:
            self.logger.warning("._estimate_scale_from_distances", "No distances provided for scale estimation.")
            return None

        d = np.array(distances, dtype=float)
        S = np.median(d)
        d_max = np.max(d)

        for i in range(max_iters):
            S_prev = S
            k = np.round(d / S)
            mask = k > 0
            if not np.any(mask): break
            d_m = d[mask]
            k_m = k[mask]

            w_base = d_m / d_max

            residuals = d_m - (k_m * S)
            delta = delta_factor * S
            abs_res = np.abs(residuals)
            w_huber = np.where(abs_res <= delta, 1.0, delta / np.maximum(abs_res, 1e-9))

            w = w_base * w_huber

            numerator = np.sum(w * k_m * d_m)
            denominator = np.sum(w * k_m**2)

            if denominator == 0: break
            S = numerator / denominator

            if abs(S - S_prev) < tol: break

        self.logger.trace("._estimate_scale_from_distances", f"Estimated scale from distances: {S:.4f} units")
        return S

    def _estimate_phase_from_peaks(
        self, peaks, estimated_scale, delta_factor=0.5, max_iters=100, tol=1e-6
    ):
        x = np.array(peaks)
        angles = 2 * np.pi * (x % estimated_scale) / estimated_scale
        mean_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        phi = (mean_angle * estimated_scale) / (2 * np.pi)

        for _ in range(max_iters):
            previous_phi = phi

            k = np.round((x - phi) / estimated_scale)
            residuals = x - (k * estimated_scale + phi)

            # Robust weighting
            delta = delta_factor * estimated_scale
            abs_res = np.abs(residuals)
            w = np.where(abs_res <= delta, 1.0, delta / np.maximum(abs_res, 1e-9))

            # Weighted mean of the residuals added to current phi
            phi = phi + np.sum(w * residuals) / np.sum(w)

            if abs(phi - previous_phi) < tol:
                break

        return phi % estimated_scale

    def _find_1D_gradient_peaks(self, gradient_magnitude, threshold_percentile=90):
        peaks, _ = signal.find_peaks(gradient_magnitude, height=np.percentile(gradient_magnitude, threshold_percentile))
        return peaks

    # def _color_gradient_on_axis(self, point_cloud, axis="x", resolution=0.05):
    #     points = np.asarray(point_cloud.points)
    #     colors = np.asarray(point_cloud.colors)
    #     colors_lab = cv2.cvtColor((colors * 255).astype(np.uint8)[:, None, :], cv2.COLOR_RGB2LAB).reshape(-1, 3)

    #     ax_map = {"x": 0, "y": 1, "z": 2}
    #     a_idx = ax_map.get(axis, 0)

    #     bins = np.arange(points[:, a_idx].min(), points[:, a_idx].max() + resolution, resolution)
    #     if len(bins) < 2: return None

    #     grid_lab = np.stack([
    #         np.histogram(points[:, a_idx], bins=bins, weights=colors_lab[:, i].astype(np.float64))[0]
    #         for i in range(3)
    #     ], axis=-1)

    #     counts, _ = np.histogram(points[:, a_idx], bins=bins)
    #     grid_avg = np.divide(grid_lab, counts[:, None], out=np.zeros_like(grid_lab), where=counts[:, None] > 0)
    #     grid_smooth = ndimage.gaussian_filter(grid_avg, sigma=(1.0, 0))
    #     grads = np.gradient(grid_smooth, axis=0)
    #     gradient_magnitude = np.sqrt(np.sum(np.square(grads), axis=-1))
    #     return gradient_magnitude, resolution

    # def _plot_1D_gradient(self, gradient, resolution, title="Color Gradient Magnitude (1D)"):
    #     import matplotlib.pyplot as plt
    #     x = (np.arange(len(gradient)) - len(gradient) // 2) * resolution
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(x, gradient)
    #     plt.title(title)
    #     plt.xlabel('Distance (units)')
    #     plt.ylabel('Gradient Magnitude')
    #     plt.grid(True)
    #     plt.show()

    # def estimate_scale_by_auto_correlation(self, point_cloud, mode="1d", resolution=0.05, axes=("x", "z"), plot=True):
    #     self.logger.trace("._estimate_scale_by_auto_correlation", f"Estimating scale using auto-correlation (mode={mode})...")

    #     if mode not in {"2d", "1d"}:
    #         self.logger.warning("._estimate_scale_by_auto_correlation", f"Unknown auto-correlation mode: {mode}. Use '2d' or '1d'.")
    #         return None

    #     all_distances = []
    #     for axis in axes:
    #         if mode == "2d":
    #             result = self._auto_correlate_on_projection(point_cloud, collapsed_axis=axis, resolution=resolution)
    #             title = f"Auto-correlation (2D projection, collapsed {axis}-axis)"
    #         else:
    #             result = self._auto_correlate_on_axis(point_cloud, axis=axis, resolution=resolution)
    #             title = f"Auto-correlation (1D along {axis}-axis)"

    #         if result is None:
    #             continue

    #         autocorr, axis_resolution = result
    #         # Smooth the auto-correlation to reduce noise and make valleys more detectable
    #         autocorr = ndimage.gaussian_filter(autocorr, sigma=1.0)

    #         if plot:
    #             self._plot_autocorrelation(autocorr, axis_resolution, title=title)

    #         distances = self._get_distances_from_autocorrelation(autocorr, axis_resolution)
    #         if distances is not None and len(distances) > 0:
    #             all_distances.append(distances)

    #     if not all_distances:
    #         self.logger.warning("._estimate_scale_by_auto_correlation", "No valid auto-correlation distances found across axes.")
    #         return None
    #     self.logger.trace("._estimate_scale_by_auto_correlation", f"Auto-correlation distances found: {[d for dist in all_distances for d in dist]}")
    #     combined_distances = np.concatenate(all_distances)
    #     estimated_scale = float(np.median(combined_distances))
    #     self.logger.trace("estimate_scale_by_auto_correlation", f"Estimated scale: {estimated_scale:.4f} units")
    #     return estimated_scale

    # def _auto_correlate_on_projection(self, point_cloud, collapsed_axis="y", resolution=0.05):
    #     points = np.asarray(point_cloud.points)
    #     extents = points.max(axis=0) - points.min(axis=0)
    #     collapsed_axis_idx = {"x": 0, "y": 1, "z": 2}.get(collapsed_axis, 1)
    #     planar_axes_idx = [i for i in range(3) if i != collapsed_axis_idx]
    #     if extents[collapsed_axis_idx] < 1e-6:
    #         self.logger.warning("._auto_correlate_on_projection", f"Extent along {collapsed_axis}-axis is too small for auto-correlation. Skipping scale estimation.")
    #         return None
    #     self.logger.trace("._auto_correlate_on_projection", f"Projecting point cloud onto {collapsed_axis}-axis for auto-correlation...")

    #     planar_points = points[:, planar_axes_idx]
    #     x_range = np.arange(planar_points[:, 0].min(), planar_points[:, 0].max() + resolution, resolution)
    #     y_range = np.arange(planar_points[:, 1].min(), planar_points[:, 1].max() + resolution, resolution)
    #     if len(x_range) < 2 or len(y_range) < 2:
    #         self.logger.warning("._auto_correlate_on_projection", f"Insufficient bins on {collapsed_axis}-projection for auto-correlation.")
    #         return None

    #     counts, _, _ = np.histogram2d(planar_points[:,0], planar_points[:,1], bins=[x_range, y_range])
    #     if point_cloud.has_colors():
    #         colors = np.asarray(point_cloud.colors)
    #         h_R, _, _ = np.histogram2d(planar_points[:,0], planar_points[:,1], bins=[x_range, y_range], weights=colors[:,0])
    #         h_G, _, _ = np.histogram2d(planar_points[:,0], planar_points[:,1], bins=[x_range, y_range], weights=colors[:,1])
    #         h_B, _, _ = np.histogram2d(planar_points[:,0], planar_points[:,1], bins=[x_range, y_range], weights=colors[:,2])
    #         g_R = np.divide(h_R, counts, out=np.zeros_like(h_R), where=counts > 0)
    #         g_G = np.divide(h_G, counts, out=np.zeros_like(h_G), where=counts > 0)
    #         g_B = np.divide(h_B, counts, out=np.zeros_like(h_B), where=counts > 0)

    #         ps_total = np.zeros((len(x_range)-1, len(y_range)-1), dtype=np.float64)
    #         for g in [g_R, g_G, g_B]:
    #             g_centered = g - np.mean(g)
    #             f_transform = np.fft.fft2(g_centered)
    #             ps_total += np.abs(f_transform)**2
    #         autocorr_complex = np.fft.ifft2(ps_total)
    #     else:
    #         grid = counts - np.mean(counts)
    #         autocorr_complex = np.fft.ifft2(np.abs(np.fft.fft2(grid))**2)

    #     autocorr = np.fft.fftshift(np.real(autocorr_complex))
    #     return autocorr, resolution

    # def _auto_correlate_on_axis(self, point_cloud, axis="x", resolution=0.05):
    #     axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
    #     points = np.asarray(point_cloud.points)
    #     values = points[:, axis_idx]
    #     extent = values.max() - values.min()

    #     if extent < 1e-6:
    #         self.logger.warning("._auto_correlate_on_axis", f"Extent along {axis}-axis is too small for 1D auto-correlation. Skipping.")
    #         return None

    #     bins = np.arange(values.min(), values.max() + resolution, resolution)
    #     if len(bins) < 2:
    #         self.logger.warning("._auto_correlate_on_axis", f"Insufficient bins along {axis}-axis for 1D auto-correlation.")
    #         return None

    #     if point_cloud.has_colors():
    #         colors = np.asarray(point_cloud.colors)
    #         counts, _ = np.histogram(values, bins=bins)
    #         h_R, _ = np.histogram(values, bins=bins, weights=colors[:, 0])
    #         h_G, _ = np.histogram(values, bins=bins, weights=colors[:, 1])
    #         h_B, _ = np.histogram(values, bins=bins, weights=colors[:, 2])

    #         g_R = np.divide(h_R, counts, out=np.zeros_like(h_R, dtype=np.float64), where=counts > 0)
    #         g_G = np.divide(h_G, counts, out=np.zeros_like(h_G, dtype=np.float64), where=counts > 0)
    #         g_B = np.divide(h_B, counts, out=np.zeros_like(h_B, dtype=np.float64), where=counts > 0)

    #         ps_total = np.zeros(len(bins) - 1, dtype=np.float64)
    #         for g in [g_R, g_G, g_B]:
    #             g_centered = g - np.mean(g)
    #             f_transform = np.fft.fft(g_centered)
    #             ps_total += np.abs(f_transform) ** 2
    #         autocorr_complex = np.fft.ifft(ps_total)
    #     else:
    #         counts, _ = np.histogram(values, bins=bins)
    #         signal = counts - np.mean(counts)
    #         autocorr_complex = np.fft.ifft(np.abs(np.fft.fft(signal)) ** 2)

    #     autocorr = np.fft.fftshift(np.real(autocorr_complex))
    #     return autocorr, resolution

    # def _get_distances_from_autocorrelation(self, autocorr, resolution):
    #     self.logger.trace("._get_distances_from_autocorrelation", "Estimating scale from auto-correlation...")
    #     if autocorr.ndim == 1:
    #         return self._get_1d_peak_distances(autocorr, resolution)
    #     else:
    #         return self._get_2d_peak_distances(autocorr, resolution)

    # def _get_1d_peak_distances(self, autocorr, resolution):
    #     center = len(autocorr) // 2
    #     peak_indices = signal.argrelextrema(autocorr, np.greater, order=2)[0]
    #     valid_peaks = autocorr[peak_indices]
    #     distances = np.abs(valid_peaks - center)
    #     distances = distances[distances > 0]

    #     return distances * resolution
    # def _get_2d_peak_distances(self, autocorr, resolution):
    #     center = np.array(autocorr.shape) // 2
    #     local_max = ndimage.maximum_filter(autocorr, size=10) == autocorr
    #     threshold = np.percentile(autocorr, 95)
    #     peak_mask = (autocorr > threshold) & local_max
    #     peaks = np.argwhere(peak_mask)
    #     distances = np.linalg.norm(peaks - center, axis=1)
    #     distances = distances[distances > (resolution * 0.5)]

    #     return distances * resolution

    # def _plot_autocorrelation(self, autocorr, resolution, title="Auto-correlation"):
    #     import matplotlib.pyplot as plt
    #     if autocorr.ndim == 1:
    #         plt.figure(figsize=(8, 4))
    #         x = (np.arange(len(autocorr)) - len(autocorr) // 2) * resolution
    #         plt.plot(x, autocorr)
    #         plt.title(title)
    #         plt.xlabel('Lag distance (units)')
    #         plt.ylabel('Auto-correlation')
    #         plt.grid(True)
    #         plt.show()
    #         return

    #     plt.figure(figsize=(6, 6))
    #     extent = np.array(autocorr.shape) * resolution / 2
    #     plt.imshow(autocorr, extent=(-extent[1], extent[1], -extent[0], extent[0]), origin='lower', cmap='viridis')
    #     plt.colorbar(label='Auto-correlation')
    #     plt.title(title)
    #     plt.xlabel('Distance (units)')
    #     plt.ylabel('Distance (units)')
    #     plt.grid(True)
    #     plt.show()

    def _save_point_cloud(self, point_cloud, filename):
        self.logger.trace("._save_point_cloud", f"Saving point cloud to {filename}...")
        o3d.io.write_point_cloud(filename, point_cloud)

    def _draw_point_cloud(self, point_cloud, title="Point Cloud Visualization", 
                          plot_axis=True, estimated_axes=None,
                          plot_grid=True, num_grid_lines=10):
        self.logger.trace("._draw_point_cloud", "Visualizing point cloud with Open3D...")
        geometries = [point_cloud]
        if plot_axis and estimated_axes is not None:
            axis_length = 1.0
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)
            axes.rotate(estimated_axes.T, center=(0, 0, 0))
            geometries.append(axes)

        if plot_grid:
            # Create grid lines based on the estimated scale on the floor plane
            grid_lines = []
            for i in range(-num_grid_lines, num_grid_lines + 1):
                # Lines parallel to X-axis
                line_x = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector([[i, 0, -num_grid_lines],
                                                        [i, 0, num_grid_lines]]),
                    lines=o3d.utility.Vector2iVector([[0, 1]])
                )
                line_x.paint_uniform_color([0.8, 0.8, 0.8])
                grid_lines.append(line_x)

                # Lines parallel to Z-axis
                line_z = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector([[-num_grid_lines, 0, i],
                                                        [num_grid_lines, 0, i]]),
                    lines=o3d.utility.Vector2iVector([[0, 1]])
                )
                line_z.paint_uniform_color([0.8, 0.8, 0.8])
                grid_lines.append(line_z)
            geometries.extend(grid_lines)

        o3d.visualization.draw_geometries(geometries, window_name=title)
