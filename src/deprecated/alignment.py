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