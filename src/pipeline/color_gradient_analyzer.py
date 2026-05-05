from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage, signal
from src.util.logger import get_logger
from src.config import AlignmentConfig


class ColorGradientAnalyzer:
    """Analyzes color gradients in point clouds to estimate scale and phase."""
    
    def __init__(self, config: Optional[AlignmentConfig] = None, logger_instance=None):
        self.config = config or AlignmentConfig()
        self.logger = logger_instance or get_logger(__name__)
    
    def _compute_gradient(
        self,
        point_cloud: o3d.geometry.PointCloud,
        collapsed_axis: str = "y",
        resolution: Optional[float] = None
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Compute color gradient magnitude on 2D projection."""
        resolution = resolution or self.config.gradient_resolution
        
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        colors_lab = cv2.cvtColor(
            (colors * 255).astype(np.uint8)[:, None, :],
            cv2.COLOR_RGB2LAB
        ).reshape(-1, 3)

        projection_map = {"x": (1, 2), "y": (2, 0), "z": (1, 0)}

        if collapsed_axis not in projection_map:
            raise ValueError("collapsed_axis must be 'x', 'y', or 'z'")

        p_idx = projection_map[collapsed_axis]

        mins, maxs = points[:, p_idx].min(axis=0), points[:, p_idx].max(axis=0)
        bins = [np.arange(mins[i], maxs[i] + resolution, resolution) for i in range(2)]
        if any(len(b) < 2 for b in bins):
            return None

        grid_lab = np.stack([
            np.histogram2d(
                points[:, p_idx[0]], points[:, p_idx[1]],
                bins=bins, weights=colors_lab[:, i].astype(np.float64)
            )[0]
            for i in range(3)
        ], axis=-1)

        counts, _, _ = np.histogram2d(points[:, p_idx[0]], points[:, p_idx[1]], bins=bins)
        grid_avg = np.divide(
            grid_lab, counts[..., None],
            out=np.zeros_like(grid_lab),
            where=counts[..., None] > 0
        )
        grid_smooth = ndimage.gaussian_filter(grid_avg, sigma=(1.0, 1.0, 0))
        grads = np.gradient(grid_smooth)
        gradient_magnitude = np.sqrt(np.sum(np.square(grads[0]) + np.square(grads[1]), axis=-1))
        return gradient_magnitude, resolution

    def _find_peaks(
        self,
        gradient_magnitude: np.ndarray,
        threshold_percentile: Optional[int] = None,
        collapsed_axis: str = "y"
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Find peaks in 2D gradient magnitude."""
        threshold_percentile = threshold_percentile or self.config.gradient_threshold_percentile
        
        axis_labels = {"x": ("y", "z"), "y": ("z", "x"), "z": ("y", "x")}
        if collapsed_axis not in axis_labels:
            raise ValueError("collapsed_axis must be 'x', 'y', or 'z'")
        v_label, h_label = axis_labels.get(collapsed_axis)

        distances_v, peaks_v = [], []
        distances_h, peaks_h = [], []
        for row in gradient_magnitude:
            peaks = self._find_1d_peaks(row, threshold_percentile)
            if len(peaks) > 1:
                distances_v.append(np.diff(peaks))
                peaks_v.append(peaks)

        for col in gradient_magnitude.T:
            peaks = self._find_1d_peaks(col, threshold_percentile)
            if len(peaks) > 1:
                distances_h.append(np.diff(peaks))
                peaks_h.append(peaks)

        distances_h = np.concatenate(distances_h) if distances_h else np.array([])
        peaks_h = np.concatenate(peaks_h) if peaks_h else np.array([])
        distances_v = np.concatenate(distances_v) if distances_v else np.array([])
        peaks_v = np.concatenate(peaks_v) if peaks_v else np.array([])

        return {
            v_label: (distances_v, peaks_v),
            h_label: (distances_h, peaks_h)
        }

    def _find_1d_peaks(self, gradient_magnitude: np.ndarray, threshold_percentile: int = 90) -> np.ndarray:
        """Find peaks in 1D gradient."""
        peaks, _ = signal.find_peaks(
            gradient_magnitude,
            height=np.percentile(gradient_magnitude, threshold_percentile)
        )
        return peaks

    def _estimate_scale_iterative(
        self,
        distances: np.ndarray,
        delta_factor: Optional[float] = None,
        max_iters: Optional[int] = None,
        tolerance: Optional[float] = None
    ) -> Optional[float]:
        """Estimate scale using iterative reweighted least squares with Huber loss."""
        delta_factor = delta_factor or self.config.scale_delta_factor
        max_iters = max_iters or self.config.scale_max_iters
        tolerance = tolerance or self.config.scale_tolerance
        
        if len(distances) == 0:
            self.logger.warning(
                "ColorGradientAnalyzer._estimate_scale_iterative",
                "No distances provided for scale estimation."
            )
            return None

        d = np.array(distances, dtype=float)
        S = np.median(d)
        d_max = np.max(d)

        for i in range(max_iters):
            S_prev = S
            k = np.round(d / S)
            mask = k > 0
            if not np.any(mask):
                break
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

            if denominator == 0:
                break
            S = numerator / denominator

            if abs(S - S_prev) < tolerance:
                break

        self.logger.trace(
            "ColorGradientAnalyzer._estimate_scale_iterative",
            f"Estimated scale from distances: {S:.4f} units"
        )
        return S

    def _estimate_phase(
        self,
        peaks: np.ndarray,
        estimated_scale: float,
        delta_factor: float = 0.5,
        max_iters: int = 100,
        tol: float = 1e-6
    ) -> float:
        """Estimate phase (offset) from peaks using iterative refinement."""
        x = np.array(peaks)
        angles = 2 * np.pi * (x % estimated_scale) / estimated_scale
        mean_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        phi = (mean_angle * estimated_scale) / (2 * np.pi)

        for _ in range(max_iters):
            previous_phi = phi

            k = np.round((x - phi) / estimated_scale)
            residuals = x - (k * estimated_scale + phi)

            delta = delta_factor * estimated_scale
            abs_res = np.abs(residuals)
            w = np.where(abs_res <= delta, 1.0, delta / np.maximum(abs_res, 1e-9))

            phi = phi + np.sum(w * residuals) / np.sum(w)

            if abs(phi - previous_phi) < tol:
                break

        return phi % estimated_scale

    def estimate_scale(
        self,
        point_cloud: o3d.geometry.PointCloud,
        resolution: Optional[float] = None,
        axes: Tuple[str, str] = ("x", "z"),
        plot: bool = True
    ) -> Optional[float]:
        """Estimate scale from point cloud using color gradient analysis."""
        resolution = resolution or self.config.gradient_resolution
        
        self.logger.trace(
            "ColorGradientAnalyzer.estimate_scale",
            "Estimating scale using color gradient..."
        )
        all_distances = []
        peaks_info = {"x": [], "y": [], "z": []}
        
        for axis in axes:
            result = self._compute_gradient(point_cloud, collapsed_axis=axis, resolution=resolution)
            if result is None:
                continue

            gradient_magnitude, axis_resolution = result
            gradient_magnitude = ndimage.gaussian_filter(gradient_magnitude, sigma=1.0)
            if plot:
                self._plot_gradient(
                    gradient_magnitude,
                    axis_resolution,
                    title=f"Color Gradient Magnitude (2D projection, collapsed {axis}-axis)"
                )

            axis_results = self._find_peaks(gradient_magnitude, collapsed_axis=axis)
            for label, (distances, peaks) in axis_results.items():
                all_distances.append(distances * axis_resolution)
                peaks_info[label].append(peaks * axis_resolution)

        if not all_distances:
            self.logger.warning(
                "ColorGradientAnalyzer.estimate_scale",
                "No valid color gradient distances found across axes."
            )
            return None

        combined_distances = np.concatenate(all_distances)
        self.logger.trace(
            "ColorGradientAnalyzer.estimate_scale",
            f"Combined color gradient distances: {combined_distances}"
        )

        estimated_scale = self._estimate_scale_iterative(combined_distances)
        return estimated_scale

    def _plot_gradient(self, gradient: np.ndarray, resolution: float, title: str = "Color Gradient Magnitude (2D Projection)") -> None:
        """Plot 2D gradient magnitude using matplotlib."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        extent = np.array(gradient.shape) * resolution / 2
        plt.imshow(
            gradient,
            extent=(-extent[1], extent[1], -extent[0], extent[0]),
            origin='lower',
            cmap='inferno'
        )
        plt.colorbar(label='Gradient Magnitude')
        plt.title(title)
        plt.xlabel('Distance (units)')
        plt.ylabel('Distance (units)')
        plt.grid(True)
        plt.show()
