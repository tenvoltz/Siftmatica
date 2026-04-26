from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AlignmentConfig:
	voxel_downsample: float = 0.05
	max_planes: int = 6
	plane_distance: float = 0.03
	ransac_iterations: int = 2000
	scale_bins: int = 4096
	scale_min: float = 0.6
	scale_max: float = 1.4
	scale_smooth_window: int = 5
	axis_guide_n: int = 5
	axis_guide_subdivisions: int = 20
	projection_bins: int = 256
	projection_plot_path: Path | None = Path("data/house1-dense/workspace/dense/0/fused_projection_compare.png")
	plot_projections: bool = False


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	norm = np.linalg.norm(vec)
	if norm < eps:
		raise ValueError("Cannot normalize near-zero vector.")
	return vec / norm


def _ensure_right_handed(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	x_axis = _normalize(x_axis)
	y_axis = _normalize(y_axis)
	z_axis = _normalize(z_axis)
	if np.linalg.det(np.column_stack([x_axis, y_axis, z_axis])) < 0:
		x_axis = -x_axis
	return x_axis, y_axis, z_axis


def _project_onto_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
	return vec - np.dot(vec, normal) * normal


def _best_vertical_normal(normals: list[np.ndarray], up_hint: np.ndarray) -> np.ndarray:
	best_normal = normals[0]
	best_score = -1.0
	for normal in normals:
		alignment = abs(float(np.dot(normal, up_hint)))
		verticality = abs(float(normal[1]))
		score = 0.65 * alignment + 0.35 * verticality
		if score > best_score:
			best_score = score
			best_normal = normal
	if np.dot(best_normal, up_hint) < 0:
		best_normal = -best_normal
	if best_normal[1] < 0:
		best_normal = -best_normal
	return _normalize(best_normal)


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
	if window <= 1:
		return signal
	window = int(window)
	kernel = np.ones(window, dtype=np.float64) / float(window)
	return np.convolve(signal, kernel, mode="same")


def _autocorrelation_fft(signal: np.ndarray) -> np.ndarray:
	if signal.size == 0:
		return signal
	n = signal.size
	fft_size = 1 << (2 * n - 1).bit_length()
	freq = np.fft.rfft(signal, n=fft_size)
	ac = np.fft.irfft(freq * np.conj(freq), n=fft_size)[:n]
	if ac[0] <= 1e-12:
		return ac
	return ac / ac[0]


def _local_peak_lags(ac: np.ndarray, lag_min: int, lag_max: int) -> list[int]:
	return [
		i
		for i in range(lag_min + 1, lag_max)
		if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]
	]


def _score_autocorr_peak(ac: np.ndarray, lag: int, noise_floor: float, lag_max: int) -> float:
	prom = max(0.0, float(ac[lag] - noise_floor))
	harmonics = []
	for k in [2, 3, 4]:
		idx = lag * k
		if idx <= lag_max:
			harmonics.append(max(0.0, float(ac[idx] - noise_floor)))
	harmonic_support = float(np.mean(harmonics)) if harmonics else 0.0
	lag_penalty = 0.20 * (lag / max(lag_max, 1))
	return prom + 0.6 * harmonic_support - lag_penalty


def _choose_autocorr_peak(
	ac: np.ndarray,
	local_peaks: list[int],
	lag_max: int,
	noise_floor: float,
) -> tuple[int, float, float]:
	peak_values = np.array([ac[i] for i in local_peaks], dtype=np.float64)
	best_peak_value = float(np.max(peak_values))
	prominences = np.array([max(0.0, float(ac[i] - noise_floor)) for i in local_peaks], dtype=np.float64)

	if np.all(prominences <= 1e-9):
		chosen_lag = int(local_peaks[int(np.argmax(peak_values))])
		return chosen_lag, best_peak_value, float(peak_values[np.argmax(peak_values)])

	prom_threshold = float(np.quantile(prominences, 0.75))
	candidates = [i for i, p in zip(local_peaks, prominences, strict=False) if p >= max(0.02, prom_threshold * 0.5)]
	if not candidates:
		candidates = list(local_peaks)

	sorted_candidates = sorted(candidates, key=lambda lag: _score_autocorr_peak(ac, lag, noise_floor, lag_max), reverse=True)
	best_candidate = sorted_candidates[0]
	best_score = _score_autocorr_peak(ac, best_candidate, noise_floor, lag_max)

	for lag in sorted(candidates):
		if _score_autocorr_peak(ac, lag, noise_floor, lag_max) >= 0.90 * best_score and (ac[lag] - noise_floor) >= 0.05:
			return int(lag), best_peak_value, float(ac[lag])

	return int(best_candidate), best_peak_value, float(ac[best_candidate])


def estimate_scale_axis_autocorr(
	coords: np.ndarray,
	extent: float,
	axis_name: str,
	bins: int,
	scale_min: float,
	scale_max: float,
	smooth_window: int,
) -> dict:
	if extent <= 1e-9:
		return {
			"axis": axis_name,
			"valid": False,
			"reason": "degenerate_extent",
		}

	bins = int(max(128, bins))
	bins = int(min(16384, bins))

	hist, edges = np.histogram(coords, bins=bins, range=(0.0, extent))
	signal = hist.astype(np.float64)
	signal = _moving_average(signal, smooth_window)
	signal = signal - signal.mean()

	std = signal.std()
	if std < 1e-12:
		return {
			"axis": axis_name,
			"valid": False,
			"reason": "flat_signal",
		}
	signal = signal / std

	bin_size = float(edges[1] - edges[0])
	ac = _autocorrelation_fft(signal)

	lag_min = max(1, int(np.floor(scale_min / max(bin_size, 1e-12))))
	lag_max = min(len(ac) - 1, int(np.ceil(scale_max / max(bin_size, 1e-12))))
	if lag_max <= lag_min + 2:
		return {
			"axis": axis_name,
			"valid": False,
			"reason": "lag_window_too_small",
			"bin_size": bin_size,
		}

	local_peaks = _local_peak_lags(ac, lag_min, lag_max)

	if not local_peaks:
		best_idx = int(np.argmax(ac[lag_min: lag_max + 1]) + lag_min)
		local_peaks = [best_idx]

	noise_floor = float(np.median(ac[lag_min: lag_max + 1]))
	chosen_lag, best_peak_value, selected_peak_value = _choose_autocorr_peak(
		ac,
		local_peaks,
		lag_max,
		noise_floor,
	)
	snr = (best_peak_value - noise_floor) / (abs(noise_floor) + 1e-6)

	estimated_scale = float(chosen_lag * bin_size)
	strength = float(selected_peak_value)
	confidence = float(np.clip((strength - noise_floor) / (0.5 + abs(noise_floor) + 1e-6), 0.0, 1.0))

	top_peaks = sorted(
		[(int(i), float(ac[i]), float(i * bin_size)) for i in local_peaks],
		key=lambda item: item[1],
		reverse=True,
	)[:8]

	return {
		"axis": axis_name,
		"valid": True,
		"bin_size": bin_size,
		"lag_min": int(lag_min),
		"lag_max": int(lag_max),
		"estimated_scale": estimated_scale,
		"selected_lag": int(chosen_lag),
		"selected_peak_value": strength,
		"best_peak_value": best_peak_value,
		"noise_floor": noise_floor,
		"snr": float(snr),
		"confidence": confidence,
		"top_peaks": [
			{"lag": lag, "corr": corr, "distance": dist} for lag, corr, dist in top_peaks
		],
	}


def estimate_scale_autocorr(
	aligned_points: np.ndarray,
	bins: int,
	scale_min: float,
	scale_max: float,
	smooth_window: int,
) -> dict:
	extents = aligned_points.max(axis=0) - aligned_points.min(axis=0)
	x_result = estimate_scale_axis_autocorr(
		aligned_points[:, 0],
		float(extents[0]),
		"x",
		bins,
		scale_min,
		scale_max,
		smooth_window,
	)
	z_result = estimate_scale_axis_autocorr(
		aligned_points[:, 2],
		float(extents[2]),
		"z",
		bins,
		scale_min,
		scale_max,
		smooth_window,
	)

	valid = [r for r in [x_result, z_result] if r.get("valid", False)]
	if not valid:
		return {
			"method": "autocorrelation_xz",
			"success": False,
			"reason": "no_valid_axis_estimate",
			"axes": {"x": x_result, "z": z_result},
		}

	scales = np.array([r["estimated_scale"] for r in valid], dtype=np.float64)
	axis_conf = np.array([r["confidence"] for r in valid], dtype=np.float64)
	block_scale = float(np.median(scales))
	selected_axis = valid[int(np.argmax(axis_conf))]["axis"]
	aggregation_mode = "median"

	if len(scales) == 2:
		agreement = 1.0 - min(1.0, abs(scales[0] - scales[1]) / max(max(scales[0], scales[1]), 1e-9))
		if agreement < 0.60:
			best_idx = int(np.argmax(axis_conf))
			block_scale = float(scales[best_idx])
			selected_axis = valid[best_idx]["axis"]
			aggregation_mode = "best_axis"
	else:
		agreement = 0.6
		block_scale = float(scales[0])
		selected_axis = valid[0]["axis"]
		aggregation_mode = "single_axis"

	global_confidence = float(np.clip(0.5 * float(axis_conf.mean()) + 0.5 * agreement, 0.0, 1.0))
	if aggregation_mode == "best_axis":
		global_confidence = float(global_confidence * 0.85)

	return {
		"method": "autocorrelation_xz",
		"success": True,
		"estimated_block_scale": block_scale,
		"confidence": global_confidence,
		"axis_agreement": float(agreement),
		"selected_axis": selected_axis,
		"aggregation_mode": aggregation_mode,
		"axes": {"x": x_result, "z": z_result},
	}


def _recover_grid_interval_axis(coords: np.ndarray, axis_name: str, scale: float, phase_bins: int) -> dict:
	coord_min = float(coords.min())
	coord_max = float(coords.max())
	phases = np.mod(coords, scale)
	hist, edges = np.histogram(phases, bins=phase_bins, range=(0.0, scale))
	peak_idx = int(np.argmax(hist))
	phase_anchor = float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))

	k_first = int(np.ceil((coord_min - phase_anchor) / scale))
	start = float(phase_anchor + k_first * scale)
	end = float(start + scale)

	if end > coord_max and (start - scale) >= coord_min:
		start = float(start - scale)
		end = float(start + scale)

	if axis_name == "x":
		start_point = [start, 0.0, 0.0]
		end_point = [end, 0.0, 0.0]
	else:
		start_point = [0.0, 0.0, start]
		end_point = [0.0, 0.0, end]

	return {
		"success": True,
		"axis": axis_name,
		"phase_anchor": phase_anchor,
		"interval_1d": [start, end],
		"start_point": start_point,
		"end_point": end_point,
		"phase_bin_count": phase_bins,
		"peak_support": int(hist[peak_idx]),
	}


def recover_grid_interval(aligned_points: np.ndarray, scale_info: dict, phase_bins: int = 256) -> dict:
	if not scale_info.get("success", False):
		return {"success": False, "reason": "scale_unavailable"}

	scale = float(scale_info.get("estimated_block_scale", 0.0))
	if scale <= 1e-9:
		return {"success": False, "reason": "invalid_scale"}

	phase_bins = int(max(32, phase_bins))
	x_interval = _recover_grid_interval_axis(np.asarray(aligned_points[:, 0], dtype=np.float64), "x", scale, phase_bins)
	z_interval = _recover_grid_interval_axis(np.asarray(aligned_points[:, 2], dtype=np.float64), "z", scale, phase_bins)

	selected_axis = str(scale_info.get("selected_axis", "x")).lower()
	primary = x_interval if selected_axis == "x" else z_interval

	return {
		"success": True,
		"axis": selected_axis,
		"scale": scale,
		"interval_1d": primary["interval_1d"],
		"start_point": primary["start_point"],
		"end_point": primary["end_point"],
		"intervals": {
			"x": x_interval,
			"z": z_interval,
		},
	}


def _scale_grid_interval(grid_interval: dict, scale_factor: float) -> dict:
	if not isinstance(grid_interval, dict) or not grid_interval.get("success", False):
		return grid_interval

	intervals = {}
	for axis_key, axis_interval in grid_interval.get("intervals", {}).items():
		if not isinstance(axis_interval, dict) or not axis_interval.get("success", False):
			intervals[axis_key] = axis_interval
			continue
		start, end = axis_interval["interval_1d"]
		intervals[axis_key] = {
			**axis_interval,
			"interval_1d": [float(start * scale_factor), float(end * scale_factor)],
			"start_point": (np.asarray(axis_interval["start_point"], dtype=np.float64) * scale_factor).tolist(),
			"end_point": (np.asarray(axis_interval["end_point"], dtype=np.float64) * scale_factor).tolist(),
		}

	selected_axis = str(grid_interval.get("axis", "x"))
	primary = intervals.get(selected_axis, grid_interval)
	return {
		**grid_interval,
		"scale": float(grid_interval.get("scale", 1.0) * scale_factor),
		"interval_1d": primary.get("interval_1d", grid_interval.get("interval_1d", [0.0, 0.0])),
		"start_point": primary.get("start_point", grid_interval.get("start_point")),
		"end_point": primary.get("end_point", grid_interval.get("end_point")),
		"intervals": intervals,
	}


def _estimate_vertical_normal_y_shift(
	points: np.ndarray,
	normals: np.ndarray | None,
	vertical_normal_threshold: float = 0.85,
	min_points: int = 32,
	bin_count: int = 128,
) -> tuple[float, dict]:
	if normals is None or len(normals) == 0:
		return float(points[:, 1].min()), {
			"source": "aabb_fallback",
			"reason": "normals_unavailable",
		}

	normals = np.asarray(normals, dtype=np.float64)
	points = np.asarray(points, dtype=np.float64)
	vertical_mask = np.abs(normals[:, 1]) >= float(vertical_normal_threshold)
	vertical_points = points[vertical_mask]

	if len(vertical_points) < min_points:
		return float(points[:, 1].min()), {
			"source": "aabb_fallback",
			"reason": "too_few_vertical_points",
			"vertical_point_count": int(len(vertical_points)),
		}

	y_values = vertical_points[:, 1]
	bin_count = int(max(16, bin_count))
	hist, edges = np.histogram(y_values, bins=bin_count)

	peak_indices = [
		i
		for i in range(1, len(hist) - 1)
		if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]
	]
	if not peak_indices:
		peak_indices = [int(np.argmax(hist))]

	peak_idx = int(max(peak_indices, key=lambda i: hist[i]))
	peak_y = float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))

	return peak_y, {
		"source": "vertical_normal_peak",
		"vertical_point_count": int(len(vertical_points)),
		"threshold": float(vertical_normal_threshold),
		"bin_count": int(bin_count),
		"peak_y": peak_y,
		"peak_count": int(hist[peak_idx]),
	}


def _snap_points_along_normals(
	points: np.ndarray,
	normals: np.ndarray | None,
	) -> tuple[np.ndarray, dict]:
	if normals is None or len(normals) == 0:
		return np.asarray(points, dtype=np.float64).copy(), {
			"enabled": False,
			"reason": "normals_unavailable",
			"snapped_count": 0,
		}

	points = np.asarray(points, dtype=np.float64)
	normals = np.asarray(normals, dtype=np.float64)
	if len(points) != len(normals):
		return points.copy(), {
			"enabled": False,
			"reason": "point_normal_count_mismatch",
			"snapped_count": 0,
		}

	norm = np.linalg.norm(normals, axis=1, keepdims=True)
	valid = norm[:, 0] > 1e-9
	if not np.any(valid):
		return points.copy(), {
			"enabled": False,
			"reason": "invalid_normals",
			"snapped_count": 0,
		}

	unit_normals = np.zeros_like(normals)
	unit_normals[valid] = normals[valid] / norm[valid]
	unit_normals[~valid] = normals[~valid]

	signed_dist = np.sum(points * unit_normals, axis=1)
	snapped_dist = np.rint(signed_dist)
	delta = (snapped_dist - signed_dist)[:, None] * unit_normals
	snapped_points = points + delta
	return snapped_points, {
		"enabled": True,
		"reason": "ok",
		"snapped_count": int(len(points)),
		"max_offset": float(np.max(np.linalg.norm(delta, axis=1))),
	}


def _choose_alignment_offset(rotated_points: np.ndarray, grid_interval: dict) -> tuple[np.ndarray, str]:
	mins = rotated_points.min(axis=0)
	alignment_offset = mins.copy()
	alignment_mode = "aabb"

	if grid_interval.get("success", False):
		intervals = grid_interval.get("intervals", {})
		x_interval = intervals.get("x", {})
		z_interval = intervals.get("z", {})
		if x_interval.get("success", False):
			alignment_offset[0] = float(x_interval["interval_1d"][0])
		if z_interval.get("success", False):
			alignment_offset[2] = float(z_interval["interval_1d"][0])
		alignment_mode = "grid_offset"

	return alignment_offset, alignment_mode


def _log_alignment_summary(
	method: str,
	alignment_mode: str,
	alignment_offset: np.ndarray,
	y_shift_info: dict,
	unit_scale_factor: float,
	scale_info: dict,
	grid_interval: dict,
	axis_guide: dict,
	config: AlignmentConfig,
) -> None:
	print(f"[OK] Alignment method: {method}")
	print(f"[OK] Scaled to unit space with factor = {unit_scale_factor:.6f}")
	if y_shift_info.get("source") == "vertical_normal_peak":
		print(
			f"[OK] Shifted all points up by vertical-normal peak y = {alignment_offset[1]:.6f} "
			f"(count={y_shift_info.get('vertical_point_count', 0)})"
		)
	else:
		print(
			f"[WARN] Vertical-normal peak unavailable; fell back to AABB y = {alignment_offset[1]:.6f} "
			f"({y_shift_info.get('reason', 'unknown')})"
		)
	if alignment_mode == "grid_offset":
		print(
			f"[OK] Alignment offset recovered from grid scale: x={alignment_offset[0]:.6f}, "
			f"y={alignment_offset[1]:.6f}, z={alignment_offset[2]:.6f}"
		)
	else:
		print(
			f"[OK] Alignment offset from AABB min: x={alignment_offset[0]:.6f}, "
			f"y={alignment_offset[1]:.6f}, z={alignment_offset[2]:.6f}"
		)
	if scale_info.get("success", False):
		scale = scale_info["estimated_block_scale"]
		conf = scale_info["confidence"]
		print(f"[OK] Estimated block scale (autocorr): {scale:.6f} (confidence={conf:.3f})")
	else:
		print("[WARN] Scale estimation failed.")
	if grid_interval.get("success", False):
		start, end = grid_interval["interval_1d"]
		axis_name = grid_interval["axis"]
		print(f"[OK] Recovered one-grid interval on {axis_name}: start={start:.6f}, end={end:.6f}")
	else:
		print(f"[WARN] Grid interval recovery failed: {grid_interval.get('reason', 'unknown')}")
	if axis_guide.get("added", False):
		print(f"[OK] Added purple axis guide to x={axis_guide['line_end'][0]:.6f} using N={config.axis_guide_n}.")
	else:
		print(f"[WARN] Axis guide not added: {axis_guide.get('reason', 'unknown')}")


def append_scale_axis_guide_points(
	pcd: o3d.geometry.PointCloud,
	scale_info: dict,
	guide_n: int,
	guide_subdivisions: int,
	color_rgb: tuple[float, float, float],
	base_colors: np.ndarray | None = None,
) -> tuple[o3d.geometry.PointCloud, dict]:
	guide_meta = {
		"enabled": guide_n > 0,
		"added": False,
		"reason": None,
		"guide_n": int(guide_n),
		"guide_subdivisions": int(guide_subdivisions),
	}

	if guide_n <= 0:
		guide_meta["reason"] = "disabled"
		return pcd, guide_meta

	if not scale_info.get("success", False):
		guide_meta["reason"] = "scale_unavailable"
		return pcd, guide_meta

	scale = float(scale_info.get("estimated_block_scale", 0.0))
	if scale <= 1e-9:
		guide_meta["reason"] = "invalid_scale"
		return pcd, guide_meta

	steps_per_block = max(1, int(guide_subdivisions))
	n_samples = int(guide_n * steps_per_block + 1)
	x_values = np.linspace(0.0, guide_n * scale, n_samples, dtype=np.float64)
	guide_points = np.column_stack([
		x_values,
		np.zeros_like(x_values),
		np.zeros_like(x_values),
	])

	original_points = np.asarray(pcd.points)
	original_colors = None
	if base_colors is not None:
		original_colors = np.asarray(base_colors, dtype=np.float64)
	elif pcd.has_colors():
		original_colors = np.asarray(pcd.colors, dtype=np.float64)

	all_points = np.vstack([original_points, guide_points])
	pcd.points = o3d.utility.Vector3dVector(all_points)

	if original_colors is not None and len(original_colors) == len(original_points):
		guide_color = np.mean(original_colors, axis=0, keepdims=True)
		guide_colors = np.tile(np.clip(guide_color, 0.0, 1.0), (n_samples, 1))
		all_colors = np.vstack([original_colors, guide_colors])
	else:
		guide_colors = np.tile(np.asarray(color_rgb, dtype=np.float64), (n_samples, 1))
		default_color = np.full((original_points.shape[0], 3), 0.75, dtype=np.float64)
		all_colors = np.vstack([default_color, guide_colors])
	pcd.colors = o3d.utility.Vector3dVector(all_colors)

	guide_meta.update(
		{
			"added": True,
			"reason": "ok",
			"scale_used": scale,
			"line_end": [float(guide_n * scale), 0.0, 0.0],
			"point_count_added": int(n_samples),
			"color_rgb": [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])],
		}
	)
	return pcd, guide_meta


def create_purple_guide_line(scale_info: dict, guide_n: int) -> o3d.geometry.LineSet | None:
	if guide_n <= 0 or not scale_info.get("success", False):
		return None

	scale = float(scale_info.get("estimated_block_scale", 0.0))
	if scale <= 1e-9:
		return None

	extent = float(guide_n * scale)
	grid_points: list[list[float]] = []
	grid_lines: list[list[int]] = []
	grid_colors: list[list[float]] = []
	color = [0.7, 0.0, 1.0]

	def add_segment(p0: tuple[float, float, float], p1: tuple[float, float, float]) -> None:
		start_idx = len(grid_points)
		grid_points.append([float(p0[0]), float(p0[1]), float(p0[2])])
		grid_points.append([float(p1[0]), float(p1[1]), float(p1[2])])
		grid_lines.append([start_idx, start_idx + 1])
		grid_colors.append(color)

	for i in range(guide_n + 1):
		pos = float(i * scale)
		add_segment((0.0, 0.0, pos), (extent, 0.0, pos))
		add_segment((pos, 0.0, 0.0), (pos, 0.0, extent))

	line_set = o3d.geometry.LineSet()
	line_set.points = o3d.utility.Vector3dVector(np.asarray(grid_points, dtype=np.float64))
	line_set.lines = o3d.utility.Vector2iVector(np.asarray(grid_lines, dtype=np.int32))
	line_set.colors = o3d.utility.Vector3dVector(np.asarray(grid_colors, dtype=np.float64))
	return line_set


def visualize_alignment(aligned_pcd: o3d.geometry.PointCloud, scale_info: dict, guide_n: int) -> None:
	geometries: list[object] = [aligned_pcd]
	guide_line = create_purple_guide_line(scale_info, guide_n)
	if guide_line is not None:
		geometries.append(guide_line)
	o3d.visualization.draw_geometries(
		geometries,
		window_name="Siftmatica Alignment Preview",
		point_show_normal=False,
	)


def _draw_repeated_grid_interval(
	ax: plt.Axes,
	axis_key: str,
	axis_interval: dict,
	scale: float,
	coord_min: float,
	coord_max: float,
	interval_y: float,
	ymax: float,
) -> None:
	if not (isinstance(axis_interval, dict) and axis_interval.get("success", False)):
		return

	start, end = float(axis_interval["interval_1d"][0]), float(axis_interval["interval_1d"][1])
	k_min = int(np.floor((coord_min - end) / scale)) - 1
	k_max = int(np.ceil((coord_max - start) / scale)) + 1
	for k in range(k_min, k_max + 1):
		rep_start = start + k * scale
		rep_end = end + k * scale
		if rep_end < coord_min or rep_start > coord_max:
			continue
		draw_start = max(rep_start, coord_min)
		draw_end = min(rep_end, coord_max)
		ax.hlines(interval_y, draw_start, draw_end, colors="limegreen", linewidth=3.5, alpha=0.9)
		ax.vlines(rep_start, interval_y - 0.03 * ymax, interval_y + 0.03 * ymax, colors="limegreen", linewidth=2.2)
		ax.vlines(rep_end, interval_y - 0.03 * ymax, interval_y + 0.03 * ymax, colors="limegreen", linewidth=2.2)

	ax.vlines([start, end], interval_y - 0.03 * ymax, interval_y + 0.03 * ymax, colors="red", linewidth=2.2)
	ax.text(
		0.5 * (start + end),
		interval_y - 0.05 * ymax,
		f"{axis_key}-grid [{start:.4f}, {end:.4f}] (repeated)",
		color="limegreen",
		ha="center",
		va="top",
	)


def _draw_repeated_scale_vlines(ax: plt.Axes, scale: float, coord_min: float, coord_max: float) -> None:
	v_min = int(np.floor(coord_min / scale)) - 1
	v_max = int(np.ceil(coord_max / scale)) + 1
	for k in range(v_min, v_max + 1):
		xv = k * scale
		if coord_min <= xv <= coord_max:
			ax.axvline(xv, color="purple", linestyle="--", alpha=0.20, linewidth=1)


def plot_projection_comparison(
	aligned_points: np.ndarray,
	aligned_colors: np.ndarray | None,
	scale_info: dict,
	grid_interval: dict,
	projection_bins: int,
	output_path: Path | None,
	show: bool,
) -> None:
	if not scale_info.get("success", False):
		return

	scale = float(scale_info.get("estimated_block_scale", 0.0))
	if scale <= 1e-9:
		return

	projection_bins = int(max(32, projection_bins))
	fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
	projection_specs = [
		("X projection", aligned_points[:, 0], axes[0]),
		("Z projection", aligned_points[:, 2], axes[1]),
	]
	intervals = grid_interval.get("intervals", {})

	rng = np.random.default_rng(42)
	for title, coords, ax in projection_specs:
		axis_key = "x" if title.startswith("X") else "z"
		axis_interval = intervals.get(axis_key, None)
		coords = np.asarray(coords, dtype=np.float64)
		coord_min = float(coords.min())
		coord_max = float(coords.max())
		y_jitter = rng.normal(loc=0.0, scale=0.035, size=coords.shape[0])
		if aligned_colors is not None and len(aligned_colors) == len(coords):
			scatter_colors = np.clip(np.asarray(aligned_colors, dtype=np.float64), 0.0, 1.0)
		else:
			scatter_colors = np.tile(np.array([[0.2, 0.5, 0.9]], dtype=np.float64), (len(coords), 1))
		ax.scatter(coords, y_jitter, s=4, c=scatter_colors, alpha=0.55, edgecolors="none")
		ax.set_title(title)
		ax.set_xlabel("Position")
		ax.set_ylabel("Jittered points")
		ax.grid(True, alpha=0.25)
		ymax = max(float(np.max(np.abs(y_jitter))) * 1.6, 0.08)
		scale_y = -0.18 * ymax
		interval_y = scale_y - 0.10 * ymax
		ax.hlines(scale_y, 0.0, scale, colors="purple", linewidth=4)
		ax.text(scale * 0.5, scale_y - 0.06 * ymax, f"S = {scale:.4f}", color="purple", ha="center", va="top")
		_draw_repeated_scale_vlines(ax=ax, scale=scale, coord_min=coord_min, coord_max=coord_max)
		_draw_repeated_grid_interval(
			ax=ax,
			axis_key=axis_key,
			axis_interval=axis_interval,
			scale=scale,
			coord_min=coord_min,
			coord_max=coord_max,
			interval_y=interval_y,
			ymax=ymax,
		)
		ax.set_ylim(scale_y - 0.28 * ymax, ymax)
		ax.set_xlim(coord_min, coord_max)

	bar_ax = axes[2]
	bar_ax.set_axis_off()
	bar_ax.set_xlim(0.0, scale)
	bar_ax.set_ylim(0.0, 1.0)
	bar_ax.hlines(0.5, 0.0, scale, colors="purple", linewidth=6)
	bar_ax.text(scale * 0.5, 0.72, f"Estimated block scale S = {scale:.4f}", color="purple", ha="center", va="bottom", fontsize=12)
	bar_ax.text(0.0, 0.18, "0", ha="center", va="center")
	bar_ax.text(scale, 0.18, "S", ha="center", va="center")
	bar_ax.set_title("Scale bar underneath the projections")

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=180)
		print(f"[OK] Saved projection comparison plot: {output_path}")

	if show:
		plt.show()
	else:
		plt.close(fig)


def estimate_pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	centered = points - points.mean(axis=0, keepdims=True)
	covariance = np.cov(centered, rowvar=False)
	eigvals, eigvecs = np.linalg.eigh(covariance)
	order = np.argsort(eigvals)[::-1]
	eigvals = eigvals[order]
	eigvecs = eigvecs[:, order]

	x_axis = eigvecs[:, 0]
	y_axis = eigvecs[:, 2]
	z_axis = np.cross(x_axis, y_axis)
	x_axis, y_axis, z_axis = _ensure_right_handed(x_axis, y_axis, z_axis)
	basis = np.column_stack([x_axis, y_axis, z_axis])
	return basis, eigvals


def detect_plane_normals(
	pcd: o3d.geometry.PointCloud,
	max_planes: int = 6,
	distance_threshold: float = 0.03,
	ransac_n: int = 3,
	num_iterations: int = 2000,
	min_inlier_ratio: float = 0.02,
) -> list[tuple[np.ndarray, int]]:
	normals: list[tuple[np.ndarray, int]] = []
	remaining = pcd
	total_points = len(np.asarray(pcd.points))

	for _ in range(max_planes):
		if len(np.asarray(remaining.points)) < max(ransac_n * 3, 100):
			break

		model, inliers = remaining.segment_plane(
			distance_threshold=distance_threshold,
			ransac_n=ransac_n,
			num_iterations=num_iterations,
		)
		inlier_count = len(inliers)
		if inlier_count < int(total_points * min_inlier_ratio):
			break

		normal = _normalize(np.asarray(model[:3], dtype=np.float64))
		normals.append((normal, inlier_count))
		remaining = remaining.select_by_index(inliers, invert=True)

	return normals


def estimate_manhattan_basis(points: np.ndarray, normals_with_weights: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, str]:
	pca_basis, _ = estimate_pca_axes(points)
	up_hint = pca_basis[:, 1]

	if len(normals_with_weights) < 2:
		return pca_basis, "pca_fallback"

	normals, weights = zip(*normals_with_weights)
	normals = list(normals)
	weights = list(weights)

	y_axis = _best_vertical_normal(normals, up_hint)

	z_candidates: list[tuple[float, np.ndarray]] = []
	for idx, normal in enumerate(normals):
		if abs(float(np.dot(normal, y_axis))) > 0.90:
			continue
		orthogonality = 1.0 - abs(np.dot(normal, y_axis))
		if orthogonality < 0.15:
			continue
		projected = _project_onto_plane(normal, y_axis)
		proj_norm = np.linalg.norm(projected)
		if proj_norm < 1e-8:
			continue
		z_axis = projected / proj_norm
		score = weights[idx] * orthogonality
		z_candidates.append((score, z_axis))

	if z_candidates:
		z_axis = max(z_candidates, key=lambda item: item[0])[1]
	else:
		pca_front = pca_basis[:, 2]
		z_axis = _project_onto_plane(pca_front, y_axis)
		z_axis = _normalize(z_axis)

	if np.dot(z_axis, pca_basis[:, 2]) < 0:
		z_axis = -z_axis

	x_axis = np.cross(y_axis, z_axis)
	x_axis = _normalize(x_axis)
	z_axis = _normalize(np.cross(x_axis, y_axis))
	x_axis, y_axis, z_axis = _ensure_right_handed(x_axis, y_axis, z_axis)
	basis = np.column_stack([x_axis, y_axis, z_axis])
	return basis, "plane_ransac_y_up"

def align_point_cloud(
	input_path: Path,
	output_path: Path,
	report_path: Path,
	config: AlignmentConfig,
	visualize: bool,
) -> None:
    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"No points found in input cloud: {input_path}")

    if config.voxel_downsample > 0:
        sampled = pcd.voxel_down_sample(voxel_size=config.voxel_downsample)
    else:
        sampled = pcd
    sampled_points = np.asarray(sampled.points)

    normals = detect_plane_normals(
		sampled,
		max_planes=config.max_planes,
		distance_threshold=config.plane_distance,
		num_iterations=config.ransac_iterations,
	)
    basis, method = estimate_manhattan_basis(sampled_points, normals)

    rotation = basis.T
    rotated_points = (rotation @ points.T).T
    rotated_normals = None
    if pcd.has_normals():
        original_normals = np.asarray(pcd.normals)
        rotated_normals = (rotation @ original_normals.T).T

    y_shift, y_shift_info = _estimate_vertical_normal_y_shift(rotated_points, rotated_normals)
    scale_info = estimate_scale_autocorr(
		rotated_points,
		bins=config.scale_bins,
		scale_min=config.scale_min,
		scale_max=config.scale_max,
		smooth_window=config.scale_smooth_window,
	)
    grid_interval = recover_grid_interval(rotated_points, scale_info)
    alignment_offset, alignment_mode = _choose_alignment_offset(
        rotated_points, grid_interval
    )
    alignment_offset[1] = y_shift

    aligned_points = rotated_points - alignment_offset
    unit_scale = (
        float(scale_info.get("estimated_block_scale", 1.0))
        if scale_info.get("success", False)
        else 1.0
    )
    unit_scale = unit_scale if unit_scale > 1e-9 else 1.0
    scale_factor = 1.0 / unit_scale
    unit_scale_info = dict(scale_info)
    if unit_scale_info.get("success", False):
        unit_scale_info["estimated_block_scale"] = 1.0
        unit_scale_info["unit_scale_factor"] = scale_factor
    unit_grid_interval = _scale_grid_interval(grid_interval, scale_factor)
    # snapped_to_surface = False
    # snap_info = {"enabled": False, "reason": "not_attempted", "snapped_count": 0}
    # if scale_info.get("success", False):
    #     aligned_points = aligned_points * scale_factor
    #     aligned_points, snap_info = _snap_points_along_normals(
    #         aligned_points, rotated_normals
    #     )
    #     snapped_to_surface = bool(snap_info.get("enabled", False))

    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)
    aligned_colors = None

    if pcd.has_colors():
        aligned_colors = np.asarray(pcd.colors, dtype=np.float64).copy()
        aligned_pcd.colors = o3d.utility.Vector3dVector(aligned_colors)
    if pcd.has_normals():
        aligned_pcd.normals = o3d.utility.Vector3dVector(rotated_normals)

    aligned_pcd, axis_guide = append_scale_axis_guide_points(
        aligned_pcd,
        scale_info=unit_scale_info,
        guide_n=config.axis_guide_n,
        guide_subdivisions=config.axis_guide_subdivisions,
        color_rgb=(0.7, 0.0, 1.0),
        base_colors=aligned_colors,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), aligned_pcd)

    transform = {
		"method": method,
		"input": str(input_path),
		"output": str(output_path),
		"rotation_matrix": rotation.tolist(),
		"translation_after_rotation": (-alignment_offset).tolist(),
		"alignment_mode": alignment_mode,
		"alignment_offset": alignment_offset.tolist(),
		"y_shift": y_shift,
		"y_shift_info": y_shift_info,
		"unit_scale_factor": scale_factor,
		# "snapped_to_surface": snapped_to_surface,
		# "snap_info": snap_info,
		"aabb_min_after_alignment": aligned_points.min(axis=0).tolist(),
		"aabb_max_after_alignment": aligned_points.max(axis=0).tolist(),
		"scale_estimation": scale_info,
		"grid_interval": grid_interval,
		"unit_grid_interval": unit_grid_interval,
		"axis_guide": axis_guide,
		"detected_planes": [
			{"normal": normal.tolist(), "inliers": int(weight)} for normal, weight in normals
		],
	}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(transform, indent=2), encoding="utf-8")

    _log_alignment_summary(
		method=method,
		alignment_mode=alignment_mode,
		alignment_offset=alignment_offset,
		y_shift_info=y_shift_info,
		unit_scale_factor=scale_factor,
		scale_info=scale_info,
		grid_interval=grid_interval,
		axis_guide=axis_guide,
		config=config,
	)
    if config.plot_projections:
        plot_projection_comparison(
			aligned_points=aligned_points,
			aligned_colors=aligned_colors,
			scale_info=unit_scale_info,
			grid_interval=unit_grid_interval,
			projection_bins=config.projection_bins,
			output_path=config.projection_plot_path,
			show=True,
		)
    if visualize:
        visualize_alignment(aligned_pcd, scale_info=unit_scale_info, guide_n=config.axis_guide_n)
    print(f"[OK] Saved aligned cloud: {output_path}")
    print(f"[OK] Saved transform report: {report_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Align a COLMAP dense point cloud to Manhattan axes and translate by AABB min or recovered grid offset."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("data/house1-dense/workspace/dense/0/fused.ply"),
		help="Path to input PLY point cloud.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/house1-dense/workspace/dense/0/fused_aligned.ply"),
		help="Path for aligned output PLY.",
	)
	parser.add_argument(
		"--report",
		type=Path,
		default=Path("data/house1-dense/workspace/dense/0/fused_alignment.json"),
		help="Path for JSON report containing transform metadata.",
	)
	parser.add_argument(
		"--voxel",
		type=float,
		default=0.05,
		help="Voxel size for downsampling before plane detection (set <=0 to disable).",
	)
	parser.add_argument(
		"--max-planes",
		type=int,
		default=6,
		help="Maximum number of dominant planes to detect via RANSAC.",
	)
	parser.add_argument(
		"--plane-distance",
		type=float,
		default=0.03,
		help="Distance threshold for plane RANSAC.",
	)
	parser.add_argument(
		"--ransac-iters",
		type=int,
		default=2000,
		help="RANSAC iterations for each plane.",
	)
	parser.add_argument(
		"--scale-bins",
		type=int,
		default=4096,
		help="Number of bins for 1D density signal in autocorrelation scale estimation.",
	)
	parser.add_argument(
		"--scale-min",
		type=float,
		default=0.7,
		help="Minimum candidate block scale (same units as point cloud).",
	)
	parser.add_argument(
		"--scale-max",
		type=float,
		default=1.4,
		help="Maximum candidate block scale (same units as point cloud).",
	)
	parser.add_argument(
		"--scale-smooth-window",
		type=int,
		default=5,
		help="Moving-average window (in bins) before autocorrelation.",
	)
	parser.add_argument(
		"--axis-guide-n",
		type=int,
		default=10,
		help="Add purple guide points from (0,0,0) to (N*S,0,0); set 0 to disable.",
	)
	parser.add_argument(
		"--axis-guide-subdivisions",
		type=int,
		default=20,
		help="Number of guide points per block interval S.",
	)
	parser.add_argument(
		"--visualize",
		action="store_false",
		help="Open an Open3D viewer showing the aligned point cloud and purple guide line.",
	)
	parser.add_argument(
		"--plot-projections",
		action="store_false",
		help="Display and save a Matplotlib comparison of X/Z projections with the estimated scale bar.",
	)
	parser.add_argument(
		"--projection-bins",
		type=int,
		default=256,
		help="Reserved for projection plot sampling; kept for compatibility.",
	)
	parser.add_argument(
		"--projection-plot-path",
		type=Path,
		default=Path("data/house1-dense/workspace/dense/0/fused_projection_compare.png"),
		help="Where to save the projection comparison figure.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path = args.input
	if not input_path.exists():
		raise FileNotFoundError(f"Input cloud not found: {input_path}")
	config = AlignmentConfig(
		voxel_downsample=args.voxel,
		max_planes=args.max_planes,
		plane_distance=args.plane_distance,
		ransac_iterations=args.ransac_iters,
		scale_bins=args.scale_bins,
		scale_min=args.scale_min,
		scale_max=args.scale_max,
		scale_smooth_window=args.scale_smooth_window,
		axis_guide_n=args.axis_guide_n,
		axis_guide_subdivisions=args.axis_guide_subdivisions,
		projection_bins=args.projection_bins,
		projection_plot_path=args.projection_plot_path,
		plot_projections=args.plot_projections,
	)

	align_point_cloud(
		input_path=input_path,
		output_path=args.output,
		report_path=args.report,
		config=config,
		visualize=args.visualize,
	)


if __name__ == "__main__":
	main()
