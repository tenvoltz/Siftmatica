import os
import cv2
import numpy as np
from env import OUTPUT_FOLDER
from src.util.logger import get_logger
from src.config import PipelineConfig
from src.data.camera import MinecraftCamera
from src.pipeline.orchestrator import PipelineBuilder
from src.pipeline.line_extraction import LineExtractionFactory
from src.pipeline.vanishing_points import VanishingPointEstimator
from src.util.visualization import Visualizer
from src.util.perf import PerfContext

logger = get_logger(__name__)


if __name__ == "__main__":
    dirs_to_create = ["edges", "contours", "hough", "histograms", "vps"]
    for d in dirs_to_create:
        os.makedirs(os.path.join(OUTPUT_FOLDER, d), exist_ok=True)

    configs = [
        # PipelineConfig(preprocessors=[], detector="canny", postprocessors=[]), # Really noisy, not useful
        PipelineConfig(preprocessors=["gaussian"], detector="canny", postprocessors=[]),
        PipelineConfig(preprocessors=["bilateral"], detector="canny", postprocessors=[]),
        # PipelineConfig(preprocessors=[], detector="sobel", postprocessors=[]), # Too little edges, not useful
        # PipelineConfig(preprocessors=["gaussian"], detector="sobel", postprocessors=[]), # Too little edges, not useful
        # PipelineConfig(preprocessors=["bilateral"], detector="sobel", postprocessors=[]), # Too little edges, not useful
    ]

    # img_path = "data/house1/images/2026-03-23_19.31.05.png" # 2-point perspective, should find 2 VPs
    img_path = "data/house1/images/2026-03-23_19.30.49.png" # 1-point perspective, should find 1 VPs
    # img_path = "data/elven-house/images/2026-04-01_23.44.16.png"

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    logger.validate("image_loaded", img is not None, f"Shape: {img.shape}")
    logger.trace("initialization", "Starting pipeline", {"image_path": img_path})

    for cfg_idx, cfg in enumerate(configs, 1):
        with PerfContext(f"config_{cfg_idx}_total", logger):
            logger.trace("config", f"Processing pipeline {cfg_idx}/{len(configs)}", cfg.to_dict())
            cfg_name = PipelineBuilder.config_to_name(cfg)

            with PerfContext("camera_init", logger):
                camera = MinecraftCamera(width=img.shape[1], height=img.shape[0], fov=cfg.camera_fov)
                K = camera.get_intrinsic_matrix()
                logger.trace("camera", "Initialized MinecraftCamera", camera.get_parameters())

            with PerfContext("edge_detection", logger):
                builder = PipelineBuilder()
                pipeline = builder.build(cfg)
                edges = pipeline.run(img)
                kernel = np.ones((3, 3), np.uint8)
                edges_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                logger.trace("edge_detection", "Detected edges", {"shape": edges_clean.shape, "nonzero": np.count_nonzero(edges_clean)})
            Visualizer.plot_filtered_img(edges_clean, os.path.join(OUTPUT_FOLDER, "edges", f"{cfg_name}_edges.png"), f"{cfg_name} - Edges")

            if cfg.line_extractor == "contour":
                with PerfContext("contour_line_extraction", logger):
                    extractor = LineExtractionFactory.create("contour", logger_instance=logger, **cfg.to_dict())
                    contours = extractor.extract_contours(edges_clean, min_area=100)
                    logger.trace("contours", f"Found {len(contours)} contours")
                    contour_filters = {
                        "min_vertices": cfg.min_vertices,
                        "require_convex": cfg.require_convex,
                        "require_quadrilateral": cfg.require_quadrilateral,
                        "enable_min_area": cfg.enable_min_area,
                        "min_area": cfg.min_area,
                        "enable_min_aspect_ratio": cfg.enable_min_aspect_ratio,
                        "min_aspect_ratio": cfg.min_aspect_ratio
                    }
                    lines, line_metadata = extractor.extract_lines([c.reshape(-1, 1, 2) for c in contours])
            else:
                with PerfContext("hough_line_extraction", logger):
                    hough_extractor = LineExtractionFactory.create("hough", logger_instance=logger, **cfg.to_dict())
                    lines, line_metadata = hough_extractor.extract_lines(edges)
                    logger.trace("hough_lines", f"Found {len(lines)} Hough lines")

            if cfg.remove_horizontal_vertical:
                filtered_lines = []
                for line in lines:
                    x1, y1 = line[0]
                    x2, y2 = line[1]
                    angle = np.arctan2(y2 - y1, x2 - x1)
                    threshold_rad = np.deg2rad(cfg.removal_angle_threshold_deg)
                    if not (abs(angle) < threshold_rad or abs(angle - np.pi/2) < threshold_rad or abs(angle + np.pi/2) < threshold_rad):
                        filtered_lines.append(line)
                logger.trace("line_filtering", f"Removed horizontal/vertical lines, {len(filtered_lines)} remain", {"removed": len(lines) - len(filtered_lines)})
                lines = filtered_lines
                
            if cfg.line_extractor == "contour":
                Visualizer.plot_contours(img, contours, os.path.join(OUTPUT_FOLDER, "contours", f"{cfg_name}_contours.png"), f"{cfg_name} - Contours")
                logger.trace("visualization", "Plotted contours", {"output": f"{cfg_name}_contours.png"})
            else:
                Visualizer.plot_hough_lines(edges, lines, os.path.join(OUTPUT_FOLDER, "hough", f"{cfg_name}_hough_lines.png"), f"{cfg_name} - Hough Lines")
                logger.trace("visualization", "Plotted Hough lines", {"output": f"{cfg_name}_hough_lines.png"})
                

            with PerfContext("edge_visualization", logger):
                Visualizer.plot_polar_histogram(
                        lines,
                        os.path.join(OUTPUT_FOLDER, "histograms", f"{cfg_name}_edge_direction.png"),
                        f"{cfg_name} - Edge Direction Distribution"
                    )
                logger.trace("visualization", "Plotted edge directions", {"output": f"{cfg_name}_edge_direction.png"})

            with PerfContext("vanishing_point_extraction", logger):
                vp_est = VanishingPointEstimator(inlier_decision_method=cfg.inlier_decision_method, 
                                                angle_threshold=cfg.angle_threshold, 
                                                distance_threshold=cfg.distance_threshold)
                vps, inlier_line_indices_list = vp_est.extract_multiple_vps(lines=lines, iterations=cfg.ransac_iterations, vp_distance_threshold=cfg.vp_distance_threshold)
                if vps:
                    logger.validate("vanishing_points", True, f"Found {len(vps)} vanishing points")
                    for i, vp in enumerate(vps):
                        d = np.linalg.inv(K) @ np.array([vp[0], vp[1], 1])
                        d = d / np.linalg.norm(d)
                        logger.trace("vp_detail", f"VP {i+1}", {"vp": vp[:2].tolist(), "direction": d.tolist(), "inlier_count": len(inlier_line_indices_list[i])})
                else:
                    logger.validate("vanishing_points", False, "No vanishing points found")

            filtered_vps, filtered_inlier_line_indices_list = vp_est.filter_vps(vps, inlier_line_indices_list, min_inliers=cfg.vp_inlier_amount_threshold)
            logger.trace("vp_filtering", f"Filtered VPs with at least {cfg.vp_inlier_amount_threshold} inliers", {"initial_count": len(vps), "filtered_count": len(filtered_vps)})
            Visualizer.plot_vanishing_points(edges_clean, lines, filtered_vps, filtered_inlier_line_indices_list, os.path.join(OUTPUT_FOLDER, "vps", f"{cfg_name}_vps.png"), f"{cfg_name} - Vanishing Points", plot_line_extensions=cfg.plot_line_extensions)


    logger.validate("pipeline", True, "Pipeline execution completed successfully")
