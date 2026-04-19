from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Any
import json
import numpy as np
from env import DATA_PATH, OUTPUT_FOLDER
from src.data.camera import MinecraftCamera

@dataclass
class PipelineConfig:
    # Preprocessing
    preprocessors: List[str] = field(default_factory=list)
    gaussian_ksize: int = 5
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    etf_iterations: int = 3
    etf_kernel_radius: int = 5

    # Edge Detection
    detector: str = "canny"
    canny_t1: int = 50
    canny_t2: int = 150
    sobel_threshold: int = 50
    xdog_sigma: float = 1.0
    xdog_tau: float = 10.0

    # Postprocessing
    postprocessors: List[str] = field(default_factory=list)
    pixelate_factor: int = 4

    # Line Extraction
    line_extractor: str = "hough"
    remove_horizontal_vertical: bool = True
    removal_angle_threshold_deg: float = 5.0
    # Contour Analysis
    min_vertices: int = 3
    require_convex: bool = True
    require_quadrilateral: bool = True
    enable_min_area: bool = False
    min_area: int = 10
    enable_min_aspect_ratio: bool = False
    min_aspect_ratio: float = 0.1
    # Hough Transform
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180
    hough_threshold: int = 50
    hough_min_line_length: int = 20
    hough_max_line_gap: int = 10

    # Vanishing Point
    ransac_iterations: int = 3000
    ransac_threshold: float = 5.0
    use_kmeans_clustering: bool = False
    plot_line_extensions: bool = False
    inlier_decision_method: str = "distance"  # "angle" or "distance"
    angle_threshold: float = np.deg2rad(1)
    distance_threshold: float = 5.0
    vp_distance_threshold: float = 100.0
    vp_inlier_amount_threshold: int = 40

    # Camera
    camera_fov: float = 70.0

    # I/O
    data_path: str = field(default_factory=lambda: DATA_PATH)
    output_dir: str = field(default_factory=lambda: OUTPUT_FOLDER)

    def get_camera(self, width: int, height: int) -> MinecraftCamera:
        return MinecraftCamera(width=width, height=height, fov=self.camera_fov)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path) as f: return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f: json.dump(self.to_dict(), f, indent=2)
