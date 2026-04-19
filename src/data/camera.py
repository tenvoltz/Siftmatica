import numpy as np
from typing import Tuple


class MinecraftCamera:
    """
    In Minecraft, the FOV setting represents the vertical field of view in degrees.

    Attributes:
        fov (float): Vertical field of view in degrees (default: 70)
        width (int): Image width in pixels
        height (int): Image height in pixels
        fx (float): Focal length in x-axis (in pixels)
        fy (float): Focal length in y-axis (in pixels)
        cx (float): Principal point x-coordinate (in pixels)
        cy (float): Principal point y-coordinate (in pixels)
        K (np.ndarray): 3x3 camera intrinsic matrix
    """
    
    def __init__(self, width: int, height: int, fov: float = 70.0):
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive values")
        if fov <= 0:
            raise ValueError("FOV must be a positive value")
        
        self.fov = fov
        self.width = width
        self.height = height
        self._compute_intrinsics()
    
    def _compute_intrinsics(self) -> None:
        """
        The focal length is calculated using the relationship:
            f = (height / 2) / tan(fov_vertical / 2)
        
        Where fov_vertical is the vertical field of view in radians.
        """
        fov_rad = np.radians(self.fov)
        self.fy = (self.height / 2.0) / np.tan(fov_rad / 2.0)
        self.fx = self.fy
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.K = np.array([
            [self.fx,      0,  self.cx],
            [     0,  self.fy,  self.cy],
            [     0,       0,        1]
        ], dtype=np.float64)
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        return self.K.copy()
    
    def get_focal_length(self) -> Tuple[float, float]:
        return (self.fx, self.fy)
    
    def get_principal_point(self) -> Tuple[float, float]:
        return (self.cx, self.cy)
    
    def get_horizontal_fov(self) -> float:
        """
        The horizontal FOV is calculated from the vertical FOV and aspect ratio:
            tan(fov_h / 2) = (width / height) * tan(fov_v / 2)
        """
        fov_v_rad = np.radians(self.fov)
        aspect_ratio = self.width / self.height
        fov_h_rad = 2 * np.arctan(aspect_ratio * np.tan(fov_v_rad / 2.0))
        return np.degrees(fov_h_rad)
    
    def get_parameters(self) -> dict:
        return {
            'fov_vertical': self.fov,
            'fov_horizontal': self.get_horizontal_fov(),
            'width': self.width,
            'height': self.height,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'aspect_ratio': self.width / self.height
        }
    
    def __repr__(self) -> str:
        return (
            f"MinecraftCamera("
            f"width={self.width}, height={self.height}, "
            f"fov={self.fov}°, "
            f"fx={self.fx:.2f}, fy={self.fy:.2f})"
        )
