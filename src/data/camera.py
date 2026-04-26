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

    def get_intrinsic_parameters(self) -> str:
        # Return parameters in the order expected by pycolmap SIMPLE_PINHOLE: fx, cx, cy
        return f"{self.fx}, {self.cx}, {self.cy}"

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

    def compute_extrinsics_from_vps(self, vps_image):
        if not vps_image or len(vps_image) == 0:
            return None

        K_inv = np.linalg.inv(self.K)

        dirs = []
        for vp in vps_image[:3]:  # assume Manhattan: max 3
            v = np.array([vp[0], vp[1], 1.0], dtype=np.float64)
            d = K_inv @ v
            d /= np.linalg.norm(d)
            dirs.append(d)

        if len(dirs) == 2:
            # Third direction must be perpendicular to the first two
            d3 = np.cross(dirs[0], dirs[1])
            d3 /= np.linalg.norm(d3)
            dirs.append(d3)

        # pad if fewer than 3 VPs
        while len(dirs) < 3:
            dirs.append(np.eye(3)[len(dirs)])

        D = np.column_stack(dirs)

        # SVD orthonormalization (Procrustes)
        U, _, Vt = np.linalg.svd(D)
        R = U @ Vt

        # Enforce right-handed system
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return R

    def rotation_matrix_to_euler_angles(self, R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = 0

        return yaw, pitch, roll

if __name__ == "__main__":
    camera = MinecraftCamera(width=800, height=600, fov=70)
    print(camera)
    print("Intrinsic Matrix:\n", camera.get_intrinsic_matrix())
    print("Focal Length (fx, fy):", camera.get_focal_length())
    print("Principal Point (cx, cy):", camera.get_principal_point())
    print("Horizontal FOV:", camera.get_horizontal_fov())
