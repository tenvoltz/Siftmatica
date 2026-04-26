import cv2
import numpy as np
import pycolmap
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

from src.config import PipelineConfig
from src.data.camera import MinecraftCamera
from src.util.visualize_model import Model


class VoxelReconstructor:
    def __init__(self, reconstruction, image_dir, downsample_factor=4):
        self.reconstruction = reconstruction
        self.downsample_factor = downsample_factor
        self.images = self._load_and_downsample(image_dir)

    def _load_and_downsample(self, image_dir):
        images = {}
        for img_data in self.reconstruction.images.values():
            path = image_dir / img_data.name
            img = cv2.imread(str(path))
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Downsample for performance
                new_size = (
                    img.shape[1] // self.downsample_factor,
                    img.shape[0] // self.downsample_factor,
                )
                images[img_data.image_id] = cv2.resize(img, new_size)
        return images

    def reconstruct(self, voxel_size=0.1, variance_threshold=300):
        # 1. Define Bounding Box from Sparse Points
        points3d = np.array([p.xyz for p in self.reconstruction.points3D.values()])
        p_min, p_max = points3d.min(axis=0), points3d.max(axis=0)
        
        # Create grid coordinates
        x_range = np.arange(p_min[0], p_max[0], voxel_size)
        y_range = np.arange(p_min[1], p_max[1], voxel_size)
        z_range = np.arange(p_min[2], p_max[2], voxel_size)

        # 2. Setup Scaled Intrinsics
        cam_id = next(iter(self.reconstruction.cameras))
        cam = self.reconstruction.cameras[cam_id]
        # SIMPLE_PINHOLE: [f, cx, cy]
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]) / self.downsample_factor
        K[2, 2] = 1.0

        consistent_points = []
        consistent_colors = []

        # 3. Photo-Consistency Check (Looping through a subset for demo)
        # In a real scenario, use an occupancy grid or neighbor-check to optimize
        print("Starting Photo-Consistency check...")
        total_voxels = len(x_range) * len(y_range) * len(z_range)
        voxel_iterator = (
            (px, py, pz)
            for px in x_range
            for py in y_range
            for pz in z_range
        )

        for px, py, pz in tqdm(
            voxel_iterator,
            total=total_voxels,
            desc="Photo-consistency",
            unit="voxel",
        ):
            voxel_world = np.array([px, py, pz])
            colors = []

            for img_id, img_data in self.reconstruction.images.items():
                # World to Camera
                pose = img_data.cam_from_world()
                p_cam = pose.rotation.matrix() @ voxel_world + pose.translation

                if p_cam[2] <= 0:
                    continue

                # Project to pixels
                p_pix = K @ p_cam
                u, v = int(p_pix[0] / p_pix[2]), int(p_pix[1] / p_pix[2])

                img = self.images[img_id]
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    colors.append(img[v, u])

            if len(colors) >= 3:
                # Consistency: Average Variance across RGB channels
                var = np.mean(np.var(colors, axis=0))
                if var < variance_threshold:
                    consistent_points.append(voxel_world)
                    consistent_colors.append(np.mean(colors, axis=0) / 255.0)

        # 4. Return as Open3D PointCloud (the "Dense" result)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(consistent_points)
        pcd.colors = o3d.utility.Vector3dVector(consistent_colors)
        return pcd


def main():
    DATA_PATH = Path("data/house1-dense")
    image_dir = DATA_PATH / "images"
    output_path = DATA_PATH / "output"
    output_path.mkdir(exist_ok=True)
    database_path = output_path / "database.db"

    cfg = PipelineConfig()
    sample_image_path = next(image_dir.glob("*.png"))
    img = cv2.imread(str(sample_image_path))
    camera = MinecraftCamera(
        width=img.shape[1], height=img.shape[0], fov=cfg.camera_fov
    )

    # ... [Feature extraction and matching remain same] ...
    reader_options = pycolmap.ImageReaderOptions(
        camera_model="SIMPLE_PINHOLE", camera_params=camera.get_intrinsic_parameters()
    )
    pycolmap.extract_features(
        database_path,
        image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
    )
    pycolmap.match_exhaustive(database_path)

    
    reconstructions = pycolmap.incremental_mapping(
        database_path, image_dir, output_path,
    )
    reconstruction = reconstructions[0]

    # --- VOXEL DENSE RECONSTRUCTION ---
    # reconstructor = VoxelReconstructor(reconstruction, image_dir, downsample_factor=4)
    # voxel_size 1.0 = 1 Minecraft Block (assuming scale is correct)
    # dense_pcd = reconstructor.reconstruct(voxel_size=1, variance_threshold=250)

    # --- VISUALIZATION ---
    model = Model(reconstruction=reconstruction)
    model.create_window()
    model.add_points()  # Sparse COLMAP points

    # Add the new dense voxel points
    # model.visualizer.add_geometry(dense_pcd)

    model.add_cameras()
    model.show()


if __name__ == "__main__":
    main()
