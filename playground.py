import cv2
import numpy as np
import pycolmap
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

from src.pipeline.alignment import PointCloudAlignment

def main():
    data_dir = Path("data/house2-dense")
    input_path = data_dir / "workspace" / "dense" / "0" / "fused.ply"
    pcd = o3d.io.read_point_cloud(str(input_path))
    aligner = PointCloudAlignment()
    voxel_grid = aligner.align_point_cloud(pcd)
    print("Alignment completed. Voxel grid created.")
    
    print(f"Number of voxels: {len(voxel_grid)}")
    print("Sample voxel data:")
    for i, (voxel_key, voxel_data) in enumerate(voxel_grid.items()):
        print(f"Voxel {i}: Key={voxel_key}, Data Keys={list(voxel_data.keys())}")
        if i >= 5:  # Print only the first 5 voxels for brevity
            break
        

if __name__ == "__main__":
    main()
