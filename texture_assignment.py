import open3d as o3d
from pathlib import Path
from src.model import model
from src.pipeline.alignment import PointCloudAlignment
from src.model.model import MinecraftTextureClassifier
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

def points_to_image(points, colors, face, size=16):
    if colors.max() <= 1.0:
        colors = colors * 255.0

    proj = []
    if face in ['+x', '-x']:
        proj =  points[:, [1, 2]] 
    elif face in ['+y', '-y']:
        proj =  points[:, [0, 2]] 
    else:
        proj =  points[:, [0, 1]] 
    min_xy = proj.min(axis=0)
    max_xy = proj.max(axis=0)
    norm_xy = (proj - min_xy) / (max_xy - min_xy + 1e-8)

    coords = (norm_xy * (size - 1)).astype(int)

    img = np.zeros((size, size, 3), dtype=np.float32)
    count = np.zeros((size, size), dtype=np.float32)

    for (x, y), c in zip(coords, colors):
        img[y, x] += c
        count[y, x] += 1

    mask = count > 0
    img[mask] /= count[mask][:, None]

    if np.any(count > 0):
        mean_color = img[count > 0].mean(axis=0)
        img[count == 0] = mean_color

    return img.astype(np.uint8)

def assign_textures():
    data_dir = Path("./data/house2-dense/workspace/dense/0/fused.ply")
    pcd = o3d.io.read_point_cloud(str(data_dir))
    aligner = PointCloudAlignment()
    voxel_grid = aligner.align_point_cloud(pcd)

    # o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

    with open("./src/classification/minecraft_class_names.json", 'r') as f:
        idx_to_class = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinecraftTextureClassifier(num_classes=len(idx_to_class))
    model.load_state_dict(torch.load("./src/classification/minecraft_texture_classifier.pth", map_location=device, weights_only=False))
    model.eval()

    print(f"Number of voxels: {len(voxel_grid)}")
    print("Sample voxel data:")

    faces = ['-x', '+x', '-y', '+y', '-z', '+z']

    key_texture_dict = {}
    for i, (voxel_key, voxel_data) in enumerate(voxel_grid.items()):
        outputs_list = []
        face_points = {}
        for face in faces:
            points = np.array(voxel_data[face].points)
            colors = np.asarray(voxel_data[face].colors)
            if len(points) < 5:
                continue
            img = points_to_image(points, colors, face)

            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            img_tensor = (img_tensor - mean) / std
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs.data, 1)
                if idx_to_class[predicted.item()] not in face_points:
                    face_points[idx_to_class[predicted.item()]] = len(points)
                else:
                    face_points[idx_to_class[predicted.item()]] += len(points)
                outputs_list.append(outputs)
        if len(outputs_list) == 0:
            continue
        # print(face_points)
        predicted_texture = max(face_points, key=face_points.get)
        # avg_outputs = torch.mean(torch.stack(outputs_list), dim=0)
        # _, predicted = torch.max(avg_outputs.data, 1)
        # predicted_texture = idx_to_class[predicted.item()]
        key_texture_dict[voxel_key] = predicted_texture
    return key_texture_dict

if __name__ == "__main__":
    assign_textures()