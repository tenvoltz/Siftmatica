import cv2
import open3d as o3d
import numpy as np


def visualize_pointcloud(points, image=None):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    if image is not None:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colors = img_rgb.reshape(-1, 3) / 255.0
        pc.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pc])

def visualize_depth(depth):
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    depth_colored = (depth_normalized * 255).astype(np.uint8)
    cv2.imshow("Depth", depth_colored)
    cv2.waitKey(0)