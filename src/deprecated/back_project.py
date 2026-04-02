import numpy as np


def backproject_vectorized(depth, fx, fy, cx, cy):
    h, w = depth.shape

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)
    return points.reshape(-1, 3)
