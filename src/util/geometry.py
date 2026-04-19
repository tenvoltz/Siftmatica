import numpy as np
import cv2
from typing import Optional, Tuple


def points_to_line(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (np.asarray(p1, dtype=np.float32), np.asarray(p2, dtype=np.float32))

def line_intersection(line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
    vp = np.cross(line1, line2)
    if abs(vp[2]) < 1e-6:
        return None
    return vp / vp[2]

def image_to_homogeneous(point: Tuple[float, float]) -> np.ndarray:
    """Convert image coordinates to homogeneous coordinates."""
    return np.array([point[0], point[1], 1.0], dtype=np.float32)

def homogeneous_to_image(homo_point: np.ndarray) -> np.ndarray:
    """Convert homogeneous coordinates to image coordinates."""
    return homo_point[:2] if len(homo_point) >= 2 else homo_point

def get_homogeneous_line(line: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Convert a line to homogeneous line format."""
    p1, p2 = line
    return np.cross(image_to_homogeneous(p1), image_to_homogeneous(p2))

def shift_point(p: Tuple[float, float], offset: Tuple[float, float]) -> Tuple[int, int]:
    """Shift a point by an offset."""
    p = np.asarray(p).flatten()
    return (int(p[0] - offset[0]), int(p[1] - offset[1]))

def line_to_endpoints(line: Tuple[np.ndarray, np.ndarray], canvas_h: int, canvas_w: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Extend a line to canvas bounds and return endpoints."""
    p1, p2 = line
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    
    # Handle degenerate case
    if np.linalg.norm(p2 - p1) < 1e-6:
        return (int(p1[0]), int(p1[1])), (int(p1[0]), int(p1[1]))
    
    # Direction vector
    direction = p2 - p1
    direction = direction / np.linalg.norm(direction)
    
    # Parametric form: point = p1 + t * direction
    # Find t values where line intersects canvas bounds
    t_min, t_max = -1e6, 1e6
    
    # Check x bounds
    if abs(direction[0]) > 1e-8:
        t_x0 = -p1[0] / direction[0]
        t_x1 = (canvas_w - p1[0]) / direction[0]
        t_min = max(t_min, min(t_x0, t_x1))
        t_max = min(t_max, max(t_x0, t_x1))
    
    # Check y bounds
    if abs(direction[1]) > 1e-8:
        t_y0 = -p1[1] / direction[1]
        t_y1 = (canvas_h - p1[1]) / direction[1]
        t_min = max(t_min, min(t_y0, t_y1))
        t_max = min(t_max, max(t_y0, t_y1))
    
    # Compute endpoints
    p_min = p1 + t_min * direction
    p_max = p1 + t_max * direction
    
    # Clamp to canvas bounds
    p_min = np.clip(p_min, [0, 0], [canvas_w - 1, canvas_h - 1])
    p_max = np.clip(p_max, [0, 0], [canvas_w - 1, canvas_h - 1])
    
    return (int(p_min[0]), int(p_min[1])), (int(p_max[0]), int(p_max[1]))


def draw_infinite_line(
    canvas: np.ndarray,
    p: Tuple[int, int],
    vp: Tuple[int, int],
    color: Tuple[int, int, int],
    length: int = 20000
) -> None:
    """Draw a line from p extending towards vp."""
    px, py = p
    vx, vy = vp
    dx, dy = vx - px, vy - py
    norm = np.hypot(dx, dy)
    if norm < 1e-6:
        return
    dx, dy = dx / norm, dy / norm
    cv2.line(
        canvas,
        (int(px - dx * length), int(py - dy * length)),
        (int(px + dx * length), int(py + dy * length)),
        color,
        1
    )
