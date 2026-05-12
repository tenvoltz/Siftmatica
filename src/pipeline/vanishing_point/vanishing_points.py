import numpy as np
from tqdm import tqdm
from src.util.logger import get_logger
from src.util.geometry import (
    get_homogeneous_line,
    homogeneous_to_image,
    line_intersection,
)

logger = get_logger(__name__)


class VanishingPointEstimator:
    def __init__(self, logger_instance=logger, 
                 inlier_decision_method="angle", 
                 angle_threshold=np.deg2rad(10),
                 distance_threshold=5.0):
        self.logger = logger_instance
        self.rng = np.random.default_rng(42)
        self.inlier_decision_method = inlier_decision_method
        self.angle_threshold = angle_threshold
        self.distance_threshold = distance_threshold

    def extract_multiple_vps(
        self,
        lines,
        iterations=3000,
        max_vps=3,
        vp_distance_threshold=10.0
    ):
        if lines is None or len(lines) < 3: return [], []
        homogeneous_lines = [get_homogeneous_line(line) for line in lines]
        remaining = [
            {
                "points": line,
                "line": homogeneous_lines[i],
                "orig_idx": i
            }
            for i, line in enumerate(lines)
        ]

        vps, inliers_all = [], []

        def remove_inliers(inliers):
            mask = np.ones(len(remaining), dtype=bool)
            mask[inliers] = False
            return [remaining[i] for i in np.nonzero(mask)[0]]

        while len(remaining) >= 3 and len(vps) < max_vps:
            result = self._ransac_vp(remaining, iterations)
            if result is None: break

            vp, inliers = result
            if len(inliers) < 2: break

            orig_inliers = [remaining[i]["orig_idx"] for i in inliers]
            remaining[:] = remove_inliers(inliers)

            merged = False
            for i, existing_vp in enumerate(vps):
                if self._distance_between_vps(vp, existing_vp) < vp_distance_threshold:
                    inliers_all[i] = list(set(inliers_all[i]).union(orig_inliers))
                    all_lines = [homogeneous_lines[j] for j in inliers_all[i]]
                    refined_vp = self._refine_vp(all_lines)
                    if refined_vp is not None: vps[i] = refined_vp
                    merged = True
                    break

            if merged: continue

            vps.append(vp)
            inliers_all.append(orig_inliers)

        self.logger.trace("vp_extraction", f"Found {len(vps)} VPs")
        vps_image = [homogeneous_to_image(vp) for vp in vps]
        return vps_image, inliers_all

    def _distance_between_vps(self, vp1, vp2):
        p1 = vp1[:2] / vp1[2]
        p2 = vp2[:2] / vp2[2]

        return np.linalg.norm(p1 - p2)

    def _ransac_vp(self, lines, iterations):
        best_vp, best_inliers = None, []

        for _ in tqdm(range(iterations), desc="RANSAC VP estimation"):
            idx = self.rng.choice(len(lines), 2, replace=False)
            l1 = lines[idx[0]]["line"]
            l2 = lines[idx[1]]["line"]

            vp = line_intersection(l1, l2)
            if vp is None: continue

            if self.inlier_decision_method == "angle":
                inliers = [i for i, l in enumerate(lines) if self._point_to_line_angle(vp, l["line"]) < self.angle_threshold]
            else:
                inliers = [i for i, l in enumerate(lines) if self._point_to_line_distance(vp, l["line"]) < self.distance_threshold]

            if len(inliers) > len(best_inliers):
                best_vp = vp
                best_inliers = inliers

        if best_vp is None: return None

        inlier_lines = [lines[i]["line"] for i in best_inliers]
        best_vp = self._refine_vp(inlier_lines)

        if best_vp is None: return None, []
        return best_vp, best_inliers

    def _point_to_line_angle(self, vp, line):
        raise NotImplementedError("_point_to_line_angle method needs to be implemented")

    def _point_to_line_distance(self, point: np.ndarray, line: np.ndarray) -> float:
        # For homogeneous line ax+by+c=0 and point (x,y,1)
        # distance = |ax+by+c| / sqrt(a^2+b^2)
        a, b, c = line
        x, y = point[0], point[1]
        return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

    def _refine_vp(self, homogeneous_lines):
        # Solve Lv = 0 using SVD
        L = np.asarray(homogeneous_lines, dtype=np.float64)
        if L.ndim == 1: L = L.reshape(1, -1)
        if L.shape[0] < 2: return None
        _, _, Vt = np.linalg.svd(L)

        vp = Vt[-1]
        if abs(vp[2]) > 1e-12:
            vp = vp / vp[2]

        return vp

    def filter_vps(self, vps, inliers_list, min_inliers=3):
        filtered_vps = []
        filtered_inliers_list = []

        for vp, inliers in zip(vps, inliers_list):
            if len(inliers) >= min_inliers:
                filtered_vps.append(vp)
                filtered_inliers_list.append(inliers)

        return filtered_vps, filtered_inliers_list
