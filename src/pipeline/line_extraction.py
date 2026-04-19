import cv2
import numpy as np
from abc import ABC, abstractmethod
from src.util.logger import get_logger
from src.util.geometry import points_to_line

logger = get_logger(__name__)


class LineExtractor(ABC):
    @abstractmethod
    def extract_lines(self, source):
        pass


class ContourLineExtractor(LineExtractor):
    def __init__(self, logger_instance=logger, 
                 min_vertices=3, require_convex=False, require_quadrilateral=False,
                 enable_min_area=True, min_area=100, enable_min_aspect_ratio=False, min_aspect_ratio=0.5):
        self.logger = logger_instance
        self.min_vertices = min_vertices
        self.require_convex = require_convex
        self.require_quadrilateral = require_quadrilateral
        self.enable_min_area = enable_min_area
        self.min_area = min_area
        self.enable_min_aspect_ratio = enable_min_aspect_ratio
        self.min_aspect_ratio = min_aspect_ratio

    def extract_contours(self, edges, min_area=100):
        contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [self._simplify_contours(c) for c in contours if cv2.contourArea(c) > min_area]
        self.logger.trace("contour_extraction", f"Found {len(filtered)} contours", {"min_area": min_area})
        return filtered

    def extract_lines(self, contours):
        lines = []
        line_lengths = []
        for c in contours:
            if len(c) < 2: continue
            for i in range(len(c)):
                p1 = c[i][0].astype(np.float32)
                p2 = c[(i + 1) % len(c)][0].astype(np.float32)
                line = points_to_line(p1, p2)
                lines.append(line)
                line_lengths.append(np.linalg.norm(p2 - p1))
        metadata = {
            'lengths': line_lengths,
            'confidence': [1.0] * len(lines),
            'source_indices': list(range(len(lines)))
        }
        self.logger.trace("line_extraction", f"Extracted {len(lines)} lines from {len(contours)} contours",
                         {"total_lines": len(lines)})
        return lines, metadata

    def _is_valid_contour(self, approx):
        if len(approx) < self.min_vertices:
            return False
        if self.require_convex and not cv2.isContourConvex(approx):
            return False
        if self.require_quadrilateral and len(approx) != 4:
            return False
        if self.enable_min_area and cv2.contourArea(approx) < self.min_area:
            return False
        if self.enable_min_aspect_ratio and not self._check_aspect_ratio(approx):
            return False
        return True

    def _check_aspect_ratio(self, approx):
        _, _, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h != 0 else 0
        return aspect_ratio >= self.min_aspect_ratio

    def _simplify_contours(self, contour, epsilon_range=2.0):
        if contour is None or len(contour) == 0:
            return contour
        if isinstance(epsilon_range, (list, tuple)):
            epsilons = epsilon_range
        else:
            epsilons = [epsilon_range]
        perimeter = cv2.arcLength(contour, True)
        results = []
        for epsilon_factor in epsilons:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if self._is_valid_contour(approx):
                results.append(approx)
        return results[-1] if results else contour


class HoughLineExtractor(LineExtractor):
    def __init__(self, logger_instance=logger, rho=1.0, theta=None, threshold=50, minLineLength=50, maxLineGap=10):
        self.rho = rho
        self.theta = theta if theta is not None else np.pi / 180
        self.threshold = threshold
        self.minLineLength = minLineLength
        self.maxLineGap = maxLineGap
        self.logger = logger_instance

    def extract_lines(self, edges):
        edges_uint8 = edges.astype(np.uint8)
        hough_lines = cv2.HoughLinesP(
            edges_uint8,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.minLineLength,
            maxLineGap=self.maxLineGap
        )
        lines = []
        line_lengths = []
        if hough_lines is None:
            metadata = {'lengths': [], 'confidence': [], 'source_indices': []}
            return lines, metadata
        hough_lines = hough_lines.reshape(-1, 4)
        for x1, y1, x2, y2 in hough_lines:
            p1 = np.array([x1, y1], dtype=np.float32)
            p2 = np.array([x2, y2], dtype=np.float32)
            line = points_to_line(p1, p2)
            lines.append(line)
            line_lengths.append(np.linalg.norm(p2 - p1))
        metadata = {
            'lengths': line_lengths,
            'confidence': [1.0] * len(lines),
            'source_indices': list(range(len(lines)))
        }
        self.logger.trace("line_extraction", f"Extracted {len(lines)} lines from Hough",
                         {"threshold": self.threshold, "minLineLength": self.minLineLength})
        return lines, metadata


class LineExtractionFactory:
    @staticmethod
    def create(extractor_type: str, logger_instance=None, **kwargs):
        if logger_instance is None:
            logger_instance = logger
        if extractor_type == "contour":
            contour_kwargs = {
                "min_vertices": kwargs.get("contour_min_vertices", 3),
                "require_convex": kwargs.get("contour_require_convex", False),
                "require_quadrilateral": kwargs.get("contour_require_quadrilateral", False),
                "enable_min_area": kwargs.get("contour_enable_min_area", True),
                "min_area": kwargs.get("contour_min_area", 100),
                "enable_min_aspect_ratio": kwargs.get("contour_enable_min_aspect_ratio", False),
                "min_aspect_ratio": kwargs.get("contour_min_aspect_ratio", 0.5),
            }
            return ContourLineExtractor(logger_instance=logger_instance, **contour_kwargs)
        elif extractor_type == "hough":
            hough_kwargs = {
                'rho': kwargs.get('hough_rho', 1.0),
                'theta': kwargs.get('hough_theta', np.pi / 180),
                'threshold': kwargs.get('hough_threshold', 50),
                'minLineLength': kwargs.get('hough_min_line_length', 50),
                'maxLineGap': kwargs.get('hough_max_line_gap', 10),
            }
            return HoughLineExtractor(logger_instance=logger_instance, **hough_kwargs)
        else:
            raise ValueError(f"Unknown line extractor: {extractor_type}")
