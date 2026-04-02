from abc import ABC, abstractmethod
from logging import getLogger
import time
import cv2
import numpy as np
LOGGER = getLogger(__name__)

class EdgeDetector(ABC):
    def __init__(self, logger=LOGGER):
        self.logger = logger

    @abstractmethod
    def detect(self, img):
        pass

class CannyDetector(EdgeDetector):
    def __init__(self, t1=50, t2=150, logger=LOGGER):
        super().__init__(logger)
        self.t1 = t1
        self.t2 = t2

    def detect(self, img):
        start = time.time()
        
        out = cv2.Canny(img, self.t1, self.t2)
        self.logger.info(
            f"CannyDetector: shape={img.shape} -> {out.shape} "
            f"time={(time.time() - start):.4f}s"
        )
        return out


class SobelDetector(EdgeDetector):
    def __init__(self, logger=LOGGER):
        super().__init__(logger)

    def detect(self, img):
        start = time.time()
        
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        out = cv2.magnitude(gx, gy)
        self.logger.info(
            f"SobelDetector: shape={img.shape} -> {out.shape} "
            f"time={(time.time() - start):.4f}s"
        )
        return out.astype(np.uint8)
