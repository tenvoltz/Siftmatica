from logging import getLogger
import time

import cv2
from abc import ABC, abstractmethod

LOGGER = getLogger(__name__)

class Preprocessor(ABC):
    def __init__(self, logger=LOGGER):
        self.logger = logger

    @abstractmethod
    def run(self, img):
        pass

class GaussianBlur(Preprocessor):
    def __init__(self, ksize=5, logger=LOGGER):
        super().__init__(logger)
        self.ksize = ksize

    def run(self, img):
        start = time.time()
        
        out = cv2.GaussianBlur(img, (self.ksize, self.ksize), 0)

        self.logger.info(
            f"GaussianBlur: shape={img.shape} -> {out.shape} "
            f"time={(time.time() - start):.4f}s"
        )
        
        return cv2.GaussianBlur(img, (self.ksize, self.ksize), 0)


class BilateralFilter(Preprocessor):
    def __init__(self, d=9, sigma_color=75, sigma_space=75, logger=LOGGER):
        super().__init__(logger)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def run(self, img):
        start = time.time()

        out = cv2.bilateralFilter(img, self.d,
                                self.sigma_color,
                                self.sigma_space)

        self.logger.info(
            f"BilateralFilter: shape={img.shape} -> {out.shape} "
            f"time={(time.time() - start):.4f}s"
        )

        return out