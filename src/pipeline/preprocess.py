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

import cv2
import numpy as np
import time


class ETFComputer(Preprocessor):
    def __init__(self, iterations=3, kernel_radius=5, logger=None):
        super().__init__(logger)
        self.iterations = iterations
        self.kernel_radius = kernel_radius

    def run(self, img):
        start = time.time()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

        mag = np.sqrt(gx**2 + gy**2) + 1e-8

        # Initial tangent field (perpendicular to gradient)
        tx = -gy / mag
        ty = gx / mag

        # Iterative smoothing (ETF refinement)
        for _ in range(self.iterations):
            tx_new = np.zeros_like(tx)
            ty_new = np.zeros_like(ty)

            for dx in range(-self.kernel_radius, self.kernel_radius + 1):
                for dy in range(-self.kernel_radius, self.kernel_radius + 1):
                    weight = np.exp(-(dx**2 + dy**2) / (2 * self.kernel_radius))

                    shifted_tx = np.roll(tx, (dy, dx), axis=(0, 1))
                    shifted_ty = np.roll(ty, (dy, dx), axis=(0, 1))

                    dot = tx * shifted_tx + ty * shifted_ty
                    dot = np.clip(dot, -1, 1)

                    tx_new += weight * dot * shifted_tx
                    ty_new += weight * dot * shifted_ty

            norm = np.sqrt(tx_new**2 + ty_new**2) + 1e-8
            tx = tx_new / norm
            ty = ty_new / norm

        if self.logger:
            self.logger.debug(
                f"ETF computed | iter={self.iterations} | time={time.time()-start:.4f}s"
            )

        return (img, (tx, ty))
