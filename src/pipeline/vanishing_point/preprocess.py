from abc import ABC, abstractmethod
import time
import cv2
import numpy as np
from scipy.ndimage import convolve
from src.util.logger import get_logger

LOGGER = get_logger(__name__)

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
        elapsed = time.time() - start
        self.logger.trace("gaussian_blur", "Blur applied", {"input_shape": str(img.shape), "output_shape": str(out.shape), "time_s": round(elapsed, 4)})
        return out

class BilateralFilter(Preprocessor):
    def __init__(self, d=9, sigma_color=75, sigma_space=75, logger=LOGGER):
        super().__init__(logger)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def run(self, img):
        start = time.time()
        out = cv2.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space)
        elapsed = time.time() - start
        self.logger.trace("bilateral_filter", "Filter applied", {"input_shape": str(img.shape), "output_shape": str(out.shape), "time_s": round(elapsed, 4)})

        return out

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
        tx = -gy / mag  
        ty = gx / mag

        kernel = self._create_gaussian_kernel()
        
        for _ in range(self.iterations):
            dot = tx * tx + ty * ty
            dot = np.clip(dot, -1, 1)
            tx_new = convolve(tx * dot, kernel, mode='wrap')
            ty_new = convolve(ty * dot, kernel, mode='wrap')
            norm = np.sqrt(tx_new**2 + ty_new**2) + 1e-8
            tx = tx_new / norm
            ty = ty_new / norm

        if self.logger:
            elapsed = time.time() - start
            self.logger.trace("etf_computer", "ETF computed", {"iterations": self.iterations, "time_s": round(elapsed, 4)})
        return (img, (tx, ty))
    
    def _create_gaussian_kernel(self):
        kr = self.kernel_radius
        y, x = np.ogrid[-kr:kr+1, -kr:kr+1]
        return np.exp(-(x**2 + y**2) / (2 * kr)) / (2 * np.pi * kr**2)
