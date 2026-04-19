from src.util.logger import get_logger
import time

import cv2
import numpy as np
from abc import ABC, abstractmethod

LOGGER = get_logger(__name__)

class Postprocessor(ABC):
    def __init__(self, logger=LOGGER):
        self.logger = logger

    @abstractmethod
    def run(self, edges):
        pass

class Pixelate(Postprocessor):
    def __init__(self, size=16, logger=LOGGER):
        super().__init__(logger)
        self.size = size

    def run(self, img):
        start = time.time()
        h, w = img.shape[:2]
        small = cv2.resize(
            img, (w // self.size, h // self.size), interpolation=cv2.INTER_NEAREST
        )
        elapsed = time.time() - start
        self.logger.trace("pixelate", "Pixelation applied", {"input_shape": str(img.shape), "output_shape": str(small.shape), "time_s": round(elapsed, 4)})

        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class Dilate(Postprocessor):
    def __init__(self, k=2, logger=LOGGER):
        super().__init__(logger)
        self.kernel = np.ones((k, k), np.uint8)

    def run(self, img):
        start = time.time()
        out = cv2.dilate(img, self.kernel)
        elapsed = time.time() - start
        self.logger.trace("dilate", "Dilation applied", {"input_shape": str(img.shape), "output_shape": str(out.shape), "time_s": round(elapsed, 4)})
        
        return out
