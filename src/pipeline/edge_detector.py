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
    def __init__(self, threshold=100, logger=LOGGER):
        super().__init__(logger)
        self.threshold = threshold

    def detect(self, img):
        start = time.time()

        # 1. Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray = gray.astype(np.float32)

        # 2. Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        mag = cv2.magnitude(gx, gy)

        # 3. Normalize to 0–255
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # 4. Threshold → binary
        edges = np.where(mag > self.threshold, 255, 0).astype(np.uint8)

        if self.logger:
            self.logger.info(
                f"SobelDetector | shape={img.shape} | "
                f"time={time.time() - start:.4f}s"
            )

        return edges

class XDoGDetector(EdgeDetector):
    def __init__(self, sigma=0.8, k=1.6, p=0.98, epsilon=0.01, phi=10, logger=None):
        super().__init__(logger)
        self.sigma = sigma
        self.k = k
        self.p = p
        self.epsilon = epsilon
        self.phi = phi

    def detect(self, img):
        start = time.time()

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32) / 255.0

        g1 = cv2.GaussianBlur(img, (0, 0), self.sigma)
        g2 = cv2.GaussianBlur(img, (0, 0), self.sigma * self.k)

        dog = g1 - self.p * g2

        xdog = np.where(
            dog < self.epsilon, 1.0, 1.0 + np.tanh(self.phi * (dog - self.epsilon))
        )

        xdog = (xdog * 255).astype(np.uint8)

        if self.logger:
            self.logger.debug(
                f"XDoG | sigma={self.sigma} | time={time.time()-start:.4f}s"
            )

        return xdog


class FDoGDetector(EdgeDetector):
    def __init__(
        self,
        sigma_c=1.0,  # across flow (DoG)
        sigma_s=2.0,  # along flow smoothing
        tau=0.99,
        step=1.0,
        max_len=5,
        logger=None,
    ):
        super().__init__(logger)
        self.sigma_c = sigma_c
        self.sigma_s = sigma_s
        self.tau = tau
        self.step = step
        self.max_len = max_len

    def detect(self, data):
        start = time.time()

        img, (tx, ty) = data
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        h, w = gray.shape
        response = np.zeros_like(gray)

        for y in range(h):
            for x in range(w):

                t = np.array([tx[y, x], ty[y, x]])
                n = np.array([-t[1], t[0]])  # normal direction

                # --- Step 1: DoG along normal ---
                sum1 = 0.0
                sum2 = 0.0
                w1 = 0.0
                w2 = 0.0

                for s in np.arange(-self.max_len, self.max_len + self.step, self.step):
                    px = x + n[0] * s
                    py = y + n[1] * s

                    if px < 0 or px >= w or py < 0 or py >= h:
                        continue

                    val = bilinear_sample(gray, px, py)

                    g1 = gaussian_weight(s, self.sigma_c)
                    g2 = gaussian_weight(s, self.sigma_c * 1.6)

                    sum1 += val * g1
                    sum2 += val * g2

                    w1 += g1
                    w2 += g2

                if w1 == 0 or w2 == 0:
                    continue

                dog = (sum1 / w1) - self.tau * (sum2 / w2)

                # --- Step 2: accumulate along tangent ---
                acc = 0.0
                w_acc = 0.0

                for s in np.arange(-self.max_len, self.max_len + self.step, self.step):
                    px = x + t[0] * s
                    py = y + t[1] * s

                    if px < 0 or px >= w or py < 0 or py >= h:
                        continue

                    weight = gaussian_weight(s, self.sigma_s)

                    acc += dog * weight
                    w_acc += weight

                if w_acc > 0:
                    response[y, x] = acc / w_acc

        # --- Step 3: threshold ---
        edges = np.where(response < 0, 1.0, 0.0)

        edges = (edges * 255).astype(np.uint8)

        if self.logger:
            self.logger.info(f"True FDoG complete | time={time.time()-start:.2f}s")

        return edges


def gaussian_weight(x, sigma):
    return np.exp(-(x * x) / (2 * sigma * sigma))


def bilinear_sample(img, x, y):
    h, w = img.shape

    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    val = (
        img[y0, x0] * (1 - dx) * (1 - dy)
        + img[y0, x1] * dx * (1 - dy)
        + img[y1, x0] * (1 - dx) * dy
        + img[y1, x1] * dx * dy
    )

    return val


class DoGEdgeDetector(EdgeDetector):
    def __init__(self, sigma=1.0, k=1.6, tau=1.0, threshold=0.0, logger=None):
        super().__init__(logger)
        self.sigma = sigma
        self.k = k
        self.tau = tau
        self.threshold = threshold

    def detect(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32) / 255.0

        g1 = cv2.GaussianBlur(img, (0, 0), self.sigma)
        g2 = cv2.GaussianBlur(img, (0, 0), self.sigma * self.k)

        dog = g1 - self.tau * g2

        edges = np.where(dog > self.threshold, 255, 0).astype(np.uint8)

        return edges
