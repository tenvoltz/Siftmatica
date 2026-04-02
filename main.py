import cv2
from env import OUTPUT_FOLDER
from src.pipeline.edge_detector import CannyDetector, SobelDetector
from src.pipeline.orchestrator import EdgePipeline
from src.pipeline.postprocessing import Dilate, Pixelate
from src.pipeline.preprocess import BilateralFilter, GaussianBlur


def build_pipeline(config):
    pre_map = {"gaussian": GaussianBlur(), "bilateral": BilateralFilter()}

    det_map = {
        "canny": CannyDetector(),
        "sobel": SobelDetector(),
    }

    post_map = {"pixelate": Pixelate(), "dilate": Dilate()}

    pre = [pre_map[name] for name in config["pre"]]
    det = det_map[config["detector"]]
    post = [post_map[name] for name in config["post"]]

    return EdgePipeline(pre, det, post)


configs = [
    {"pre": ["gaussian"], "detector": "canny", "post": []},
    {"pre": ["bilateral"], "detector": "canny", "post": []},
    {"pre": [], "detector": "sobel", "post": ["dilate"]},
]

#img = cv2.imread("data\\house1\\images\\2026-03-23_19.30.55.png")
img = cv2.imread("data\\elven-house\\images\\2026-04-01_23.44.16.png")

for i, cfg in enumerate(configs):
    pipeline = build_pipeline(cfg)
    result = pipeline.run(img)
    cv2.imwrite(f"{OUTPUT_FOLDER}\\output_{i}.png", result)
