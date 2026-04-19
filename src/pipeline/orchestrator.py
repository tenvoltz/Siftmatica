from src.pipeline.preprocess import GaussianBlur, BilateralFilter, ETFComputer
from src.pipeline.edge_detector import CannyDetector, SobelDetector, XDoGDetector, FDoGDetector, DoGEdgeDetector
from src.pipeline.postprocessing import Dilate, Pixelate


class EdgePipeline:
    def __init__(self, preprocessors, detector, postprocessors):
        self.preprocessors = preprocessors
        self.detector = detector
        self.postprocessors = postprocessors

    def run(self, img):
        for p in self.preprocessors:
            img = p.run(img)

        edges = self.detector.detect(img)

        for p in self.postprocessors:
            edges = p.run(edges)

        return edges


class PipelineBuilder:
    PRE_MAP = {
        "gaussian": GaussianBlur(),
        "bilateral": BilateralFilter(),
        "etf": ETFComputer()
    }
    DET_MAP = {
        "canny": CannyDetector(),
        "sobel": SobelDetector(),
        "xdog": XDoGDetector(),
        "fdog": FDoGDetector(),
        "dog": DoGEdgeDetector()
    }
    POST_MAP = {
        "pixelate": Pixelate(),
        "dilate": Dilate()
    }

    @staticmethod
    def build(config):
        pre = [PipelineBuilder.PRE_MAP[n] for n in config.preprocessors]
        det = PipelineBuilder.DET_MAP[config.detector]
        post = [PipelineBuilder.POST_MAP[n] for n in config.postprocessors]
        return EdgePipeline(pre, det, post)

    @staticmethod
    def config_to_name(cfg):
        parts = []
        if cfg.preprocessors:
            parts.append("+".join(cfg.preprocessors))
        parts.append(cfg.detector)
        if cfg.postprocessors:
            parts.append("+".join(cfg.postprocessors))
        return "_".join(parts)
