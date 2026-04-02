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
