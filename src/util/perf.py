import time
import logging


class PerfContext:
    def __init__(self, stage: str, logger):
        self.stage = stage
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start_time
        self.logger.perf(self.stage, duration)
