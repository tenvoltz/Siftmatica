import logging
import sys
import inspect
from typing import Optional, Dict, Any
from tqdm import tqdm

class ColoredFormatter(logging.Formatter):
    COLORS = {'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m', 'ERROR': '\033[31m', 'CRITICAL': '\033[35m'}
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        return (f"{self.BOLD}{color}[{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}]{self.RESET} "
                f"{self.BOLD}{color}[{record.levelname}]{self.RESET} "
                f"{color}({record.filename}:{record.lineno}){self.RESET} "
                f"{record.getMessage()}")

class SiftmaticaLogger:
    def __init__(self, logger):
        self._logger = logger
        self._progress_bar = None
    
    def _get_caller_info(self, depth=2):
        frame = inspect.currentframe()
        try:
            for _ in range(depth):
                if frame is None: return "unknown", 0
                frame = frame.f_back
            if frame is None:
                return "unknown", 0
            return frame.f_code.co_filename.split('\\')[-1], frame.f_lineno
        finally: del frame
    
    def trace(self, stage: str, msg: str, data: Optional[Dict[str, Any]] = None):
        data_str = f" | {data}" if data else ""
        filename, lineno = self._get_caller_info(2)
        record = self._logger.makeRecord(
            self._logger.name, logging.DEBUG, filename, lineno,
            f"[{stage}] {msg}{data_str}", (), None
        )
        self._logger.handle(record)
    
    def perf(self, stage: str, duration: float, count: Optional[int] = None):
        rate = f" | {count/duration:.0f} ops/sec" if count else ""
        filename, lineno = self._get_caller_info(2)
        record = self._logger.makeRecord(
            self._logger.name, logging.INFO, filename, lineno,
            f"[{stage}] Duration: {duration:.4f}s{rate}", (), None
        )
        self._logger.handle(record)
    
    def validate(self, check_name: str, passed: bool, details: str = ""):
        status = "✓" if passed else "✗"
        detail_str = f" - {details}" if details else ""
        level = logging.INFO if passed else logging.WARNING
        filename, lineno = self._get_caller_info(2)
        record = self._logger.makeRecord(
            self._logger.name, level, filename, lineno,
            f"{status} {check_name}{detail_str}", (), None
        )
        self._logger.handle(record)
    
    def progress(self, iterable, description="", total=None):
        return tqdm(iterable, desc=description, total=total, unit="item")

def get_logger(name, level='DEBUG'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return SiftmaticaLogger(logger)