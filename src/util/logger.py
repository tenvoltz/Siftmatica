import logging
import sys

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and better formatting"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        
        # Format: [TIMESTAMP] [LEVEL] (filename:lineno) message
        formatted = (
            f"{self.BOLD}{color}[{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}]{self.RESET} "
            f"{self.BOLD}{color}[{record.levelname}]{self.RESET} "
            f"{color}({record.filename}:{record.lineno}){self.RESET} "
            f"{record.getMessage()}"
        )
        
        return formatted


def get_logger(name, level='INFO'):
    """Get a configured logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear() # Remove any existing handlers
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    formatter = ColoredFormatter()
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger