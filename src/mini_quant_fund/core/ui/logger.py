
import logging
import sys
import os
import json
from datetime import datetime

class InstitutionalLogger:
    """Institutional-grade logging with structured output."""
    
    @staticmethod
    def setup(level=logging.INFO, log_file=None):
        """Configure the global logging system."""
        root = logging.getLogger()
        root.setLevel(level)
        
        # Remove existing handlers
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        formatter = logging.Formatter(format_str)
        
        # Console handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root.addHandler(stdout_handler)
        
        # File handler (Structured JSON for production)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    log_entry = {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "name": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "line": record.lineno
                    }
                    if record.exc_info:
                        log_entry["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_entry)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(JsonFormatter())
            root.addHandler(file_handler)
            
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        return root

def get_logger(name):
    """Get a named logger."""
    return logging.getLogger(name)
