import json
import logging
import os
import sys
from datetime import datetime

DEFAULT_LOG_FILE = os.getenv("NEXUS_LOG_FILE", "nexus_platform.log")
LOG_JSON = os.getenv("NEXUS_LOG_JSON", "false").lower() == "true"

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for institutional observability."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

def setup_logging(level: int = logging.INFO, log_file: str = DEFAULT_LOG_FILE) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    
    if LOG_JSON:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    root.setLevel(level)
    logging.info("Logging initialized for Nexus platform (JSON=%s).", LOG_JSON)
