import logging
import os
import sys
import time
import warnings

# Third-party imports for structured and rich logging
try:
    from pythonjsonlogger import jsonlogger
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    # Fallback - no rich logging
    RICH_AVAILABLE = False
    Console = None
    RichHandler = None
    jsonlogger = None

def setup_logging(name: str = "mini_quant", log_dir: str = "runtime/logs") -> logging.Logger:
    """
    Setup institutional-grade logging with three sinks:
    1. Console: Rich (human readable, pretty)
    2. File: live.jsonl (machine readable, all INFO+)
    3. File: errors.jsonl (machine readable, structured ERROR+)
    """

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates during reload
    if logger.handlers:
        logger.handlers.clear()

    # 1. Console Handler (Rich with Throttling)
    if RICH_AVAILABLE:
        class ThrottledRichHandler(RichHandler):
            """
            Throttled console handler - limits INFO/DEBUG to min_interval seconds.
            ERROR/CRITICAL always pass through immediately.
            """
            def __init__(self, min_interval=5, **kwargs):
                super().__init__(**kwargs)
                self.min_interval = min_interval
                self._last_emit_time = 0
                self._buffer = []

            def emit(self, record):
                # Always emit ERROR and CRITICAL immediately
                if record.levelno >= logging.ERROR:
                    super().emit(record)
                    return

                # Throttle INFO and DEBUG
                now = time.time()
                if now - self._last_emit_time >= self.min_interval:
                    # Emit buffered + current
                    for buffered_record in self._buffer:
                        super().emit(buffered_record)
                    self._buffer.clear()
                    super().emit(record)
                    self._last_emit_time = now
                else:
                    # Buffer for later
                    if len(self._buffer) < 100:  # Prevent unbounded growth
                        self._buffer.append(record)

        console_handler = ThrottledRichHandler(
            min_interval=5,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True
        )
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)
    else:
        # Fallback console handler if rich is not available
        fallback_handler = logging.StreamHandler(sys.stdout)
        fallback_handler.setLevel(logging.INFO)
        fallback_fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fallback_handler.setFormatter(fallback_fmt)
        logger.addHandler(fallback_handler)

    # 2. Main Structured File Handler (JSONL)
    json_log_path = os.path.join(log_dir, "live.jsonl")
    file_handler = logging.FileHandler(json_log_path)
    file_handler.setLevel(logging.INFO)

    # 3. Dedicated Error File Handler (JSONL)
    error_log_path = os.path.join(log_dir, "errors.jsonl")
    error_handler = logging.FileHandler(error_log_path)
    error_handler.setLevel(logging.ERROR)

    # JSON Formatter - only if jsonlogger is available
    if jsonlogger is not None:
        class CustomJsonFormatter(jsonlogger.JsonFormatter):
            def add_fields(self, log_record, record, message_dict):
                super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
                if not log_record.get('timestamp'):
                    from datetime import datetime
                    log_record['timestamp'] = datetime.utcnow().isoformat()
                if log_record.get('level'):
                    log_record['level'] = log_record['level'].upper()
                else:
                    log_record['level'] = record.levelname

                # Include traceback for errors
                if record.levelno >= logging.ERROR and record.exc_info:
                    log_record['stack_trace'] = self.formatException(record.exc_info)

        json_formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    else:
        # Fallback formatter without JSON
        json_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')

    file_handler.setFormatter(json_formatter)
    error_handler.setFormatter(json_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    # Warnings are configured in main.py

    # Prevent propagation to root logger
    logger.propagate = False

    return logger

# Global console instance (only if rich is available)
if RICH_AVAILABLE:
    console = Console()
else:
    console = None
