import logging
import os
import sys

DEFAULT_LOG_FILE = os.getenv("NEXUS_LOG_FILE", "nexus_platform.log")

def setup_logging(level: int = logging.INFO, log_file: str = DEFAULT_LOG_FILE) -> None:
    root = logging.getLogger()
    root.handlers.clear()
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
    logging.info("Logging initialized for Nexus platform.")
