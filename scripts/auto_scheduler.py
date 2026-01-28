#!/usr/bin/env python3
"""
Auto-scheduler for Ingestion Agent

Designed to run nightly via Cron or Airflow.
Responsibility: Trigger ingest_history.py and check for completion.
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SCHEDULER")

def run_nightly_ingestion():
    logger.info("Starting scheduled nightly ingestion...")

    try:
        # Run ingest_history.py
        result = subprocess.run(
            [sys.executable, "ingest_history.py"],
            capture_output=True,
            text=True,
            check=True
        )

        logger.info("Ingestion completed successfully.")
        logger.debug(result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error(f"Ingestion FAILED with exit code {e.returncode}")
        logger.error(e.stderr)
        # In a real environment, trigger critical alerts here
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"Unexpected error in scheduler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_nightly_ingestion()
