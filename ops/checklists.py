
import os
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Optional
import dotenv

logger = logging.getLogger(__name__)

class PreFlightChecklist:
    """
    Mandatory System Readiness Checks.
    MUST pass before any trading loop starts.
    """

    @staticmethod
    def run_checks() -> bool:
        logger.info("[CHECKLIST] Initiating Pre-Flight Checks...")
        all_passed = True

        # 1. Environment Variables
        if not PreFlightChecklist._check_env():
            all_passed = False

        # 2. Time Synchronization
        if not PreFlightChecklist._check_time_sync():
            all_passed = False

        # 3. Directory Structure
        if not PreFlightChecklist._check_directories():
            all_passed = False

        if all_passed:
            logger.info("[CHECKLIST] [OK] ALL SYSTEMS GO.")
            return True
        else:
            logger.critical("[CHECKLIST] [FAIL] PRE-FLIGHT CHECKS FAILED. ABORTING.")
            sys.exit(1)

    @staticmethod
    def _check_env() -> bool:
        """Verify .env is loaded and critical keys exist."""
        # Reload dotenv to be sure
        dotenv.load_dotenv(override=True)

        required_keys = ["ALPHAVANTAGE_API_KEY"]
        missing = []
        for key in required_keys:
            val = os.getenv(key)
            if not val or val.strip() == "":
                missing.append(key)

        if missing:
            logger.error(f"[CHECKLIST] Missing Env Vars: {missing}")
            return False

        return True

    @staticmethod
    def _check_time_sync() -> bool:
        """Ensure system clock is reasonably close to UTC."""
        sys_time = datetime.now(timezone.utc)

        if sys_time.year < 2025:
            logger.error(f"[CHECKLIST] System Clock Error: Year is {sys_time.year}. Update your clock.")
            return False

        logger.info(f"[CHECKLIST] System Time (UTC): {sys_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True

    @staticmethod
    def _check_directories() -> bool:
        required = ["data/cache", "logs", "configs"]
        for d in required:
            if not os.path.exists(d):
                try:
                    os.makedirs(d)
                    logger.info(f"[CHECKLIST] Created missing directory: {d}")
                except Exception as e:
                    logger.error(f"[CHECKLIST] Failed to create {d}: {e}")
                    return False
        return True

def generate_daily_checklist(config_hash: Optional[str] = None) -> bool:
    """
    Wrapper for main.py compatibility.
    Runs the pre-flight checklist and returns status.
    """
    logger.info(f"Generating Daily Checklist (Config Hash: {config_hash or 'N/A'})...")
    return PreFlightChecklist.run_checks()
