#!/usr/bin/env python3
"""
Nexus Institutional Trading Platform Launcher
Launches the full 24/7 autonomous trading system.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Launcher")

def main():
    root = Path(__file__).parent
    os.chdir(root)

    logger.info("Starting Nexus Institutional Trading Platform...")

    # Ensure virtual environment is activated
    venv_path = root / ".venv" / "Scripts" / "activate.ps1"
    if venv_path.exists():
        logger.info("Activating virtual environment...")
        # Note: In a real deployment, handle venv activation properly
    else:
        logger.warning("Virtual environment not found, proceeding with system Python.")

    # Launch orchestrator
    logger.info("Launching Nexus Orchestrator...")
    try:
        subprocess.run([sys.executable, "nexus_orchestrator.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Orchestrator failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
        sys.exit(0)

if __name__ == "__main__":
    main()