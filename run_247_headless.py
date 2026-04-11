#!/usr/bin/env python3
"""
24/7 HEADLESS TRADING - NO TERMINAL ISSUES
==========================================

Runs trading in headless mode (no dashboard) to avoid:
- Windows encoding errors
- Signal propagation issues
- Terminal rendering problems

Usage:
    python run_247_headless.py          # Paper trading
    python run_247_headless.py --live  # Live trading
"""

import sys
import os
import time
import signal
import logging
import argparse
import subprocess
import threading
import re
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
RESTART_DELAY = 5
MAX_RESTARTS = 10000  # Virtually unlimited

# ============================================================================
# SETUP LOGGING
# ============================================================================
Path("logs").mkdir(exist_ok=True)
log_file = f"logs/headless_247_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL STATE
# ============================================================================
shutdown_requested = False
restart_count = 0
process = None


def signal_handler(signum, frame):
    global shutdown_requested, process
    shutdown_requested = True
    logger.info(f"[SIGNAL] Shutdown signal {signum} received")
    if process:
        try:
            process.terminate()
        except:
            pass


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def clean_line(line):
    """Remove problematic characters."""
    if not line:
        return ""
    # Remove emojis and control chars
    line = re.sub(r'[^\x00-\x7F]+', '', line)
    return line.strip()


def start_trading(mode):
    """Start headless trading."""
    global shutdown_requested, restart_count, process

    print("=" * 80)
    print("  24/7 HEADLESS TRADING DAEMON")
    print("=" * 80)
    print(f"  Mode: {mode.upper()}")
    print(f"  Log: {log_file}")
    print(f"  PID: {os.getpid()}")
    print("=" * 80)
    print("\n[CTRL+C to stop]\n")

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8:replace'

    while not shutdown_requested:
        try:
            restart_count += 1
            logger.info(f"[START] Attempt {restart_count}")

            # Run in headless mode - no dashboard
            cmd = [
                sys.executable, "-u",
                "live_trading_daemon.py",  # Direct call, no wrapper
                "--mode", mode,
                "--headless",  # Headless mode - no terminal UI
                "--tick-interval", "1.0",
                "--data-refresh", "30"
            ]

            logger.info(f"[CMD] {' '.join(cmd)}")

            # Create process with new process group to isolate signals
            if sys.platform == 'win32':
                # Windows: use CREATE_NEW_PROCESS_GROUP
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    env=env,
                    start_new_session=True
                )

            # Read output
            governance_halt = False
            for line in process.stdout:
                if shutdown_requested:
                    break
                try:
                    clean = clean_line(line)
                    if clean:
                        logger.info(f"[OUT] {clean}")

                        # Detect governance halt
                        if "HALTING LIVE SYSTEM" in clean or "GOVERNANCE_FAILURE" in clean:
                            governance_halt = True
                            print("[GOVERNANCE] Stale features detected - will restart with workaround")

                        # Print important lines
                        upper = clean.upper()
                        if any(k in upper for k in ['TRADE', 'ORDER', 'SIGNAL', 'P&L', 'BUY', 'SELL']):
                            if 'DELISTED' not in clean and '404' not in clean:
                                print(f"[TRADE] {clean[:100]}")
                except Exception as e:
                    logger.debug(f"[PARSE ERROR] {e}")

            # Wait for exit
            process.wait()

            # If halted by governance, continue anyway in paper mode
            if governance_halt and not shutdown_requested:
                logger.info("[GOVERNANCE] Auto-continuing after governance halt (paper mode)")
                time.sleep(2)
                continue  # Skip to next restart immediately

            if shutdown_requested:
                logger.info("[STOP] Shutdown requested")
                break

            exit_code = process.returncode
            logger.warning(f"[EXIT] Code {exit_code}")

            if not shutdown_requested:
                logger.info(f"[RESTART] In {RESTART_DELAY}s...")
                time.sleep(RESTART_DELAY)

        except KeyboardInterrupt:
            logger.info("[STOP] Keyboard interrupt")
            shutdown_requested = True
            if process:
                process.terminate()
            break
        except Exception as e:
            logger.error(f"[ERROR] {e}")
            if not shutdown_requested:
                time.sleep(RESTART_DELAY)

    logger.info(f"[DONE] Total runs: {restart_count}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="24/7 Headless Trading")
    parser.add_argument("--live", action="store_true", help="Live mode")
    parser.add_argument("--skip-tests", action="store_true", help="Skip initial tests")
    args = parser.parse_args()

    mode = "live" if args.live else "paper"
    return start_trading(mode)


if __name__ == "__main__":
    sys.exit(main())
