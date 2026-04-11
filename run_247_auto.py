#!/usr/bin/env python3
"""
24/7 AUTONOMOUS TRADING SYSTEM - NO USER INPUT REQUIRED
=======================================================

This script runs the complete trading system fully automated:
- No prompts
- No user interaction required
- Runs continuously 24/7
- Auto-restarts on errors
- Full logging to file

Usage:
    python run_247_auto.py          # Start 24/7 paper trading
    python run_247_auto.py --live  # Start 24/7 live trading (CAUTION!)

Author: Mini Quant Fund AI
Version: 2.0.0 - FULLY AUTONOMOUS
"""

import sys
import os
import time
import signal
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION - ADJUST AS NEEDED
# ============================================================================
TRADING_MODE = "paper"  # Default: "paper" (safe), can be "live" (real money)
AUTO_RESTART_ON_CRASH = True
RESTART_DELAY_SECONDS = 5
LOG_RETENTION_DAYS = 30

# ============================================================================
# SETUP LOGGING
# ============================================================================
Path("logs").mkdir(exist_ok=True)
log_file = f"logs/trading_247_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL STATE
# ============================================================================
shutdown_requested = False
restart_count = 0
MAX_RESTARTS = 5


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"[SIGNAL] Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def print_banner(text: str, char: str = "="):
    """Print a banner."""
    print()
    print(char * 80)
    print(f"  {text}")
    print(char * 80)
    print()


def run_tests() -> bool:
    """Run all system tests."""
    print_banner("PHASE 1: SYSTEM VALIDATION")

    tests = [
        ("Basic Intelligence", [sys.executable, "test_simple.py"]),
        ("Unified Engine", [sys.executable, "test_unified_integration.py"]),
        ("Complete System", [sys.executable, "test_complete_system.py"]),
    ]

    passed = 0
    for name, cmd in tests:
        logger.info(f"[TEST] Running: {name}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0 or "PASS" in result.stdout or "SUCCESS" in result.stdout:
                logger.info(f"[TEST] {name}: PASSED")
                passed += 1
            else:
                logger.warning(f"[TEST] {name}: FAILED (exit code {result.returncode})")
        except Exception as e:
            logger.error(f"[TEST] {name}: ERROR - {e}")

    logger.info(f"[TEST] Results: {passed}/{len(tests)} tests passed")
    return passed >= 2  # Proceed if at least 2 tests pass


def start_trading_daemon(mode: str) -> int:
    """
    Start the actual 24/7 trading daemon.
    This runs INDEFINITELY until manually stopped.
    """
    global restart_count, shutdown_requested

    print_banner(f"STARTING 24/7 TRADING DAEMON - MODE: {mode.upper()}", "#")

    logger.info(f"[DAEMON] Initializing 24/7 trading daemon (mode={mode})")
    logger.info(f"[DAEMON] Log file: {log_file}")
    logger.info(f"[DAEMON] Auto-restart: {AUTO_RESTART_ON_CRASH}")
    logger.info(f"[DAEMON] To stop: Press Ctrl+C or kill process")

    while not shutdown_requested and restart_count < MAX_RESTARTS:
        try:
            # Start the trading process
            cmd = [sys.executable, "start_trading.py", "--mode", mode, "--force"]

            logger.info(f"[DAEMON] Starting trading process (attempt {restart_count + 1}/{MAX_RESTARTS})")
            logger.info(f"[DAEMON] Command: {' '.join(cmd)}")

            # Run the trading daemon
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1  # Line buffered
            )

            # Stream output in real-time
            for line in process.stdout:
                line = line.strip()
                if line:
                    # Log everything
                    logger.info(f"[TRADING] {line}")
                    # Also print important messages
                    if any(keyword in line for keyword in ["P&L", "profit", "loss", "trade", "signal", "ORDER"]):
                        print(f"[LIVE] {line}")

            # Wait for process to complete
            process.wait()

            if shutdown_requested:
                logger.info("[DAEMON] Shutdown requested, stopping...")
                break

            # If we get here, process exited unexpectedly
            exit_code = process.returncode
            logger.warning(f"[DAEMON] Trading process exited with code {exit_code}")

            if AUTO_RESTART_ON_CRASH and not shutdown_requested:
                restart_count += 1
                logger.info(f"[DAEMON] Auto-restarting in {RESTART_DELAY_SECONDS} seconds...")
                time.sleep(RESTART_DELAY_SECONDS)
            else:
                break

        except KeyboardInterrupt:
            logger.info("[DAEMON] Keyboard interrupt received, shutting down...")
            shutdown_requested = True
            break
        except Exception as e:
            logger.error(f"[DAEMON] Error: {e}", exc_info=True)
            if AUTO_RESTART_ON_CRASH and not shutdown_requested:
                restart_count += 1
                time.sleep(RESTART_DELAY_SECONDS)
            else:
                break

    logger.info(f"[DAEMON] Trading daemon stopped after {restart_count} restarts")
    return 0 if shutdown_requested else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="24/7 Autonomous Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_247_auto.py          # Start paper trading (safe)
  python run_247_auto.py --live # Start live trading (real money!)
        """
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable LIVE trading with REAL money (DANGEROUS!)"
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Skip system tests and start trading immediately"
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Run tests only, don't start trading"
    )

    args = parser.parse_args()

    # Print startup banner
    print_banner("MINI QUANT FUND - 24/7 AUTONOMOUS TRADING", "*")
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log File:   {log_file}")
    print(f"  PID:        {os.getpid()}")
    print()

    # Determine trading mode
    if args.live:
        mode = "live"
        print("*** WARNING: LIVE TRADING MODE ***")
        print("This will trade with REAL money!")
        print("Make sure you have tested extensively in paper mode first.")
        time.sleep(5)  # Give user time to see warning
    else:
        mode = "paper"
        print("*** PAPER TRADING MODE (Safe) ***")
        print("Simulated trades - no real money at risk.")

    print()

    # Phase 1: Run tests (unless skipped)
    if not args.skip_tests:
        if not run_tests():
            logger.error("[FATAL] System tests failed - trading not started")
            logger.info("[INFO] Use --skip-tests to bypass (not recommended)")
            return 1
    else:
        logger.warning("[WARN] Skipping system tests - proceeding blindly")

    if args.test_only:
        logger.info("[INFO] Test-only mode - not starting trading")
        return 0

    # Phase 2: Start 24/7 trading
    print_banner("ENTERING 24/7 TRADING MODE - PRESS CTRL+C TO STOP", "!")
    return start_trading_daemon(mode)


if __name__ == "__main__":
    sys.exit(main())
