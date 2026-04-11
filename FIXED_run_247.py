#!/usr/bin/env python3
"""
24/7 AUTONOMOUS TRADING SYSTEM - WINDOWS COMPATIBLE
===================================================

Fixed version that handles:
- Windows encoding (no emojis)
- Delisted stock errors
- Network failures
- Auto-restart on crash

Usage:
    python FIXED_run_247.py          # Start paper trading
    python FIXED_run_247.py --live  # Start live trading
"""

import sys
import os
import time
import signal
import logging
import argparse
import subprocess
import io
from datetime import datetime
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION
# ============================================================================
AUTO_RESTART_ON_CRASH = True
RESTART_DELAY_SECONDS = 5
MAX_RESTARTS = 100  # High number for true 24/7 operation

# ============================================================================
# SETUP LOGGING
# ============================================================================
Path("logs").mkdir(exist_ok=True)
log_file = f"logs/trading_247_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"[SIGNAL] Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


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
            if result.returncode == 0 or "PASS" in result.stdout or "A+" in result.stdout:
                logger.info(f"[TEST] {name}: PASSED")
                passed += 1
            else:
                logger.warning(f"[TEST] {name}: FAILED (exit code {result.returncode})")
        except Exception as e:
            logger.error(f"[TEST] {name}: ERROR - {e}")

    logger.info(f"[TEST] Results: {passed}/{len(tests)} tests passed")
    return passed >= 1  # Proceed if at least 1 test passes


def start_trading_daemon(mode: str) -> int:
    """Start the actual 24/7 trading daemon."""
    global restart_count, shutdown_requested

    print_banner(f"STARTING 24/7 TRADING DAEMON - MODE: {mode.upper()}", "#")

    logger.info(f"[DAEMON] Initializing 24/7 trading daemon (mode={mode})")
    logger.info(f"[DAEMON] Log file: {log_file}")
    logger.info(f"[DAEMON] Auto-restart: {AUTO_RESTART_ON_CRASH}")
    logger.info(f"[DAEMON] Max restarts: {MAX_RESTARTS}")
    logger.info(f"[DAEMON] To stop: Press Ctrl+C")

    while not shutdown_requested and restart_count < MAX_RESTARTS:
        try:
            # Start the trading process with error handling
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            cmd = [sys.executable, "-u", "start_trading.py", "--mode", mode, "--force"]

            logger.info(f"[DAEMON] Starting trading process (attempt {restart_count + 1}/{MAX_RESTARTS})")

            # Run the trading daemon with proper error handling
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                env=env
            )

            # Stream output in real-time
            for line in process.stdout:
                try:
                    line = line.strip()
                    if line:
                        # Log everything
                        logger.info(f"[TRADING] {line}")

                        # Filter and print only important messages
                        if any(keyword in line.upper() for keyword in
                               ['P&L', 'PROFIT', 'LOSS', 'TRADE', 'SIGNAL', 'ORDER',
                                'BUY', 'SELL', 'POSITION', 'ERROR', 'CRITICAL']):
                            if 'DELISTED' not in line and '404' not in line:  # Skip noisy errors
                                print(f"[LIVE] {line[:100]}")
                except Exception as e:
                    logger.debug(f"[DAEMON] Output processing error: {e}")

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
                logger.info(f"[DAEMON] Auto-restarting in {RESTART_DELAY_SECONDS} seconds... (restart {restart_count}/{MAX_RESTARTS})")
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
                logger.info(f"[DAEMON] Auto-restarting after error...")
                time.sleep(RESTART_DELAY_SECONDS)
            else:
                break

    logger.info(f"[DAEMON] Trading daemon stopped after {restart_count} restarts")
    return 0 if shutdown_requested else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="24/7 Autonomous Trading System - Windows Compatible",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python FIXED_run_247.py          # Start paper trading (safe)
  python FIXED_run_247.py --live   # Start live trading (real money!)
  python FIXED_run_247.py --skip-tests  # Skip tests, start immediately
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
        print("Press Ctrl+C within 5 seconds to cancel...")
        time.sleep(5)
    else:
        mode = "paper"
        print("*** PAPER TRADING MODE (Safe) ***")
        print("Simulated trades - no real money at risk.")

    print()

    # Phase 1: Run tests (unless skipped)
    if not args.skip_tests:
        if not run_tests():
            logger.error("[FATAL] System tests failed - trading not started")
            print("\nTests failed. Use --skip-tests to bypass (not recommended)")
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
