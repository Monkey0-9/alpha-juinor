#!/usr/bin/env python3
"""
Start Live Trading Daemon with Trade Execution Enabled

This script:
1. Sets up the environment for live trading
2. Enables EXECUTE_TRADES flag
3. Removes any kill switches
4. Starts the 24/7 trading daemon
5. Provides real-time dashboard

Usage:
    python start_trading.py --mode paper
    python start_trading.py --mode live          # Real trading (careful!)
    python start_trading.py --status              # Check trading status
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("TRADING_LAUNCHER")


def setup_trading_environment(paper_mode: bool = True):
    """Setup environment for trading"""
    logger.info("=" * 80)
    logger.info("[SYS] TRADING ENVIRONMENT SETUP")
    logger.info("=" * 80)

    # 1. Enable trade execution
    os.environ["EXECUTE_TRADES"] = "true"
    logger.info("[OK] EXECUTE_TRADES=true (trades WILL be executed)")

    # 2. Set paper/live mode
    if paper_mode:
        os.environ["TRADING_MODE"] = "paper"
        logger.info("[OK] TRADING_MODE=paper (no real money)")
    else:
        os.environ["TRADING_MODE"] = "live"
        logger.warning("[WARN] TRADING_MODE=live (REAL MONEY - be careful!)")

    # 3. Remove kill switch if present
    kill_switch_path = Path("runtime/KILL_SWITCH")
    if kill_switch_path.exists():
        kill_switch_path.unlink()
        logger.info("[OK] Removed kill switch (trading enabled)")
    else:
        logger.info("[OK] No kill switch present")

    # 4. Remove kill_switch_local.state
    local_kill_switch = Path("runtime/kill_switch_local.state")
    if local_kill_switch.exists():
        local_kill_switch.unlink()
        logger.info("[OK] Cleared local kill switch state")

    # 5. Ensure runtime directories exist
    Path("runtime").mkdir(exist_ok=True)
    Path("runtime/logs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    logger.info("[OK] Runtime directories ready")

    # 6. Print environment summary
    logger.info("\n" + "=" * 80)
    logger.info("[STAT] TRADING ENVIRONMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"EXECUTE_TRADES: {os.environ.get('EXECUTE_TRADES')}")
    logger.info(f"TRADING_MODE: {os.environ.get('TRADING_MODE')}")
    logger.info(
        f"Alpaca API Key: {'[OK] Set' if os.environ.get('ALPACA_API_KEY') else '[ERR] Missing'}"
    )
    logger.info(
        f"Alpaca Secret Key: {'[OK] Set' if os.environ.get('ALPACA_SECRET_KEY') else '[ERR] Missing'}"
    )
    logger.info("=" * 80 + "\n")


def check_trading_status():
    """Check if trading is properly enabled"""
    logger.info("=" * 80)
    logger.info("[CHK] CHECKING TRADING STATUS")
    logger.info("=" * 80)

    checks = {
        "EXECUTE_TRADES enabled": os.environ.get("EXECUTE_TRADES", "").lower()
        == "true",
        "Runtime directory exists": Path("runtime").exists(),
        "No kill switch present": not Path("runtime/KILL_SWITCH").exists(),
        "Database accessible": Path("runtime/institutional_trading.db").exists(),
    }

    for check_name, passed in checks.items():
        status = "[OK]" if passed else "[ERR]"
        logger.info(f"{status} {check_name}")

    all_passed = all(checks.values())
    logger.info(
        "\n" + ("[OK] All checks passed!" if all_passed else "[ERR] Some checks failed")
    )
    logger.info("=" * 80 + "\n")

    return all_passed


def start_live_trading_daemon(paper_mode: bool = True):
    """Start the live trading daemon"""
    logger.info("[INIT] STARTING LIVE TRADING DAEMON...\n")

    # Import and start
    from live_trading_daemon import LiveTradingDaemon

    try:
        daemon = LiveTradingDaemon(
            tick_interval=1.0,
            data_refresh_interval_min=30,
            paper_mode=paper_mode,
            market_hours_only=False,  # 24/7 trading
            headless=False,
        )

        daemon.start()

    except KeyboardInterrupt:
        logger.info("\n\n[HALT] Trading daemon stopped by user")
    except Exception as e:
        logger.error(f"[ERR] Error in trading daemon: {e}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Start Live Trading with Trade Execution Enabled"
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode: paper (default) or live",
    )
    parser.add_argument(
        "--status", action="store_true", help="Check trading status without starting"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force start even if checks fail"
    )

    args = parser.parse_args()

    # Setup environment
    setup_trading_environment(paper_mode=args.mode == "paper")

    # Check status
    if args.status:
        check_trading_status()
        return

    # Verify all systems go
    status_ok = check_trading_status()

    if not status_ok and not args.force:
        logger.warning("[WARN] Some checks failed. Use --force to start anyway.")
        sys.exit(1)

    start_live_trading_daemon(paper_mode=args.mode == "paper")


if __name__ == "__main__":
    main()
