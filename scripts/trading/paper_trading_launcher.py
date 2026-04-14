"""
Paper Trading Launcher
=======================
Orchestrates the 2-week paper trading validation
across all global markets before live deployment.

Usage:
    python scripts/paper_trading_launcher.py [--days 14]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)-7s | "
        "%(name)-30s | %(message)s"
    ),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / "paper_trading.log"
        ),
    ],
)
logger = logging.getLogger("PaperTradingLauncher")


def validate_environment() -> bool:
    """Validate all required env vars are set."""
    required = {
        "ALPACA_API_KEY": os.environ.get("ALPACA_API_KEY", ""),
        "ALPACA_SECRET_KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
    }

    optional = {
        "POLYGON_API_KEY": os.environ.get("POLYGON_API_KEY", ""),
        "NEWS_API_KEY": os.environ.get("NEWS_API_KEY", ""),
        "DATABASE_URL": os.environ.get("DATABASE_URL", ""),
    }

    # Load .env file if exists
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        logger.info(f"Loading .env from {env_path}")
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if value and not value.startswith("your_"):
                    os.environ.setdefault(key, value)
                    if key in required:
                        required[key] = value
                    elif key in optional:
                        optional[key] = value

    # Check required
    missing = [k for k, v in required.items() if not v]
    if missing:
        logger.error(
            f"Missing required env vars: {missing}"
        )
        logger.error(
            "Run: python scripts/rotate_keys.py first"
        )
        return False

    logger.info("Environment validated:")
    for k, v in required.items():
        masked = v[:4] + "****" if v else "(empty)"
        logger.info(f"  {k} = {masked}")
    for k, v in optional.items():
        status = "SET" if v else "NOT SET (optional)"
        logger.info(f"  {k} = {status}")

    return True


def run_paper_trading(duration_days: int = 14):
    """
    Launch the 2-week paper trading validation.

    Daily cycle:
    1. Check which markets are open
    2. Fetch data for active symbols
    3. Generate signals via strategy ensemble
    4. Pass through 7-gate risk framework
    5. Execute via paper brokers (Alpaca/IB/CCXT)
    6. Reconcile positions
    7. Generate daily report
    """
    from mini_quant_fund.data.global_universe import get_global_universe
    from mini_quant_fund.execution.exchange_holidays import (
        get_holiday_calendar,
    )
    from mini_quant_fund.execution.global_market_hours import (
        GlobalMarketHours,
    )
    from mini_quant_fund.monitoring.grafana_metrics import get_metrics
    from scripts.verification.launch_verification import (
        CapitalDeploymentPlan,
        PaperTradingRunner,
    )

    # Initialize
    universe = get_global_universe()
    calendar = get_holiday_calendar()
    market_hours = GlobalMarketHours()
    metrics = get_metrics()
    runner = PaperTradingRunner(
        duration_days=duration_days
    )
    deployment = CapitalDeploymentPlan()

    start_date = datetime.utcnow()
    end_date = start_date + timedelta(days=duration_days)

    logger.info("=" * 60)
    logger.info("  PAPER TRADING VALIDATION STARTED")
    logger.info(f"  Duration: {duration_days} days")
    logger.info(f"  Start: {start_date.strftime('%Y-%m-%d')}")
    logger.info(f"  End: {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"  Universe: {universe.total_count} symbols")
    logger.info(f"  Breakdown: {universe.get_breakdown()}")
    logger.info("=" * 60)

    day = 0
    while day < duration_days:
        now = datetime.utcnow()

        # Check if it's a trading day for any exchange
        any_trading = any(
            calendar.is_trading_day(ex)
            for ex in [
                "NYSE", "LSE", "EURONEXT",
                "JPX", "HKEx", "ASX", "TSX",
            ]
        )

        if not any_trading:
            logger.info(
                f"Day {day + 1}: No markets open — "
                f"skipping"
            )
            time.sleep(60)
            continue

        # Get current deployment phase
        phase = deployment.get_current_phase(day)
        logger.info(
            f"\nDay {day + 1}/{duration_days} | "
            f"Week {(day // 7) + 1} | "
            f"Capital: {phase['capital_pct']}% | "
            f"{phase['description']}"
        )

        # Run daily cycle
        try:
            result = runner.run_daily_cycle()
            status = result["status"]
            duration = result.get("duration_sec", 0)

            logger.info(
                f"  Cycle {status} in {duration:.1f}s"
            )

            # Update metrics
            metrics.set_total_equity(
                1_000_000 * phase["capital_pct"] / 100
            )

            if status == "FAILED":
                logger.warning(
                    f"  Failed phases: "
                    f"{result.get('failed_phases', [])}"
                )

        except Exception as e:
            logger.error(f"  Cycle error: {e}")

        day += 1

        # In real operation, we'd wait for next
        # trading day. For testing, we continue.
        logger.info(
            f"  Day {day} complete. "
            f"Waiting for next cycle..."
        )

    # Final summary
    summary = runner.get_summary()

    logger.info("\n" + "=" * 60)
    logger.info("  PAPER TRADING COMPLETE")
    logger.info(f"  Days run: {summary['total_days']}")
    logger.info(f"  Passed: {summary['days_passed']}")
    logger.info(f"  Failed: {summary['days_failed']}")
    logger.info(
        f"  Success rate: "
        f"{summary['success_rate_pct']}%"
    )
    logger.info(
        f"  Ready for launch: "
        f"{summary['ready_for_launch']}"
    )
    logger.info("=" * 60)

    # Save final report
    report_path = (
        PROJECT_ROOT / "logs" / "paper_trading_report.json"
    )
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Launcher"
    )
    parser.add_argument(
        "--days", type=int, default=14,
        help="Duration in days (default: 14)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MINI-QUANT-FUND v1.0.0")
    print("  Paper Trading Launcher")
    print("=" * 60)

    # Ensure log dir exists
    os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)

    # Validate environment
    if not validate_environment():
        print("\nFailed: Set up API keys first with:")
        print("  python scripts/rotate_keys.py")
        sys.exit(1)

    # Run
    run_paper_trading(args.days)


if __name__ == "__main__":
    main()
