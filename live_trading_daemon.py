#!/usr/bin/env python3
"""
24/7 Institutional Live Trading Daemon

Main entry point for the live trading system with:
- Per-second decision loop with visible terminal output
- Market data refresh every 30-60 minutes
- Real-time position and P&L tracking
- Risk metrics dashboard
- Kill switch support
- Graceful shutdown handling

Usage:
    python live_trading_daemon.py --mode paper --tick-interval 1.0
    python live_trading_daemon.py --mode live --data-refresh 30
    python live_trading_daemon.py --status
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config_manager import ConfigManager
from database.manager import DatabaseManager
from orchestration.live_decision_loop import LiveDecisionLoop, LivePosition, LiveSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/live_trading_daemon.log"),
    ],
)
logger = logging.getLogger("LIVE_TRADING_DAEMON")


@dataclass
class TerminalDashboard:
    """Real-time terminal dashboard for live trading status"""

    def __init__(self):
        self.last_render_time = None
        self.render_interval = 1.0  # Update every second

    def render(
        self,
        cycle_count: int,
        uptime: str,
        system_status: str,
        positions: Dict[str, LivePosition],
        signals: Dict[str, LiveSignal],
        last_decision_time: Optional[datetime],
        last_data_refresh: Optional[datetime],
        risk_metrics: Dict[str, float],
        error_count: int,
    ):
        """Render the terminal dashboard"""
        now = datetime.utcnow()

        # Rate limit rendering
        if (
            self.last_render_time
            and (now - self.last_render_time).total_seconds() < self.render_interval
        ):
            return

        self.last_render_time = now

        # Clear screen ( ANSI escape )
        print("\033[2J\033[H", end="")

        HEADER_STR = "════════════════════════════════════════════════════════════════════════════════════════════════════"
        DIVIDER_STR = "────────────────────────────────────────────────────────────────────────────────────────────────────"

        print(HEADER_STR)
        print(
            f"  QUANTITATIVE TRADING ENGINE v7.1.0-RC1  |  INSTITUTIONAL TIER  |  CYCLE: {cycle_count:08d}"
        )
        print(HEADER_STR)

        # 1. CORE SYSTEM STATUS
        print(" [ SYSTEM CORE ]")
        state_color = "\033[92m" if system_status.upper() == "RUNNING" else "\033[93m"
        reset_color = "\033[0m"
        print(f"   UPTIME     : {uptime:<20}    EXEC_MODE: {system_status.upper()}")
        print(f"   ALLOCATION : {len(positions):<20}    SIGNALS  : {len(signals)}")
        print(DIVIDER_STR)

        # 2. MARKET DATA FEED
        last_refresh_str = (
            last_data_refresh.strftime("%Y-%m-%d %H:%M:%S UTC")
            if last_data_refresh
            else "WAITING DATA"
        )
        next_refresh = (
            (last_data_refresh + timedelta(minutes=30)).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            if last_data_refresh
            else "IMMEDIATE"
        )
        print(" [ MARKET FEED SYNCHRONIZATION ]")
        print(f"   LAST SYNC  : {last_refresh_str:<25} NEXT SYNC: {next_refresh}")

        if error_count > 0:
            print(f"   FEED STATE : [WARN] {error_count} EXCEPTIONS LOGGED")
        else:
            print(f"   FEED STATE : [OK] NOMINAL")
        print(DIVIDER_STR)

        # 3. PORTFOLIO
        if positions:
            print(f" [ PORTFOLIO ALLOCATION ]  ({len(positions)} ASSETS)")
            print(
                f"   {'TICKER':<8} | {'EXPOSURE':>12} | {'ENTRY':>10} | {'LAST_PX':>10} | {'UNREAL_PNL':>12} | {'RETURN':>9} | {'CNVCTN':>8}"
            )
            print(
                f"   {'':-<8}-+-{'-':-<12}-+-{'-':-<10}-+-{'-':-<10}-+-{'-':-<12}-+-{'-':-<9}-+-{'-':-<8}"
            )

            for symbol, pos in sorted(positions.items()):
                signal = signals.get(symbol)
                conviction = f"{signal.conviction:.2f}" if signal else "N/A"
                pnl_prefix = "+" if pos.unrealized_pnl >= 0 else ""
                print(
                    f"   {symbol:<8} | {pos.quantity:>12.4f} | {pos.entry_price:>10.2f} | {pos.current_price:>10.2f} | {pnl_prefix}{pos.unrealized_pnl:>12.2f} | {pos.pnl_pct:>8.2f}% | {conviction:>8}"
                )

            total_pnl = sum(pos.unrealized_pnl for pos in positions.values())
            total_value = sum(pos.market_value for pos in positions.values())
            ret_pct = (total_pnl / total_value * 100) if total_value > 0 else 0
            print(
                f"   {'':-<8}-+-{'-':-<12}-+-{'-':-<10}-+-{'-':-<10}-+-{'-':-<12}-+-{'-':-<9}-+-{'-':-<8}"
            )
            print(
                f"   {'AGG':<8} | {' ':>12} | {' ':>10} | {' ':>10} | {total_pnl:>12.2f} | {ret_pct:>8.2f}% | {' ':>8}"
            )
        else:
            print(" [ PORTFOLIO ALLOCATION ]")
            print("   STATUS: NO CAPITAL DEPLOYED")

        print(DIVIDER_STR)

        # 4. LIVE SIGNALS (ALPHA)
        print(f" [ ALPHA GENERATION ARRAY ]  ({len(signals)} VECTORS)")
        print(
            f"   {'TICKER':<8} | {'DIRECTIVE':>10} | {'MU_EST':>10} | {'VOL_EST':>10} | {'CNVCTN':>10} | {'DATA_Q':>8}"
        )
        print(
            f"   {'':-<8}-+-{'-':-<10}-+-{'-':-<10}-+-{'-':-<10}-+-{'-':-<10}-+-{'-':-<8}"
        )

        for symbol, sig in sorted(signals.items()):
            signal_str = (
                "EXEC_BUY"
                if sig.signal == "EXECUTE_BUY"
                else (
                    "EXEC_SELL"
                    if sig.signal == "EXECUTE_SELL"
                    else ("HOLD" if sig.signal == "HOLD" else "REJECTED")
                )
            )
            print(
                f"   {symbol:<8} | {signal_str:>10} | {sig.mu_hat:>10.5f} | {sig.sigma_hat:>10.5f} | {sig.conviction:>10.3f} | {sig.data_quality:>8.3f}"
            )

        print(DIVIDER_STR)

        # 5. EXECUTION & RISK
        buy_count = sum(1 for s in signals.values() if s.signal == "EXECUTE_BUY")
        sell_count = sum(1 for s in signals.values() if s.signal == "EXECUTE_SELL")
        hold_count = sum(1 for s in signals.values() if s.signal == "HOLD")

        leverage = risk_metrics.get("leverage", 0.0)
        total_exposure = risk_metrics.get("total_exposure", 0.0)

        print(" [ GLOBAL RISK & EXECUTION TELEMETRY ]")
        print(f"   GROSS LVRG : {leverage:>8.4f}x           LONG  ACQ : {buy_count:>6}")
        print(
            f"   GROSS EXP  : {total_exposure:>8.2f} USD        SHORT DIS : {sell_count:>6}"
        )
        print(
            f"   OPEN POS   : {risk_metrics.get('positions_count', 0):>8}            HOLD / NO : {hold_count:>6}"
        )

        print(HEADER_STR)
        print(
            f"  LAST_SYNC: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC   |   INTERRUPT CAUSE: SIGINT (Ctrl+C)"
        )
        print(HEADER_STR)


class LiveTradingDaemon:
    """
    24/7 Live Trading Daemon

    Responsibilities:
    - Run LiveDecisionLoop in a separate thread
    - Render real-time terminal dashboard
    - Handle kill switch and shutdown signals
    - Monitor system health
    """

    def __init__(
        self,
        tick_interval: float = 1.0,
        data_refresh_interval_min: int = 30,
        paper_mode: bool = True,
        market_hours_only: bool = True,
        symbols: Optional[List[str]] = None,
        headless: bool = False,
        bypass_governance: bool = False,
    ):
        self.tick_interval = tick_interval
        self.data_refresh_interval = data_refresh_interval_min
        self.paper_mode = paper_mode
        self.market_hours_only = market_hours_only
        self.symbols = symbols
        self.headless = headless
        self.bypass_governance = bypass_governance

        # State
        self.running = False
        self.paused = False
        self.daemon_start_time = datetime.utcnow()

        # Components
        self.db = DatabaseManager()
        self.config = ConfigManager()
        self.dashboard = TerminalDashboard()

        # Decision loop
        self.decision_loop: Optional[LiveDecisionLoop] = None
        self.decision_thread: Optional[threading.Thread] = None

        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(
            f"LiveTradingDaemon initialized | Tick: {tick_interval}s | Data Refresh: {data_refresh_interval_min}min"
        )

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.stop()
        sys.exit(0)

    def _check_kill_switch(self) -> bool:
        """Check if kill switch file exists"""
        kill_switch_path = Path("runtime/KILL_SWITCH")
        if kill_switch_path.exists():
            logger.warning("Kill switch activated. Pausing trading.")
            return True
        return False

    def _decision_thread_func(self):
        """Run the decision loop in a separate thread"""
        try:
            self.decision_loop.start()
        except Exception as e:
            logger.error(f"Decision loop error: {e}")

    def start(self):
        """Start the live trading daemon"""
        self.running = True

        print("\n")
        logger.info("=" * 80)
        logger.info("24/7 INSTITUTIONAL LIVE TRADING DAEMON STARTED")
        logger.info("=" * 80)
        logger.info(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"Tick Interval: {self.tick_interval}s (per-second decisions)")
        logger.info(f"Data Refresh: Every {self.data_refresh_interval} minutes")
        logger.info(f"Market Hours Only: {self.market_hours_only}")
        logger.info("=" * 80)

        # Create and start decision loop
        self.decision_loop = LiveDecisionLoop(
            tick_interval=self.tick_interval,
            data_refresh_interval_min=self.data_refresh_interval,
            paper_mode=self.paper_mode,
            market_hours_only=self.market_hours_only,
            symbols=self.symbols,
            bypass_governance=self.bypass_governance,
        )

        # Start decision loop in background thread
        self.decision_thread = threading.Thread(
            target=self._decision_thread_func, daemon=True
        )
        self.decision_thread.start()

        # Give the loop time to initialize
        time.sleep(2)

        # Main monitoring loop
        try:
            while self.running:
                # Check kill switch
                if self._check_kill_switch():
                    self.paused = True
                    self.decision_loop.paused = True
                else:
                    if self.paused:
                        logger.info("Kill switch released. Resuming trading.")
                        self.paused = False
                        self.decision_loop.paused = False

                # Get current status
                status = self.decision_loop.get_status()

                # Calculate uptime
                uptime = self._get_uptime()

                # Render dashboard (unless headless)
                if not self.headless:
                    self.dashboard.render(
                        cycle_count=status["cycle_count"],
                        uptime=uptime,
                        system_status=status["system_status"],
                        positions=status["positions"],
                        signals=status["signals"],
                        last_decision_time=(
                            datetime.fromisoformat(status["last_decision_time"])
                            if status["last_decision_time"]
                            else None
                        ),
                        last_data_refresh=(
                            datetime.fromisoformat(status["last_data_refresh"])
                            if status["last_data_refresh"]
                            else None
                        ),
                        risk_metrics=status["risk_metrics"],
                        error_count=status["error_count"],
                    )
                else:
                    # Headless mode: log heartbeat
                    logger.debug(
                        f"[HEARTBEAT] Cycle #{status['cycle_count']} | Signals: {len(status['signals'])}"
                    )

                # Sleep for dashboard refresh
                time.sleep(1.0)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Daemon error: {e}")
        finally:
            self.stop()

    def _get_uptime(self) -> str:
        """Get formatted uptime string"""
        elapsed = datetime.utcnow() - self.daemon_start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"

    def stop(self):
        """Stop the live trading daemon"""
        logger.info("Stopping live trading daemon...")
        self.running = False

        if self.decision_loop:
            self.decision_loop.stop()

        logger.info("Live trading daemon stopped.")

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status"""
        if self.decision_loop:
            return self.decision_loop.get_status()
        return {"system_status": "NOT_INITIALIZED"}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="24/7 Institutional Live Trading Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in paper mode with 1-second tick
  python live_trading_daemon.py --mode paper --tick-interval 1.0

  # Run in live mode with 30-minute data refresh
  python live_trading_daemon.py --mode live --data-refresh 30

  # Run headless (no terminal dashboard)
  python live_trading_daemon.py --mode paper --headless

  # Check daemon status
  python live_trading_daemon.py --status
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=1.0,
        help="Decision loop tick interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--data-refresh",
        type=int,
        default=30,
        help="Data refresh interval in minutes (default: 30)",
    )
    parser.add_argument(
        "--market-hours-only",
        action="store_true",
        default=True,
        help="Only trade during market hours (default: True)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without terminal dashboard"
    )
    parser.add_argument("--status", action="store_true", help="Check daemon status")
    parser.add_argument(
        "--symbols", type=str, help="Comma-separated list of symbols (optional)"
    )
    parser.add_argument(
        "--bypass-governance", action="store_true", help="Bypass governance checks for continuous operation"
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = args.symbols.split(",") if args.symbols else None

    # Create daemon
    daemon = LiveTradingDaemon(
        tick_interval=args.tick_interval,
        data_refresh_interval_min=args.data_refresh,
        paper_mode=(args.mode == "paper"),
        market_hours_only=args.market_hours_only,
        symbols=symbols,
        headless=args.headless,
        bypass_governance=args.bypass_governance,
    )

    if args.status:
        # Just show status
        status = daemon.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        # Start daemon
        daemon.start()


if __name__ == "__main__":
    main()
