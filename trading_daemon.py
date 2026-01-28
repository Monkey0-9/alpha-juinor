#!/usr/bin/env python3
"""
24/7 Institutional Trading Daemon

Continuous trading loop with:
- Decision/Trade Triggers: Every 5 seconds
- Data Refresh: Every 30-60 minutes
- Market Hours Awareness
- Kill Switch & Health Monitoring
"""

import os
import sys
import time
import json
import logging
import signal
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.manager import DatabaseManager
from data.collectors.data_router import DataRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/trading_daemon.log')
    ]
)
logger = logging.getLogger("TRADING_DAEMON")

class TradingDaemon:
    """
    24/7 Production Trading Daemon
    """
    def __init__(
        self,
        trigger_interval_sec: int = 5,
        data_refresh_interval_min: int = 30,
        enable_market_hours_only: bool = False
    ):
        self.trigger_interval = trigger_interval_sec
        self.data_refresh_interval = data_refresh_interval_min * 60  # Convert to seconds
        self.enable_market_hours_only = enable_market_hours_only

        # State
        self.running = False
        self.kill_switch_active = False
        self.last_data_refresh = None
        self.cycle_count = 0

        # Components
        self.db = DatabaseManager()
        self.router = DataRouter()

        # Health monitoring
        self.last_heartbeat = datetime.utcnow()
        self.error_count = 0
        self.max_consecutive_errors = 10

        # Load universe
        self._load_universe()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Trading Daemon initialized | Trigger={trigger_interval_sec}s | Data Refresh={data_refresh_interval_min}min")

    def _load_universe(self):
        """Load trading universe"""
        try:
            with open("configs/universe.json", "r") as f:
                universe = json.load(f)
            self.symbols = universe.get("active_tickers", [])
            logger.info(f"Loaded universe: {len(self.symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to load universe: {e}")
            self.symbols = []

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.stop()

    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (NYSE 9:30 AM - 4:00 PM ET)"""
        if not self.enable_market_hours_only:
            return True  # Always active if market hours check is disabled

        from datetime import datetime
        import pytz

        # Get current time in ET
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)

        # Check if weekend
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check if within trading hours (9:30 AM - 4:00 PM ET)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now_et <= market_close

    def _refresh_market_data(self):
        """Refresh market data for all symbols (30-60 min interval)"""
        logger.info(f"[DATA_REFRESH] Refreshing market data for {len(self.symbols)} symbols...")

        refresh_start = time.time()
        success_count = 0

        try:
            # Fetch latest prices in parallel
            latest_prices = self.router.get_latest_prices_parallel(self.symbols)
            success_count = len(latest_prices)

            logger.info(f"[DATA_REFRESH] Completed in {time.time() - refresh_start:.2f}s | Success: {success_count}/{len(self.symbols)}")
            self.last_data_refresh = datetime.utcnow()

        except Exception as e:
            logger.error(f"[DATA_REFRESH] Failed: {e}")

    def _should_refresh_data(self) -> bool:
        """Check if it's time to refresh market data"""
        if self.last_data_refresh is None:
            return True

        elapsed = (datetime.utcnow() - self.last_data_refresh).total_seconds()
        return elapsed >= self.data_refresh_interval

    def _execute_trading_cycle(self):
        """Execute one trading decision cycle (every 5 seconds)"""
        cycle_id = f"cycle_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{self.cycle_count}"

        try:
            # Check kill switch
            if self._check_kill_switch():
                logger.critical("[KILL_SWITCH] Trading halted by kill switch!")
                self.stop()
                return

            # Market hours check
            if not self._is_market_hours():
                logger.debug("[MARKET_HOURS] Outside trading hours, skipping cycle")
                return

            # Refresh data if needed
            if self._should_refresh_data():
                self._refresh_market_data()

            # Import and run main trading logic
            # For now, we'll just log. In production, this would call your main.py logic
            logger.info(f"[CYCLE {self.cycle_count}] Executing trading decisions...")

            # Placeholder: Here you would call your actual trading logic
            # from main import execute_trading_cycle
            # execute_trading_cycle(cycle_id, self.symbols)

            # Update heartbeat
            self.last_heartbeat = datetime.utcnow()
            self.error_count = 0  # Reset error count on success

        except Exception as e:
            self.error_count += 1
            logger.error(f"[CYCLE {self.cycle_count}] Error: {e}")

            if self.error_count >= self.max_consecutive_errors:
                logger.critical(f"[CRITICAL] Max consecutive errors ({self.max_consecutive_errors}) reached. Halting daemon.")
                self.stop()

    def _check_kill_switch(self) -> bool:
        """Check if kill switch file exists"""
        kill_switch_path = Path("runtime/KILL_SWITCH")
        if kill_switch_path.exists():
            self.kill_switch_active = True
            return True
        return False

    def _health_monitor(self):
        """Background thread to monitor daemon health"""
        while self.running:
            elapsed_since_heartbeat = (datetime.utcnow() - self.last_heartbeat).total_seconds()

            if elapsed_since_heartbeat > 60:  # No heartbeat in 60 seconds
                logger.warning(f"[HEALTH] No heartbeat in {elapsed_since_heartbeat:.0f}s. Daemon may be stuck.")

            time.sleep(30)  # Check every 30 seconds

    def start(self):
        """Start the 24/7 trading daemon"""
        self.running = True
        logger.info("=" * 80)
        logger.info("TRADING DAEMON STARTED")
        logger.info("=" * 80)
        logger.info(f"Trigger Interval: {self.trigger_interval}s")
        logger.info(f"Data Refresh: Every {self.data_refresh_interval / 60:.0f} minutes")
        logger.info(f"Market Hours Only: {self.enable_market_hours_only}")
        logger.info(f"Symbols: {len(self.symbols)}")
        logger.info("=" * 80)

        # Start health monitor thread
        health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        health_thread.start()

        # Initial data refresh
        self._refresh_market_data()

        # Main loop
        while self.running:
            try:
                self._execute_trading_cycle()
                self.cycle_count += 1
                time.sleep(self.trigger_interval)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(self.trigger_interval)

    def stop(self):
        """Stop the daemon gracefully"""
        logger.info("Stopping Trading Daemon...")
        self.running = False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="24/7 Institutional Trading Daemon")
    parser.add_argument("--trigger", type=int, default=5, help="Trigger interval in seconds (default: 5)")
    parser.add_argument("--data-refresh", type=int, default=30, help="Data refresh interval in minutes (default: 30)")
    parser.add_argument("--market-hours-only", action="store_true", help="Only trade during market hours")

    args = parser.parse_args()

    daemon = TradingDaemon(
        trigger_interval_sec=args.trigger,
        data_refresh_interval_min=args.data_refresh,
        enable_market_hours_only=args.market_hours_only
    )

    daemon.start()
