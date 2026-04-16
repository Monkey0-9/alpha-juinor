
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional

from mini_quant_fund.core.engine.loop import TradingLoop
from mini_quant_fund.core.ui.dashboard import TerminalDashboard, NullUI
from mini_quant_fund.core.ui.logger import InstitutionalLogger
from mini_quant_fund.strategies.institutional_strategy import InstitutionalStrategy
from mini_quant_fund.data.collectors.data_router import DataRouter
from mini_quant_fund.execution.alpaca_handler import AlpacaExecutionHandler
from mini_quant_fund.execution.advanced_execution import get_execution_engine, ExecutionAlgo
from decimal import Decimal
from mini_quant_fund.database.manager import DatabaseManager
from mini_quant_fund.core.monitoring.infrastructure_guard import get_infra_guard
from mini_quant_fund.core.global_session_tracker import get_global_session_tracker
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Orchestrator:
    """The high-level coordinator of the trading system."""
    
    def __init__(self, mode="paper", headless=False):
        self.mode = mode
        self.headless = headless
        self.loop = TradingLoop()
        self.ui = NullUI() if headless else TerminalDashboard()
        self.start_time = datetime.utcnow()
        self.cycle_count = 0
        self.strategy = InstitutionalStrategy()
        self.router = DataRouter()
        self.broker = AlpacaExecutionHandler(paper=(mode == "paper"))
        self.db = DatabaseManager()
        self.tickers = self._load_universe()
        self.execution_engine = get_execution_engine()
        self.active_plans = []
        self.infra_guard = get_infra_guard()
        self.session_tracker = get_global_session_tracker()
        
        # Setup signals
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        
    def _shutdown(self, signum, frame):
        logger.info(f"Shutdown signal {signum} received. Stopping system...")
        self.stop()
        sys.exit(0)

    def _tick(self):
        """Single decision tick: State -> Data -> Strategy -> Execution."""
        self.cycle_count += 1
        
        try:
            # 0. Market Session Check
            if not self.session_tracker.is_market_open("NYSE") and self.mode == "live":
                logger.info("Market is CLOSED. Skipping cycle.")
                return

            # 1. Fetch latest market data
            # We use a 5-day lookback for the live loop as per Phase 0 guards
            start_date = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
            market_data = self.router.get_panel_parallel(self.tickers, start_date=start_date)
            
            if market_data.empty:
                logger.warning("No market data received. Skipping cycle.")
                return

            # 2. Generate signals
            signals_df = self.strategy.generate_signals(market_data)
            
            # 3. Execution logic: Plan generation and slice execution
            if not signals_df.empty:
                latest_signals = signals_df.iloc[-1]
                for ticker, conviction in latest_signals.items():
                    # Avoid creating duplicate plans for the same ticker if one is active
                    if any(p.symbol == ticker for p in self.active_plans): continue

                    if conviction > 0.8:  # Very High Conviction
                        price = Decimal(str(market_data[ticker]["Close"].iloc[-1]))
                        plan = self.execution_engine.create_execution_plan(ticker, "BUY", 100, price)
                        self.active_plans.append(plan)
                        logger.info(f"[EXEC] Created BUY plan for {ticker} using {plan.algorithm}")
                    elif conviction < 0.2:
                        price = Decimal(str(market_data[ticker]["Close"].iloc[-1]))
                        plan = self.execution_engine.create_execution_plan(ticker, "SELL", 100, price)
                        self.active_plans.append(plan)
                        logger.info(f"[EXEC] Created SELL plan for {ticker} using {plan.algorithm}")

            # 3b. Execute pending slices
            for plan in self.active_plans[:]:
                current_time = datetime.utcnow()
                for order in plan.orders:
                    if order.execute_after <= current_time and not getattr(order, "executed", False):
                        self.broker.submit_order(order.symbol, order.slice_quantity, order.side.lower())
                        order.executed = True
                        logger.info(f"[EXEC] Executed slice {order.slice_number}/{order.total_slices} for {order.symbol}")
                
                # Remove completed plans
                if all(getattr(o, "executed", False) for o in plan.orders):
                    self.active_plans.remove(plan)
                    logger.info(f"[EXEC] Completed execution plan for {plan.symbol}")

            # 4. Update UI State
            account = self.broker.get_account() or {}
            positions_list = self.broker.get_positions() or []
            
            state = {
                "status": "RUNNING",
                "cycle_count": self.cycle_count,
                "total_pnl": float(account.get("unrealized_intraday_pl", 0)),
                "positions": {p["symbol"]: p for p in positions_list if isinstance(p, dict)}
            }
            self.ui.update(state)
            
        except Exception as e:
            logger.error(f"Cycle #{self.cycle_count} failed: {e}", exc_info=True)

    def _load_universe(self) -> list:
        """Load target tickers from universe configuration."""
        import json
        try:
            with open("src/mini_quant_fund/configs/universe.json", "r") as f:
                return json.load(f).get("active_tickers", ["AAPL", "MSFT", "SPY"])
        except Exception:
            return ["AAPL", "MSFT", "SPY"]


    def _refresh(self):
        """Data refresh task."""
        logger.info("Performing scheduled market data refresh...")

    def start(self):
        """Launch the institutional trading system."""
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        InstitutionalLogger.setup(log_file=os.path.join(log_dir, f"quant_fund_{self.mode}.log"))
        logger.info(f"System launching in {self.mode} mode...")
        
        # Pre-flight checks
        if not self.infra_guard.verify_all():
            logger.critical("Pre-flight checks FAILED. Infrastructure unreachable.")
            if self.mode == "live":
                sys.exit(1)
            else:
                logger.warning("Continuing in paper mode despite infra failures.")

        self.loop.start(tick_func=self._tick, refresh_func=self._refresh)
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.loop.stop()
        logger.info("System shutdown complete.")
