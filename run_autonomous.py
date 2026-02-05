"""
Autonomous Trading Orchestrator - Full System Integration.

Combines ALL models and modules for fully autonomous trading:
- Full market scanning
- Multi-model signal generation
- Automatic position sizing
- Auto-execution
- Real-time risk management

This is the main entry point for autonomous trading.
"""

import logging
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_config import setup_logging

logger = setup_logging("AUTONOMOUS_TRADER", log_dir="runtime/logs")


class AutonomousTradingOrchestrator:
    """
    Master orchestrator for fully autonomous trading.

    Integrates:
    - Universe Scanner (166+ stocks)
    - Full Market Alpha (parallel signals)
    - All ML Models (HMM, Deep Ensemble, RL, etc.)
    - Risk Management
    - Auto Execution
    """

    def __init__(self, config_path: str = "configs/daily_trading_config.json"):
        self.config_path = config_path
        self.config = self._load_config()

        # Components (lazy loaded)
        self._scanner = None
        self._alpha_generator = None
        self._auto_trader = None
        self._yahoo_provider = None

        # State
        self.running = False
        self.last_scan_time = None
        self.positions = {}
        self.signals = {}

    def _load_config(self) -> Dict:
        """Load daily trading configuration."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            logger.warning(f"Config load failed: {e}, using defaults")
            return {
                "auto_execute": False,
                "max_positions": 50,
                "position_size_pct": 0.02,
                "min_signal_confidence": 0.6
            }

    def reload_config(self):
        """Hot-reload configuration (call daily or on demand)."""
        self.config = self._load_config()
        logger.info("Configuration reloaded")

    @property
    def scanner(self):
        """Lazy load universe scanner."""
        if self._scanner is None:
            from data.universe_scanner import get_universe_scanner
            self._scanner = get_universe_scanner()
        return self._scanner

    @property
    def alpha_generator(self):
        """Lazy load alpha generator."""
        if self._alpha_generator is None:
            from alpha_families.full_market_alpha import get_market_alpha_generator
            self._alpha_generator = get_market_alpha_generator()
        return self._alpha_generator

    @property
    def auto_trader(self):
        """Lazy load auto trader."""
        if self._auto_trader is None:
            from execution.auto_trader import get_auto_trader
            self._auto_trader = get_auto_trader()
            self._auto_trader.initialize()
        return self._auto_trader

    @property
    def yahoo(self):
        """Lazy load Yahoo provider."""
        if self._yahoo_provider is None:
            from data.providers.yahoo_finance import get_yahoo_provider
            self._yahoo_provider = get_yahoo_provider()
        return self._yahoo_provider

    def scan_full_market(self) -> List[str]:
        """Scan entire market and return tradable symbols."""
        logger.info("=" * 60)
        logger.info("[SCAN] Starting full market scan...")

        # Refresh universe
        num_assets = self.scanner.refresh_universe()
        symbols = self.scanner.get_tradable_symbols()

        logger.info(f"[SCAN] Found {num_assets} tradable assets")

        return symbols

    def fetch_market_data(self, symbols: List[str], days: int = 60) -> Dict[str, pd.Series]:
        """Fetch price data for all symbols."""
        logger.info(f"[DATA] Fetching data for {len(symbols)} symbols...")

        market_data = {}
        batch_size = 50

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]

            for symbol in batch:
                try:
                    data = self.yahoo.get_historical_data(symbol, days=days)
                    if data is not None and len(data) > 20:
                        market_data[symbol] = data["close"]
                except Exception as e:
                    logger.debug(f"Fetch failed for {symbol}: {e}")

            logger.info(f"[DATA] Fetched {min(i+batch_size, len(symbols))}/{len(symbols)}")
            time.sleep(0.3)  # Rate limit

        return market_data

    def generate_signals(self, market_data: Dict[str, pd.Series]):
        """Generate signals using ALL models."""
        logger.info("[SIGNALS] Generating signals using all models...")

        # Use full market alpha generator
        scan_result = self.alpha_generator.scan_market(market_data)

        logger.info(
            f"[SIGNALS] Scanned {scan_result.total_scanned} stocks, "
            f"{len(scan_result.long_candidates)} long, "
            f"{len(scan_result.short_candidates)} short"
        )

        self.signals = scan_result
        return scan_result

    def apply_additional_models(self, symbols: List[str], market_data: Dict):
        """Apply additional ML models for enhanced signals."""
        enhanced_signals = {}

        # Apply HMM if enabled
        if self.config.get("models_enabled", {}).get("hmm", True):
            try:
                from ml.hmm_predictor import get_hmm_predictor
                hmm = get_hmm_predictor()
                for symbol in symbols[:20]:  # Top 20
                    if symbol in market_data:
                        returns = market_data[symbol].pct_change().dropna()
                        if len(returns) > 50:
                            hmm.fit(returns)
                            pred = hmm.predict(returns)
                            enhanced_signals[symbol] = {
                                "hmm_state": pred.current_state.name,
                                "hmm_confidence": pred.confidence
                            }
            except Exception as e:
                logger.debug(f"HMM enhancement failed: {e}")

        return enhanced_signals

    def execute_trades(self, top_signals: List, current_prices: Dict[str, float]):
        """Execute trades based on signals."""
        if not self.config.get("auto_execute", False):
            logger.info("[EXEC] Auto-execute DISABLED - showing signals only")
            return 0, 0

        logger.info("[EXEC] Executing trades...")

        # Get account value
        account = self.auto_trader.get_account_info()
        account_value = account.get("equity", 100000)

        # Generate orders
        orders = self.auto_trader.generate_orders(
            top_signals,
            account_value,
            current_prices
        )

        # Execute
        filled, failed = self.auto_trader.execute_orders(orders)

        logger.info(f"[EXEC] Orders: {filled} filled, {failed} failed")

        return filled, failed

    def manage_risk(self):
        """Apply risk management rules."""
        logger.info("[RISK] Checking risk limits...")

        risk_limits = self.config.get("risk_limits", {})

        # Close losing positions
        max_loss = risk_limits.get("max_position_loss", 0.05)
        closed = self.auto_trader.close_losing_positions(max_loss_pct=-max_loss)

        if closed > 0:
            logger.warning(f"[RISK] Closed {closed} losing positions")

        return closed

    def run_trading_cycle(self) -> Dict:
        """Run a complete trading cycle."""
        cycle_start = time.time()

        logger.info("=" * 60)
        logger.info(f"[CYCLE] Starting at {datetime.now().strftime('%H:%M:%S')}")
        logger.info("=" * 60)

        # 1. Reload config (hot reload)
        self.reload_config()

        # 2. Scan market
        if self.config.get("use_full_market_scan", True):
            symbols = self.scan_full_market()
        else:
            # Use configured universe
            symbols = self.scanner.get_tradable_symbols()

        # FULL MARKET MODE - No symbol limit
        # Process ALL symbols for comprehensive market analysis
        logger.info(f"[FULL_MARKET] Processing all {len(symbols)} symbols")

        # 3. Fetch data
        market_data = self.fetch_market_data(symbols)

        if not market_data:
            logger.error("[CYCLE] No market data - skipping cycle")
            return {"status": "failed", "reason": "no_data"}

        # 4. Generate signals
        scan_result = self.generate_signals(market_data)

        # 5. Get current prices
        current_prices = {
            sym: prices.iloc[-1]
            for sym, prices in market_data.items()
        }

        # 6. Display top opportunities
        logger.info("\n[TOP OPPORTUNITIES]")
        logger.info("-" * 50)

        for i, signal in enumerate(scan_result.top_picks[:10]):
            direction = "LONG " if signal.signal_type == "LONG" else "SHORT"
            logger.info(
                f"  {i+1}. {direction} {signal.symbol:6s} | "
                f"Score: {signal.alpha_score:+.2f} | "
                f"Conf: {signal.confidence:.0%}"
            )

        # 7. Execute if enabled
        filled, failed = self.execute_trades(
            scan_result.top_picks[:20],
            current_prices
        )

        # 8. Risk management
        closed = self.manage_risk()

        # 9. Summary
        cycle_time = time.time() - cycle_start

        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "symbols_scanned": len(market_data),
            "signals_generated": scan_result.total_scanned,
            "long_candidates": len(scan_result.long_candidates),
            "short_candidates": len(scan_result.short_candidates),
            "orders_filled": filled,
            "orders_failed": failed,
            "positions_closed": closed,
            "cycle_time_seconds": cycle_time
        }

        logger.info(f"\n[CYCLE] Complete in {cycle_time:.1f}s")

        return result

    def run_continuous(self, interval_minutes: int = 30):
        """Run continuous trading loop."""
        self.running = True

        logger.info(f"[AUTONOMOUS] Starting continuous trading (every {interval_minutes} min)")

        while self.running:
            try:
                result = self.run_trading_cycle()
                logger.info(f"[AUTONOMOUS] Cycle result: {result['status']}")

            except KeyboardInterrupt:
                logger.info("[AUTONOMOUS] Stopping...")
                self.running = False
                break

            except Exception as e:
                logger.error(f"[AUTONOMOUS] Cycle error: {e}")

            if self.running:
                logger.info(f"\n[AUTONOMOUS] Next cycle in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

    def stop(self):
        """Stop the trading loop."""
        self.running = False
        logger.info("[AUTONOMOUS] Stop requested")


def main():
    """Main entry point for autonomous trading."""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Trading System")
    parser.add_argument("--single", action="store_true", help="Run single cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between cycles")
    parser.add_argument("--execute", action="store_true", help="Enable auto-execution")

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = AutonomousTradingOrchestrator()

    # Override auto-execute if flag provided
    if args.execute:
        orchestrator.config["auto_execute"] = True

    if args.continuous:
        orchestrator.run_continuous(args.interval)
    else:
        result = orchestrator.run_trading_cycle()
        print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
