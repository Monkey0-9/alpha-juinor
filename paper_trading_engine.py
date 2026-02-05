#!/usr/bin/env python3
"""
MULTI-STRATEGY PAPER TRADING ENGINE (S-CLASS)
=============================================

The Production Engine for the "Strategy of Strategies".
Integrates:
1. StrategyManager (MR + Trend + Sentiment)
2. MetaController (Regime-based Risk Allocation)
3. S-Class Execution (TWAP Order Slicing)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
import sys
import logging

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_factory.manager import StrategyManager
from allocator.meta_controller import MetaController

# Logger setup (configured by entry point)
logger = logging.getLogger("PAPER_TRADING")

# Configuration
MAX_POSITION = 0.25  # 25% per asset cap
REBALANCE_THRESHOLD = 0.0 # Force trades for testing

@dataclass
class Trade:
    timestamp: str
    symbol: str
    action: str
    quantity: float
    price: float
    rationale: str

class ExecutionAlgo:
    """
    S-Class Execution: Slicing orders to minimize impact.
    """
    @staticmethod
    def twap(symbol: str, quantity: float, duration_mins: int = 15, slices: int = 4) -> List[Trade]:
        """
        Slice order into smaller child orders (TWAP).
        """
        child_orders = []
        slice_qty = quantity / slices
        action = "BUY" if quantity > 0 else "SELL"

        for i in range(slices):
            # Simulate time delay (logically)
            timestamp = (datetime.now() + timedelta(minutes=i * (duration_mins/slices))).isoformat()

            # In a real system, we'd wait. Here we generate the schedule.
            child = Trade(
                timestamp=timestamp,
                symbol=symbol,
                action=action,
                quantity=abs(slice_qty),
                price=0.0, # Filled at market
                rationale=f"TWAP Slice {i+1}/{slices}"
            )
            child_orders.append(child)

        return child_orders

class CircuitBreaker:
    """
    S-Class Risk Control: Hard stop for drawdown or failures.
    """
    def __init__(self, max_drawdown=0.03):
        self.max_drawdown = max_drawdown
        self.triggered = False
        self.peak_equity = 0.0

    def check(self, current_equity: float) -> bool:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0

        if drawdown > self.max_drawdown:
            self.triggered = True
            logger.critical(f"CIRCUIT BREAKER TRIGGERED! Drawdown: {drawdown:.2%}")
            return False # Halt trading

        return True # Continue

# Imports
from risk.gateway import RiskGateway
from execution.sor import SmartOrderRouter

class MultiStrategyEngine:
    """
    Orchestrates the Multi-Strategy Platform in live/paper mode.
    """

    def __init__(self):
        self.manager = StrategyManager()
        self.controller = MetaController()

        self.breaker = CircuitBreaker()
        self.gateway = RiskGateway() # Pre-Trade Risk

        self.positions: Dict[str, float] = {} # Symbol -> Weight
        self.equity = 100000.0 # Starting Equity
        self.trade_log: List[Trade] = []

        logger.info("MultiStrategyEngine initialized (S-Class Execution Enabled)")
        logger.info(f"  Strategies: {[s.name for s in self.manager.strategies]}")

    def save_state(self, filename="system_state.json"):
        """Persist system state to disk."""
        state = {
            "positions": self.positions,
            "equity": self.equity,
            "peak_equity": self.breaker.peak_equity,
            "breaker_triggered": self.breaker.triggered,
            "timestamp": datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"State saved to {filename}")

    def load_state(self, filename="system_state.json"):
        """Load system state from disk."""
        if not os.path.exists(filename):
            logger.info("No state file found. Starting fresh.")
            return

        try:
            with open(filename, 'r') as f:
                state = json.load(f)

            self.positions = state.get("positions", {})
            self.equity = state.get("equity", 100000.0)
            self.breaker.peak_equity = state.get("peak_equity", self.equity)
            self.breaker.triggered = state.get("breaker_triggered", False)

            logger.info(f"State loaded from {filename}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def generate_target_weight(
        self,
        symbol: str,
        prices: pd.Series,
        benchmark_prices: pd.Series
    ) -> float:
        """
        Generate target weight for an asset using the full stack.
        """
        # A. Detect Regime (Global/Benchmark)
        regime_data = self.controller.regime_detector.detect(benchmark_prices)

        # B. Generate Strategy Signals (Regime Aware)
        regime_dict = {
            'regime': regime_data.regime,
            'risk_multiplier': regime_data.risk_multiplier
        }

        strat_signals = self.manager.generate_all_signals(symbol, prices, regime_dict)

        # C. Allocate (Meta Controller)
        portfolio_sigs, _ = self.controller.generate_portfolio_signals(strat_signals, benchmark_prices)
        target_weight = portfolio_sigs.get(symbol, 0.0)

        # D. Cap Weight
        target_weight = np.clip(target_weight, -MAX_POSITION, MAX_POSITION)

        # Log details
        log_msg = f"  {symbol} [{regime_data.regime}]: "
        for name, sig in strat_signals.items():
            try:
                # Handle signals efficiently
                log_msg += f"{name.split('_')[0]}={sig.strength:.2f} "
            except: pass
        log_msg += f"-> NET={target_weight:.1%}"
        logger.info(log_msg)

        return target_weight

    def run_daily_cycle(
        self,
        market_data: Dict[str, pd.Series], # Symbol -> History
        current_prices: Dict[str, float],
        portfolio_value: float,
        benchmark_symbol: str = "SPY"
    ) -> Dict:
        """Run daily trading cycle."""
        logger.info("=" * 60)
        logger.info(f"DAILY CYCLE: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)

        # 0. Update Equity & Check Circuit Breaker
        self.equity = portfolio_value
        if not self.breaker.check(self.equity):
             logger.warning("TRADING HALTED BY CIRCUIT BREAKER.")
             return {"status": "HALTED"}

        if self.breaker.triggered:
             logger.warning("SYSTEM IN HALTED STATE.")
             return {"status": "HALTED"}

        # Ensure benchmark data exists
        if benchmark_symbol not in market_data:
            logger.error(f"Benchmark {benchmark_symbol} missing from data!")
            return {}

        benchmark_prices = market_data[benchmark_symbol]

        target_weights = {}

        # 1. Generate Targets
        for symbol, prices in market_data.items():
            target = self.generate_target_weight(symbol, prices, benchmark_prices)
            target_weights[symbol] = target

        # 2. Rebalance with Elite Execution (Gateway -> SOR)
        trades = []
        for symbol, target in target_weights.items():
            current = self.positions.get(symbol, 0.0)

            if abs(target - current) > REBALANCE_THRESHOLD:
                price = current_prices.get(symbol, 0)
                if price <= 0: continue

                diff = target - current
                dollar_amt = diff * portfolio_value
                qty = dollar_amt / price

                # A. Pre-Trade Risk Check
                if not self.gateway.check_order(symbol, qty, price):
                    logger.warning(f"  BLOCKED: Risk Rejection for {symbol}")
                    continue

                # B. Elite Execution (SOR)
                # S-Class: Use SOR Routing
                child_orders = SmartOrderRouter.slice_and_route(symbol, qty, price)

                for child in child_orders:
                    # child.price is already set by SOR (potentially improved)
                    trades.append(child)
                    logger.info(f"    EXECUTE SOR: {child.action} {child.quantity:.2f} {symbol} @ ${child.price:.2f} ({child.rationale})")

                self.positions[symbol] = target

        # 3. Save State
        self.save_state()

        return {
            "positions": self.positions.copy(),
            "trades": len(trades),
            "equity": self.equity
        }

def demo_paper_trading():
    print("DEMO RUN (S-CLASS)")
    import yfinance as yf

    symbols = ["SPY", "QQQ", "GLD", "TLT"]
    data = yf.download(symbols, period="1y", progress=False)

    market_data = {}
    current_prices = {}

    for sym in symbols:
        if isinstance(data.columns, pd.MultiIndex):
             if ('Close', sym) in data.columns:
                 s = data[('Close', sym)].dropna()
                 market_data[sym] = s
                 current_prices[sym] = float(s.iloc[-1])

    engine = MultiStrategyEngine()
    engine.run_daily_cycle(market_data, current_prices, 100000)

    print("\nPositions:")
    for sym, w in engine.positions.items():
        print(f"  {sym}: {w:.1%}")

if __name__ == "__main__":
    demo_paper_trading()
