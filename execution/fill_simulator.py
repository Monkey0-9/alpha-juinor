"""
execution/fill_simulator.py

Section D: Execution Realism.
Simulates order slicing, latency, market impact (Almgren-Chriss style), and slippage.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("EXEC_SIM")

@dataclass
class Fill:
    symbol: str
    fill_id: str
    price: float
    qty: float
    timestamp: str
    commission: float
    slippage: float
    impact_cost: float
    market_price_snapshot: float

class FillSimulator:
    def __init__(self,
                 base_latency_ms: float = 100.0,
                 volatility_day: float = 0.02,
                 impact_lambda: float = 0.5):
        self.base_latency = base_latency_ms
        self.volatility = volatility_day
        self.impact_lambda = impact_lambda # Coeff for impact model

    def simulate_order(self,
                       order_id: str,
                       symbol: str,
                       side: str,
                       qty: float,
                       market_price: float,
                       adv: float,
                       urgency: str = "MEDIUM") -> List[Fill]:

        fills = []
        remaining = qty

        # Slicing strategy: Divide into N slices
        # N depends on qty relative to ADV
        pct_adv = (qty / adv) if adv > 0 else 0.01

        if pct_adv < 0.01: pieces = 1
        elif pct_adv < 0.05: pieces = 5
        else: pieces = 10

        slice_size = qty / pieces

        logger.info(f"Simulating {symbol} {side} {qty} in {pieces} slices (ADV%: {pct_adv:.4f})")

        current_market = market_price

        for i in range(pieces):
            # 1. Random Walk / Latency slippage
            # Volatility drift over execution time (e.g. 1 min per slice)
            drift = np.random.normal(0, self.volatility / np.sqrt(390*60)) # 1 second vol approx
            current_market *= (1 + drift)

            # 2. Market Impact (Permanent + Transient)
            # Impact = lambda * sigma * sqrt(size / adv) roughly
            # Simplified: impact bps = lambda * sqrt(pct_adv_slice)
            slice_pct = slice_size / adv if adv > 0 else 0.0
            impact_bps = self.impact_lambda * np.sqrt(slice_pct)

            direction = 1 if side == "BUY" else -1
            impact_price_move = current_market * (impact_bps / 10000.0) * direction

            exec_price = current_market + impact_price_move

            # 3. Slippage
            slippage = abs(exec_price - market_price)

            # 4. Fill
            fill = Fill(
                symbol=symbol,
                fill_id=f"{order_id}_{i}",
                price=exec_price,
                qty=slice_size,
                timestamp=datetime.utcnow().isoformat(),
                commission=max(1.0, slice_size * 0.005 * exec_price), # Example 5bps
                slippage=slippage,
                impact_cost=abs(exec_price - current_market),
                market_price_snapshot=current_market
            )
            fills.append(fill)

            # Permanent impact update for next slice
            current_market += (impact_price_move * 0.5)

        return fills
