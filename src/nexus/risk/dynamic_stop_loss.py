import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

# risk/dynamic_stop_loss.py

logger = logging.getLogger("STOP_LOSS")

class DynamicStopLoss:
    """
    Volatility-adjusted trailing stop-loss protection.
    Triggers liquidation if position drops below N standard deviations.
    """

    def __init__(self, std_multiplier: float = 2.5, min_stop_pct: float = 0.02):
        self.std_multiplier = std_multiplier
        self.min_stop_pct = min_stop_pct
        self.positions_entry_price: Dict[str, float] = {}

    def update_entry_prices(self, current_positions: Dict[str, float], current_prices: Dict[str, float]):
        """Maintain record of where we entered positions to calculate drawdown."""
        for symbol in current_positions:
            if symbol not in self.positions_entry_price:
                self.positions_entry_price[symbol] = current_prices.get(symbol, 0.0)

        # Clean up closed positions
        self.positions_entry_price = {s: p for s, p in self.positions_entry_price.items() if s in current_positions}

    def check_stops(self, current_prices: Dict[str, float], vol_data: Dict[str, float]) -> List[str]:
        """
        Check if any position has hit its stop-loss.
        Returns list of symbols to liquidate.
        """
        liquidate = []

        for symbol, entry_price in self.positions_entry_price.items():
            curr_price = current_prices.get(symbol, 0.0)
            if entry_price == 0: continue

            # Drawdown from entry
            drawdown = (curr_price - entry_price) / entry_price

            # Volatility-adjusted stop
            # e.g. if daily vol is 2%, 2.5x multiplier = 5% stop
            daily_vol = vol_data.get(symbol, 0.02)
            dynamic_threshold = -max(self.min_stop_pct, daily_vol * self.std_multiplier)

            if drawdown < dynamic_threshold:
                logger.critical(f"[STOP_LOSS] TRIGGERED: symbol={symbol} Drawdown={drawdown:.2%} Limit={dynamic_threshold:.2%}")
                liquidate.append(symbol)

        return liquidate
