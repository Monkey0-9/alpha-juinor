
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from enum import Enum

class SellReason(Enum):
    NONE = "NONE"
    HARD_STOP = "HARD_STOP"
    TRAILING_STOP = "TRAILING_STOP"
    PREDICTIVE_EXIT = "PREDICTIVE_EXIT"
    TIME_DECAY = "TIME_DECAY"

class SmartSellEngine:
    """
    Intelligent Exit Management.
    """
    def __init__(self, atr_period=14, hard_stop_atr_mult=2.0, trail_stop_atr_mult=3.0):
        self.atr_period = atr_period
        self.hard_stop_mult = hard_stop_atr_mult
        self.trail_stop_mult = trail_stop_atr_mult

        # State: {symbol: {'entry_price': x, 'highest_price': y, 'entry_time': t}}
        self.state = {}

    def register_entry(self, symbol: str, price: float, timestamp: pd.Timestamp):
        self.state[symbol] = {
            'entry_price': price,
            'highest_price': price,
            'entry_time': timestamp
        }

    def check_exit(self, symbol: str, current_price: float, history: pd.DataFrame) -> Tuple[SellReason, str]:
        """
        Check if an exit triggers.
        """
        if symbol not in self.state:
            return SellReason.NONE, ""

        pos = self.state[symbol]
        entry = pos['entry_price']

        # Update Highest
        if current_price > pos['highest_price']:
            pos['highest_price'] = current_price

        # Calc ATR
        if history.empty or len(history) < self.atr_period + 1:
            atr = current_price * 0.02 # Fallback 2%
        else:
            # Simple TR approximation
            high_low = history['High'] - history['Low']
            atr = high_low.rolling(self.atr_period).mean().iloc[-1]

        # 1. Hard Stop
        stop_price = entry - (atr * self.hard_stop_mult)
        if current_price < stop_price:
            return SellReason.HARD_STOP, f"Price {current_price:.2f} < Hard Stop {stop_price:.2f}"

        # 2. Trailing Stop
        trail_price = pos['highest_price'] - (atr * self.trail_stop_mult)
        if current_price < trail_price:
            return SellReason.TRAILING_STOP, f"Price {current_price:.2f} < Trail Stop {trail_price:.2f}"

        # 3. Predictive (Stub)
        # if predictive_model.predict_crash() > 0.8: return ...

        return SellReason.NONE, ""
