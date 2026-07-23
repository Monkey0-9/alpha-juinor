import logging
from typing import Dict
import pandas as pd
from nexus.utils.config import Config

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Advanced Position Management for Paper Trading.
    Implements ATR-based trailing stops, trailing profit locks, 
    and breakeven stops for maximum upside capture.
    """
    def __init__(self):
        # Track the highest high since entry for each position
        # symbol -> highest_price
        self._watermarks: Dict[str, float] = {}
        
    def evaluate_exit(self, symbol: str, current_price: float, avg_entry_price: float, pnl_pct: float, history: pd.DataFrame) -> bool:
        """
        Evaluate if a position should be closed based on dynamic risk rules.
        Returns True if the position should be closed.
        """
        if current_price <= 0 or avg_entry_price <= 0:
            return False
            
        # Update high watermark
        if symbol not in self._watermarks or current_price > self._watermarks[symbol]:
            self._watermarks[symbol] = current_price
            
        highest_price = self._watermarks[symbol]
        
        # Rule 1: Fixed Stop Loss (Disaster Prevention)
        if pnl_pct <= Config.STOP_LOSS_THRESHOLD:
            logger.info(f"Closing {symbol}: Fixed Stop Loss Hit ({pnl_pct:.2%})")
            return True
            
        # Rule 2: Trailing Profit Lock
        # If we are up significantly, don't let it turn into a big loss.
        if (highest_price - avg_entry_price) / avg_entry_price >= Config.TRAILING_PROFIT_LOCK:
            # If we've dropped more than half our peak profit, get out.
            peak_profit_pct = (highest_price - avg_entry_price) / avg_entry_price
            current_profit_pct = (current_price - avg_entry_price) / avg_entry_price
            
            if current_profit_pct < (peak_profit_pct * 0.5):
                logger.info(f"Closing {symbol}: Trailing Profit Lock triggered. Peak profit was {peak_profit_pct:.2%}, now {current_profit_pct:.2%}.")
                return True
                
        # Rule 3: Breakeven Stop
        # If we reached the breakeven trigger, move stop to slightly above breakeven
        if (highest_price - avg_entry_price) / avg_entry_price >= Config.BREAKEVEN_TRIGGER:
            if current_price < (avg_entry_price * 1.002): # 0.2% above entry
                logger.info(f"Closing {symbol}: Breakeven Stop hit. Locking in small gain/flat.")
                return True
                
        # Rule 4: ATR-Based Trailing Stop
        # Requires history to calculate ATR
        if not history.empty and "high" in history.columns and "low" in history.columns and "close" in history.columns and len(history) >= 14:
            try:
                high = history["high"]
                low = history["low"]
                close = history["close"]
                
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                
                if atr > 0:
                    stop_level = highest_price - (atr * Config.ATR_STOP_MULTIPLIER)
                    if current_price < stop_level:
                        logger.info(f"Closing {symbol}: ATR Trailing Stop hit. Current: {current_price:.2f}, Stop: {stop_level:.2f}")
                        return True
            except Exception as e:
                logger.debug(f"ATR calculation failed for {symbol}: {e}")
                
        # Rule 5: Hard Take Profit
        if pnl_pct >= Config.TAKE_PROFIT_THRESHOLD:
             logger.info(f"Closing {symbol}: Hard Take Profit Reached ({pnl_pct:.2%})")
             return True

        return False
        
    def reset_watermark(self, symbol: str):
        """Reset watermark when position is closed."""
        self._watermarks.pop(symbol, None)
