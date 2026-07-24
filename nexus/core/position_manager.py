import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from nexus.utils.config import Config

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Institutional Position Management & Dynamic Risk Control.
    Implements Volatility-Regime Stop Losses, Exponential Time-Decay Trailing Stops,
    and GARCH/ATR Volatility Position Scaling.
    """
    def __init__(self):
        # Track the highest high since entry for each position
        self._watermarks: Dict[str, float] = {}
        # Track entry timestamp / hold duration cycles
        self._hold_cycles: Dict[str, int] = {}

    def evaluate_exit(
        self,
        symbol: str,
        current_price: float,
        avg_entry_price: float,
        pnl_pct: float,
        history: pd.DataFrame,
        current_regime: str = "SIDEWAYS",
        hold_hours: float = 0.0
    ) -> bool:
        """
        Evaluate if a position should be closed based on dynamic risk rules.
        """
        self._hold_cycles[symbol] = self._hold_cycles.get(symbol, 0) + 1

        # 1. Volatility-Regime Dynamic Stop Loss Threshold
        regime_stop = self.get_regime_stop_loss(current_regime, Config.STOP_LOSS_THRESHOLD)
        if pnl_pct <= regime_stop:
            logger.info(f"Closing {symbol}: Dynamic Regime Stop Loss Hit ({pnl_pct:.2%}, Regime: {current_regime})")
            return True

        # 2. Hard Take Profit
        if pnl_pct >= Config.TAKE_PROFIT_THRESHOLD:
            logger.info(f"Closing {symbol}: Hard Take Profit Reached ({pnl_pct:.2%})")
            return True

        if current_price <= 0 or avg_entry_price <= 0:
            return False

        # Update High Watermark
        if symbol not in self._watermarks or current_price > self._watermarks[symbol]:
            self._watermarks[symbol] = current_price

        highest_price = self._watermarks[symbol]
        peak_profit_pct = (highest_price - avg_entry_price) / avg_entry_price

        # 3. Exponential Time-Decay Trailing Stop
        # As time elapses, tighten the allowed profit drawdown to prevent stagnant capital tie-up
        time_decay_multiplier = float(np.exp(-0.02 * hold_hours))
        effective_lock_trigger = Config.TRAILING_PROFIT_LOCK * time_decay_multiplier

        if peak_profit_pct >= effective_lock_trigger:
            current_profit_pct = (current_price - avg_entry_price) / avg_entry_price
            allowed_retention = 0.60 * time_decay_multiplier
            if current_profit_pct < (peak_profit_pct * allowed_retention):
                logger.info(
                    f"Closing {symbol}: Time-Decay Trailing Lock triggered. "
                    f"Peak: {peak_profit_pct:.2%}, Current: {current_profit_pct:.2%}, Hold: {hold_hours:.1f}h."
                )
                return True

        # 4. Breakeven Stop
        if peak_profit_pct >= Config.BREAKEVEN_TRIGGER:
            if current_price < (avg_entry_price * 1.002):
                logger.info(f"Closing {symbol}: Breakeven Stop hit. Locking in flat/gain.")
                return True

        # 5. ATR-Based Volatility Trailing Stop
        if not history.empty and len(history) >= 14 and all(col in history.columns for col in ["high", "low", "close"]):
            try:
                high = history["high"]
                low = history["low"]
                close = history["close"]

                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])

                if atr > 0:
                    stop_level = highest_price - (atr * Config.ATR_STOP_MULTIPLIER)
                    if current_price < stop_level:
                        logger.info(f"Closing {symbol}: ATR Trailing Stop hit. Current: {current_price:.2f}, Stop: {stop_level:.2f}")
                        return True
            except Exception as e:
                logger.debug(f"ATR calculation failed for {symbol}: {e}")

        return False

    @staticmethod
    def get_regime_stop_loss(regime: str, base_stop: float) -> float:
        """Scale base stop-loss according to current market regime volatility."""
        if regime == "TURBULENT":
            return base_stop * 0.70  # Tighter stop (-3.5% instead of -5%)
        elif regime == "BEAR":
            return base_stop * 0.85
        elif regime == "BULL":
            return base_stop * 1.20  # Wider stop to ride trend
        return base_stop

    def calculate_volatility_scaled_size(
        self,
        portfolio_value: float,
        asset_volatility: float,
        target_volatility: float = 0.15,
        max_position_pct: float = 0.08
    ) -> float:
        """
        Volatility Scaling: Size positions inversely proportional to annualized volatility.
        Position Size = (Portfolio Value * Target Volatility) / (Asset Volatility * N_Assets)
        """
        if asset_volatility <= 1e-4 or portfolio_value <= 0:
            return portfolio_value * max_position_pct

        vol_weight = target_volatility / asset_volatility
        scaled_pct = min(max_position_pct, max_position_pct * vol_weight)
        return float(portfolio_value * scaled_pct)

    def reset_watermark(self, symbol: str):
        """Reset watermark and hold cycles when position is closed."""
        self._watermarks.pop(symbol, None)
        self._hold_cycles.pop(symbol, None)
