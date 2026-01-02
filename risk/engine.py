# risk/engine.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple


class RiskManager:
    """
    RiskManager enforces simple, production-minded policy:
      - max_leverage: hard cap (e.g., 1.0 = no leverage, 2.0 = 2x)
      - target_vol_limit: if realized vol > target, reduce exposure proportionally
      - min_allowed: minimum fraction allowed (0 disables)
    """

    def __init__(
        self,
        max_leverage: float = 1.0,
        target_vol_limit: float = 0.12,
        min_allowed: float = 0.0,
    ):
        self.max_leverage = float(max_leverage)
        self.target_vol_limit = float(target_vol_limit)
        self.min_allowed = float(min_allowed)

    def _realized_vol(self, prices: pd.Series, window: int = 21) -> pd.Series:
        returns = prices.pct_change()
        realized = returns.rolling(window).std() * (252 ** 0.5)
        return realized.fillna(method="bfill").fillna(0.0)

    def enforce_limits(
        self, conviction: pd.Series, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Given a conviction series (0..1) and price series, return:
          - adjusted_conviction (0..1) after applying risk limits
          - allowed_leverage_series (>=0) used to scale conviction
        Mechanism:
          allowed_leverage = min(max_leverage, target_vol_limit / realized_vol)
          adjusted_conviction = conviction * (allowed_leverage / max_leverage)
        """
        realized_vol = self._realized_vol(prices)
        # Avoid divide by zero: when realized_vol == 0, allow max_leverage
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = self.target_vol_limit / realized_vol
        scale = scale.replace([np.inf, -np.inf], self.max_leverage).fillna(self.max_leverage)
        allowed_leverage = pd.Series(scale).clip(upper=self.max_leverage).fillna(self.max_leverage)
        # normalized leverage factor between 0..1 relative to max_leverage
        leverage_factor = (allowed_leverage / self.max_leverage).clip(lower=self.min_allowed, upper=1.0)
        adjusted = (conviction * leverage_factor).clip(0, 1)

        return adjusted, leverage_factor

    def summary(self, original: pd.Series, adjusted: pd.Series) -> str:
        """
        Quick textual summary for debugging.
        """
        avg_before = float(original.mean())
        avg_after = float(adjusted.mean())
        pct_reduction = 0.0
        if avg_before > 0:
            pct_reduction = 100.0 * (avg_before - avg_after) / avg_before
        return (
            f"RiskManager summary — avg conviction before: {avg_before:.3f}, "
            f"after: {avg_after:.3f}, reduction: {pct_reduction:.1f}%"
        )
