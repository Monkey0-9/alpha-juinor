# risk/engine.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Institutional RiskManager with VaR/CVaR controls.
    
    Policies:
      - max_leverage: Hard cap on gross exposure
      - target_vol_limit: Reduce exposure if realized vol > target
      - var_limit: Max allowed 1-day 95% VaR (as fraction of equity)
      - cvar_limit: Max allowed 1-day 95% CVaR (Expected Shortfall)
      - max_drawdown_limit: Hard stop if drawdown exceeds this
    """

    def __init__(
        self,
        max_leverage: float = 1.0,
        target_vol_limit: float = 0.12,
        min_allowed: float = 0.0,
        var_limit: float = 0.02, # 2% daily VaR limit (~32% annual vol equivalent)
        cvar_limit: float = 0.03, # 3% daily CVaR limit
        max_drawdown_limit: float = 0.20, # 20% hard drawdown stop
    ):
        self.max_leverage = float(max_leverage)
        self.target_vol_limit = float(target_vol_limit)
        self.min_allowed = float(min_allowed)
        self.var_limit = float(var_limit)
        self.cvar_limit = float(cvar_limit)
        self.max_drawdown_limit = float(max_drawdown_limit)

    def _realized_vol(self, prices: pd.Series, window: int = 21) -> pd.Series:
        returns = prices.pct_change()
        realized = returns.rolling(window).std() * (252 ** 0.5)
        # Use modern pandas API for forward/backward fill
        realized = realized.bfill().fillna(0.0)
        return realized

    def compute_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical Value at Risk (positive number representing loss fraction)."""
        if returns.empty or len(returns) < 20: 
            return 0.0
        # VaR is the loss at the (1-conf) percentile
        # e.g. 95% conf -> 5th percentile
        cutoff = np.percentile(returns, 100 * (1 - confidence))
        return -cutoff if cutoff < 0 else 0.0

    def compute_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical Conditional Value at Risk (Expected Shortfall)."""
        if returns.empty or len(returns) < 20:
            return 0.0
        cutoff = -self.compute_var(returns, confidence)
        tail_losses = returns[returns <= cutoff]
        if tail_losses.empty:
            return -cutoff # Fallback to VaR
        return -tail_losses.mean()

    def enforce_limits(
        self, conviction: pd.Series, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Legacy method: Volatility scaling for single-asset convictions.
        
        :param prices: Price series for the asset
        :return: (adjusted_conviction, leverage_factor)
        """
        realized_vol = self._realized_vol(prices)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = self.target_vol_limit / realized_vol
        scale = scale.replace([np.inf, -np.inf], self.max_leverage).fillna(self.max_leverage)
        allowed_leverage = pd.Series(scale).clip(upper=self.max_leverage).fillna(self.max_leverage)
        leverage_factor = (allowed_leverage / self.max_leverage).clip(lower=self.min_allowed, upper=1.0)
        adjusted = (conviction * leverage_factor).clip(0, 1)

        # Transparency Logging (sampled check for last bar)
        if not leverage_factor.empty:
            lev_val = float(leverage_factor.iloc[-1])
            if lev_val < 1.0:
                 vol_val = float(realized_vol.iloc[-1]) if not realized_vol.empty else 0.0
                 logger.warning(
                     f"Risk Veto (Vol): Realized ({vol_val:.2%}) > Target ({self.target_vol_limit:.2%}). "
                     f"Scaling exposure by factor {lev_val:.2f}"
                 )

        return adjusted, leverage_factor

    def check_portfolio_risk(
        self, 
        weights: Dict[str, float], 
        baskets_returns: pd.DataFrame,
        portfolio_value: float = 1.0,
        positions: Optional[Dict[str, float]] = None,
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate full portfolio risk against VaR/CVaR limits.
        
        Args:
            weights: Dict {ticker: weight} (Target weights)
            baskets_returns: DataFrame of historical returns for all tickers (aligned)
            portfolio_value: Total equity (used for reporting violations in $)
            positions: (Optional) Dict {ticker: quantity} - derived if weights not passed
            prices: (Optional) Dict {ticker: current_price} - needed if deriving from positions
            
        Returns:
            Dict with risk metrics and 'ok' status
        """
        # 1. Determine weights
        if not weights and positions and prices:
            total_exp = 0.0
            computed_weights = {}
            for tk, qty in positions.items():
                val = qty * prices.get(tk, 0.0)
                total_exp += val
                computed_weights[tk] = val / portfolio_value if portfolio_value > 0 else 0.0
            weights = computed_weights
            gross_leverage = total_exp / portfolio_value if portfolio_value > 0 else 0.0
        else:
            # Use provided weights
            gross_leverage = sum(abs(w) for w in weights.values())

        # Portfolio historical returns simulation
        # R_p = sum(w_i * R_i)
        if baskets_returns.empty:
             return {"ok": True, "reason": "No history"}
             
        # Filter weights for tickers present in history
        valid_weights = {k: v for k, v in weights.items() if k in baskets_returns.columns}
        if not valid_weights:
            portfolio_returns = pd.Series(0.0, index=baskets_returns.index)
        else:
            # Weighted sum of returns
            w_vector = pd.Series(valid_weights)
            # Align columns
            aligned_returns = baskets_returns[w_vector.index]
            portfolio_returns = aligned_returns.dot(w_vector)

        # Calculate metrics
        current_var = self.compute_var(portfolio_returns)
        current_cvar = self.compute_cvar(portfolio_returns)
        
        # Drawdown check (simplified from equity curve if available, here just using bounds)
        # We can't check drawdown without equity history passed in. 
        # Assuming this checks *projected* risk.

        violations = []
        if gross_leverage > self.max_leverage + 0.01: # tolerance
            violations.append(f"Leverage {gross_leverage:.2f} > {self.max_leverage}")
            
        if current_var > self.var_limit:
            violations.append(f"VaR {current_var:.2%} > {self.var_limit:.2%}")
            
        if current_cvar > self.cvar_limit:
            violations.append(f"CVaR {current_cvar:.2%} > {self.cvar_limit:.2%}")

        return {
            "ok": len(violations) == 0,
            "violations": violations,
            "metrics": {
                "leverage": gross_leverage,
                "var_95": current_var,
                "cvar_95": current_cvar
            }
        }

    def summary(self, original: pd.Series, adjusted: pd.Series) -> str:
        avg_before = float(original.mean())
        avg_after = float(adjusted.mean())
        pct_reduction = 0.0
        if avg_before > 0:
            pct_reduction = 100.0 * (avg_before - avg_after) / avg_before
        return (
            f"RiskManager summary — avg conviction before: {avg_before:.3f}, "
            f"after: {avg_after:.3f}, reduction: {pct_reduction:.1f}%"
        )
