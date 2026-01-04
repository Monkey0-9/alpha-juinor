from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from backtest.execution import Order, OrderType

logger = logging.getLogger(__name__)

@dataclass
class AllocationResult:
    target_weights: Dict[str, float]
    orders: List[Order]
    meta: Dict[str, Any]

class InstitutionalAllocator:
    """
    Central Trade Decision Engine.
    Converts: Alpha Signals -> Conviction -> Target Exposure -> Orders.
    Enforces: Risk Budget, Volatility Targeting, Regime Scaling, Liquidity Limits.
    """
    
    def __init__(self, risk_manager, max_leverage: float = 1.0):
        self.risk_manager = risk_manager
        self.max_leverage = max_leverage
        
    def _calculate_kelly_weights(self, signals: Dict[str, float], returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Calculates weights using the Kelly Criterion: weight = mu / sigma^2.
        """
        kelly_weights = {}
        for tk, sig in signals.items():
            if tk in returns and not returns[tk].empty:
                # Annualized mean and variance
                mu = returns[tk].mean() * 252
                var = returns[tk].var() * 252
                if var > 1e-6:
                    # Signal-adjusted Kelly
                    f_star = (mu / var) * (sig - 0.5) * 2.0
                    kelly_weights[tk] = f_star
                else:
                    kelly_weights[tk] = 0.0
            else:
                kelly_weights[tk] = 0.0
        return kelly_weights

    def _calculate_inverse_vol_weights(self, returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Inverse Volatility weighting (Simple Risk Parity).
        weight_i = (1/vol_i) / sum(1/vol_j)
        """
        vols = {}
        for tk, ret in returns.items():
            if not ret.empty:
                vol = ret.std() * np.sqrt(252)
                vols[tk] = max(vol, 0.01) # Floor vol at 1%
        
        if not vols:
            return {}
            
        inv_vols = {k: 1.0/v for k, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        return {k: v / (total_inv_vol + 1e-9) for k, v in inv_vols.items()}

    def allocate(
        self, 
        signals: Dict[str, float], 
        prices: Dict[str, pd.Series], # History for risk 
        volumes: Dict[str, pd.Series], # History for liquidity
        current_portfolio, 
        timestamp: pd.Timestamp,
        method: str = "signal"
    ) -> AllocationResult:
        """
        Main allocation routine.
        1. Normalize Signals -> Raw Weights
        2. Apply Risk Constraints -> Risk-Adjusted Weights
        3. Diff against Current Portfolio -> Position Deltas
        4. Generate Orders
        """
        # INSTITUTIONAL: Explicit fill_method + immediate cleaning
        returns_hist = {}
        for tk, px in prices.items():
            r = px.pct_change(fill_method=None)
            r = r.replace([np.inf, -np.inf], np.nan).dropna()
            returns_hist[tk] = pd.to_numeric(r, errors='coerce').astype(float)
        
        if method == "kelly":
            raw_weights = self._calculate_kelly_weights(signals, returns_hist)
        elif method == "risk_parity":
            rp_weights = self._calculate_inverse_vol_weights(returns_hist)
            raw_weights = {tk: rp_weights.get(tk, 0.0) * (sig - 0.5) * 2.0 for tk, sig in signals.items()}
        else:
            raw_weights = {tk: (sig - 0.5) * 2.0 for tk, sig in signals.items()}
            
        # Total signal score for normalization
        total_score = sum(abs(w) for w in raw_weights.values())
        
        norm_weights = {}
        if total_score > 0.001:
            for tk, w in raw_weights.items():
                # Normalize and scale by max leverage
                norm_weights[tk] = (w / total_score) * self.max_leverage
        else:
            # 100% Cash
            pass

        # 2. Apply Per-Asset Risk Constraints
        final_target_weights = {}
        risk_meta = {}
        equity = getattr(current_portfolio, "total_equity", 100_000.0)
        
        for tk, raw_w in norm_weights.items():
            intent_dollars = raw_w * equity
            conv_s = pd.Series([intent_dollars], index=[timestamp]) 
            px_hist = prices.get(tk, pd.Series())
            vol_hist = volumes.get(tk, pd.Series())
            
            if self.risk_manager:
                res = self.risk_manager.enforce_limits(conv_s, px_hist, vol_hist)
                adj_dollars = res.adjusted_conviction.iloc[-1]
                final_target_weights[tk] = adj_dollars / (equity + 1e-9)
                risk_meta[tk] = res.violations
            else:
                final_target_weights[tk] = raw_w

        # 3. Generate Orders (Diff Logic)
        orders = []
        current_positions = getattr(current_portfolio, "positions", {})
        all_tickers = set(final_target_weights.keys()) | set(current_positions.keys())
        
        for tk in all_tickers:
            target_w = final_target_weights.get(tk, 0.0)
            target_qty = 0
            current_price = 0.0
            
            if tk in prices and not prices[tk].empty:
                current_price = prices[tk].iloc[-1]
                
            if current_price > 0:
                target_val = target_w * equity
                target_qty = int(target_val / current_price)
                
            current_qty = current_positions.get(tk, 0)
            delta_qty = target_qty - current_qty
            
            if delta_qty != 0:
                orders.append(Order(tk, delta_qty, OrderType.MARKET, timestamp))

        return AllocationResult(
            target_weights=final_target_weights,
            orders=orders,
            meta=risk_meta
        )
