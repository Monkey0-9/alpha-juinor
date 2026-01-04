
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
        
    def allocate(
        self, 
        signals: Dict[str, float], 
        prices: Dict[str, pd.Series], # History for risk 
        volumes: Dict[str, pd.Series], # History for liquidity
        current_portfolio, 
        timestamp: pd.Timestamp
    ) -> AllocationResult:
        """
        Main allocation routine.
        1. Normalize Signals -> Raw Weights
        2. Apply Risk Constraints (Vol, Dropdown, Regime) -> Risk-Adjusted Weights
        3. Diff against Current Portfolio -> Position Deltas
        4. Generate Orders
        """
        
        # 1. Normalize Signals (0..1) to Raw Weights (-1..1)
        # Assuming Long-Only for now unless signal < 0.5?
        # Let's assume Signals are Probabilities 0..1 (0.5 Neutral)
        # Map 0.5 -> 0, 1.0 -> 1.0, 0.0 -> -1.0
        
        raw_weights = {}
        total_signal_score = 0.0
        
        for tk, sig in signals.items():
            # Linear mapping: (Sig - 0.5) * 2 
            # Sig 1.0 -> 1.0 (Long)
            # Sig 0.0 -> -1.0 (Short)
            # Sig 0.5 -> 0.0 (Cash)
            w = (sig - 0.5) * 2.0
            raw_weights[tk] = w
            total_signal_score += abs(w)
            
        # Normalize to Max Leverage if signals are strong
        # e.g. If 2 assets have strong buy (1.0 each), we split capital?
        # Or do we implicitly leverage? 
        # Institutional: Allocation usually sums to Exposure Target.
        
        # Normalize to sum(abs(weights)) == 1.0 (Fully Invested) if possible
        # Then scale by conviction.
        
        norm_weights = {}
        if total_signal_score > 0.001:
            for tk, w in raw_weights.items():
                norm_weights[tk] = w / total_signal_score # Relative weight
                
            # Scale by Risk/Conviction Capability
            # This is where we could apply "Gross Exposure" targets.
            # For now, let's target 1.0 Gross * Portfolio Leverage
        else:
             # No signals -> 100% Cash
             pass

        # 2. Apply Per-Asset Risk Constraints (The "Risk-First" Layer)
        final_target_weights = {}
        risk_meta = {}
        
        equity = getattr(current_portfolio, "total_equity", 100_000.0)
        
        for tk, raw_w in norm_weights.items():
            
            # Dollar Conviction Intent
            # e.g. 50% weight -> $50k intent
            intent_dollars = raw_w * equity
            
            # Conviction Series for Risk Manager (expects Series)
            conv_s = pd.Series([intent_dollars], index=[timestamp]) 
            
            # Get Price/Vol history
            px_hist = prices.get(tk, pd.Series())
            vol_hist = volumes.get(tk, pd.Series())
            
            # Call Risk Manager
            # enforce_limits returns ADJUSTED conviction in DOLLARS
            if self.risk_manager:
                res = self.risk_manager.enforce_limits(conv_s, px_hist, vol_hist)
                adj_dollars = res.adjusted_conviction.iloc[-1]
                
                # Convert back to Weight
                adj_w = adj_dollars / equity
                final_target_weights[tk] = adj_w
                risk_meta[tk] = res.violations
            else:
                final_target_weights[tk] = raw_w * self.max_leverage

        # 3. Generate Orders (Diff Logic)
        orders = []
        current_equity = equity
        
        # Get Current Positions
        # Portfolio.positions -> Dict[str, float] (Quantity)
        current_positions = getattr(current_portfolio, "positions", {})
        
        # Handle all tickers (in target OR in current)
        all_tickers = set(final_target_weights.keys()) | set(current_positions.keys())
        
        for tk in all_tickers:
            target_w = final_target_weights.get(tk, 0.0)
            target_qty = 0
            
            # Calculate Target Qty
            current_price = 0.0
            if tk in prices and not prices[tk].empty:
                current_price = prices[tk].iloc[-1]
            elif tk in _global_price_cache: # Fallback if main passes cache separately? 
                # Allocator shouldn't rely on global, but prices arg should have latest
                pass
                
            if current_price > 0:
                target_val = target_w * current_equity
                target_qty = int(target_val / current_price)
                
            # DEBUG: Diagnose zero trades
            if (target_qty != 0) or (signals.get(tk, 0) != 0.5):
                 logger.error(f"DEBUG ALLOC: {tk} Sig={signals.get(tk)} W={target_w:.4f} Price={current_price:.2f} Eq={current_equity:.0f} TgtQty={target_qty} CurrQty={current_positions.get(tk, 0)}")

            current_qty = current_positions.get(tk, 0)
            
            # Delta
            delta_qty = target_qty - current_qty
            
            if delta_qty != 0:
                # Order Generation
                action = OrderType.MARKET
                # Create Order
                orders.append(Order(tk, delta_qty, action, timestamp))
            else:
                pass
                # logger.debug(f"Alloc: {tk} Delta 0. Target: {target_qty} Curr: {current_qty}")

        if not orders and total_signal_score > 0.001:
             logger.info(f"Alloc Debug: Sigs: {signals} -> Norm: {norm_weights} -> RiskAdj: {final_target_weights} -> Orders: {orders}")
                
        return AllocationResult(
            target_weights=final_target_weights,
            orders=orders,
            meta=risk_meta
        )
