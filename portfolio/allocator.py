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
    
    def __init__(self, risk_manager, max_leverage: float = 1.0, allow_short: bool = False, target_vol: float = 0.15):
        self.risk_manager = risk_manager
        self.max_leverage = max_leverage
        self.allow_short = allow_short
        self.target_vol = target_vol
        
        # Institutional State Trackers
        self.last_signals: Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}
        self.entry_timestamps: Dict[str, pd.Timestamp] = {}
        
        # Configurable Thresholds
        self.entry_threshold = 0.65
        self.exit_threshold = 0.55
        self.single_name_cap = 0.01  # 1% NAV
        self.cooldown_period = pd.Timedelta(days=3)
        self.stop_timestamps: Dict[str, pd.Timestamp] = {}
        
        # Bucket weights (Requirement C)
        self.bucket_weights = {"Large": 0.50, "Mid": 0.30, "Small": 0.20}
        self.bucket_thresholds = {"Large": 10e9, "Mid": 2e9}
        
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
        metadata: Optional[Dict[str, Dict]] = None,
        method: str = "risk_parity"
    ) -> AllocationResult:
        """
        Main allocation routine.
        1. Normalize Signals -> Raw Weights
        2. Apply Risk Constraints -> Risk-Adjusted Weights
        3. Diff against Current Portfolio -> Position Deltas
        4. Generate Orders
        """
        # 0. INSTITUTIONAL: Buy/Sell Decision Layer (Hysteresis & Trend)
        current_positions = getattr(current_portfolio, "positions", {})
        processed_signals = {}
        
        for tk, sig in signals.items():
            prev_sig = self.last_signals.get(tk, 0.5)
            has_pos = abs(current_positions.get(tk, 0)) > 1e-6
            
            # A. Exit Logic (Hysteresis)
            if has_pos and sig < self.exit_threshold:
                # Conviction collapsed below buffer - Exit
                processed_signals[tk] = 0.5 
                continue
                
            # B. Entry Logic (Confirmation)
            if not has_pos:
                # Intelligent Re-entry: Check Cooldown
                if tk in self.stop_timestamps:
                    if (timestamp - self.stop_timestamps[tk]) < self.cooldown_period:
                        processed_signals[tk] = 0.5
                        continue
                    else:
                        self.stop_timestamps.pop(tk)

                if sig < self.entry_threshold:
                    # Signal not strong enough to overcome friction/entry barriers
                    processed_signals[tk] = 0.5
                    continue
                # Trend confirmation: Buy only if conviction is rising or extremely high
                if sig <= prev_sig and sig < 0.85: 
                    processed_signals[tk] = 0.5
                    continue
            
            processed_signals[tk] = sig
            
        # Update last signals for next delta calculation
        self.last_signals = signals.copy()
        signals = processed_signals # Re-route to filtered signals
        
        # INSTITUTIONAL: Explicit fill_method + immediate cleaning
        returns_hist = {}
        for tk, px in prices.items():
            r = px.pct_change(fill_method=None)
            r = r.replace([np.inf, -np.inf], np.nan).dropna()
            returns_hist[tk] = pd.to_numeric(r, errors='coerce').astype(float)
        
        # 1. Bucketing & Vol-Scaled Risk Parity (Requirement G)
        equity = getattr(current_portfolio, "total_equity", 100_000.0)
        norm_weights = {}
        buckets = {"Large": [], "Mid": [], "Small": []}
        
        # Partition Universe
        for tk, sig in signals.items():
            if sig == 0.5: continue # Neutral/Filtered
            mcap = 0.0
            if metadata and tk in metadata:
                mcap = metadata[tk].get("market_cap", 0.0)
            
            if mcap >= self.bucket_thresholds["Large"]: b = "Large"
            elif mcap >= self.bucket_thresholds["Mid"]: b = "Mid"
            else: b = "Small"
            buckets[b].append(tk)

        # Calculate Weights per Bucket
        for b_name, tickers in buckets.items():
            if not tickers: continue
            n_bucket = len(tickers)
            b_weight = self.bucket_weights.get(b_name, 0.0)
            
            # Position Vol Target (Requirement G.1)
            pos_vol_target = self.target_vol * b_weight / np.sqrt(n_bucket)
            
            for tk in tickers:
                ret = returns_hist.get(tk, pd.Series())
                vol = ret.std() * np.sqrt(252) if not ret.empty else 0.4 # Default 40%
                vol = max(vol, 0.01)
                
                # Allocation Formula (Requirement G.2)
                # target_w = (pos_vol_target / vol) * bucket_fraction
                # We interpret bucket_fraction as already included in pos_vol_target logic
                target_w = (pos_vol_target / vol) * (signals[tk] - 0.5) * 2.0
                
                # Hard Cap: 1% Single-Name (Requirement C)
                target_w = np.clip(target_w, -self.single_name_cap, self.single_name_cap)
                if not self.allow_short: target_w = max(0.0, target_w)
                
                norm_weights[tk] = target_w

        # 2. Apply Per-Asset Risk Constraints
        final_target_weights = {}
        risk_meta = {}
        reason_codes = {}
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
                reason_codes[tk] = res.primary_reason
            else:
                final_target_weights[tk] = raw_w
                reason_codes[tk] = "REBALANCE"

        # 3. Generate Orders (Diff Logic - THE AUTO-SELL ENGINE)
        orders = []
        current_positions = getattr(current_portfolio, "positions", {})
        all_tickers = set(final_target_weights.keys()) | set(current_positions.keys())
        
        # Institutional Guards
        never_sell = getattr(self.risk_manager, "never_sell_assets", [])
        manual_sell = getattr(self.risk_manager, "manual_approval_on_sell", False)

        for tk in all_tickers:
            target_w = final_target_weights.get(tk, 0.0)
            target_qty = 0
            current_price = 0.0
            
            if tk in prices and not prices[tk].empty:
                current_price = prices[tk].iloc[-1]
                
            if current_price > 0:
                target_val = target_w * equity
                # INSTITUTIONAL: Support fractional shares (round to 4 decimals for Alpaca)
                target_qty = round(float(target_val / current_price), 4)
                
            current_qty = current_positions.get(tk, 0)
            delta_qty = target_qty - current_qty
            
            # --- ADAPTIVE STOP LOSS & STAGNATION ---
            reason = reason_codes.get(tk, "REBALANCE")
            risk_metric = None

            if current_qty > 0 and tk in self.entry_prices and current_price > 0:
                entry_p = self.entry_prices[tk]
                ret_since_entry = (current_price - entry_p) / entry_p
                
                # Volatility-Adjusted Trailing Stop
                px_hist = prices.get(tk, pd.Series())
                if len(px_hist) >= 10:
                    vol = px_hist.pct_change(fill_method=None).std() * np.sqrt(252)
                    stop_dist = max(0.05, vol * 0.4) # Minimum 5% stop, or 40% of annual vol
                    if ret_since_entry < -stop_dist:
                        logger.warning(f"ADAPTIVE STOP: {tk} dropped {ret_since_entry:.1%} below entry ${entry_p:.2f} (Stop: {stop_dist:.1%})")
                        target_qty = 0
                        delta_qty = target_qty - current_qty
                        reason = "RISK_BREACH"
                        risk_metric = f"Adaptive Stop ({stop_dist:.1%})"
                        self.stop_timestamps[tk] = timestamp # Start Cooldown

            if abs(delta_qty) < 1e-8:
                continue

            # --- AUTO-SELL ENGINE LOGIC ---
            # Detect Signal Deterioration / Capital Rotation
            if delta_qty < 0 and reason == "REBALANCE":
                 # CAPITAL_ROTATION: If signals grew for OTHERS but shrank for this one
                 sig = signals.get(tk, 0.5)
                 if sig < 0.5: # Negative conviction
                     reason = "SIGNAL_DECAY"
                 else:
                     reason = "CAPITAL_ROTATION"

            # Check Safety Guards
            if delta_qty < 0: # This is a SELL
                if tk in never_sell:
                    logger.warning(f"SELL VETO: {tk} is in never_sell_assets. Skipping liquidation.")
                    continue
                if manual_sell:
                    logger.warning(f"MANUAL PROTECTION: SELL order for {tk} ({delta_qty}) requires approval. Skipping.")
                    continue

                # Set risk metric if reason is RISK_BREACH
                if reason == "RISK_BREACH" and tk in risk_meta and not risk_metric:
                    risk_metric = "; ".join(risk_meta[tk])

            # Final Order Generation
            orders.append(Order(
                ticker=tk, 
                quantity=delta_qty, 
                order_type=OrderType.MARKET, 
                timestamp=timestamp,
                reason=reason,
                risk_metric_triggered=risk_metric
            ))
            
            # --- UPDATE TRACKING STATE ---
            if target_qty > 0 and current_qty == 0:
                self.entry_prices[tk] = current_price
                self.entry_timestamps[tk] = timestamp
            elif target_qty == 0:
                self.entry_prices.pop(tk, None)
                self.entry_timestamps.pop(tk, None)

        return AllocationResult(
            target_weights=final_target_weights,
            orders=orders,
            meta=risk_meta
        )
