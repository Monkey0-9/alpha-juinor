import numpy as np
import pandas as pd
from typing import Dict, Union

class KellySizer:
    """
    Implements Institutional Kelly Criterion with Volatility Regularization.
    Why? Standard Kelly is too aggressive (variance infinite).
    We use 'Fractional Kelly' penalized by regime volatility.
    
    Formula:
    f* = (mu - r) / sigma^2
    
    Adjusted:
    f_adj = f* * (1 - gamma * (CurrentVol / TargetVol))
    """
    
    def __init__(self, target_vol: float = 0.15, max_leverage: float = 2.0):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.risk_free_rate = 0.04 # 4% T-Bill
        
    def calculate_size(self, returns: pd.Series, regime_scalar: float = 1.0) -> float:
        """
        Calculate optimal position size (fraction of equity).
        """
        if len(returns) < 30:
            return 0.10 # Safe default for insufficient history
            
        # 1. Estimate Parameters (Annualized)
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        variance = sigma ** 2
        
        if variance < 1e-6:
            return 0.0 # Avoid division by zero
            
        # 2. Raw Kelly
        # f = E[excess_return] / Variance
        raw_kelly = (mu - self.risk_free_rate) / variance
        
        # 3. Regularization (The "Steel" Part)
        # Avoid betting massive amounts on low-vol assets that might spike
        # We cap Raw Kelly at Max Leverage
        
        # 4. Volatility Penalty
        # If current vol > target vol, reduce size aggressively
        vol_penalty = 1.0
        if sigma > self.target_vol:
            vol_penalty = self.target_vol / sigma
            
        # 5. Half-Kelly (Industry Standard Safety)
        # Full Kelly is optimal for growth but maximizes drawdown.
        # Half Kelly gives 75% of growth with 50% of variance.
        fractional_kelly = raw_kelly * 0.5
        
        # 6. Apply Adjustments
        final_size = fractional_kelly * vol_penalty * regime_scalar
        
        # 7. Constraints
        # Check Direction constraint (Long Only? Short?)
        # For now assume Long Only logic primarily
        final_size = np.clip(final_size, 0.0, self.max_leverage)
        
        return float(final_size)

def compute_kelly_size(returns: pd.Series, volatility_target: float = 0.15) -> float:
    sizer = KellySizer(target_vol=volatility_target)
    return sizer.calculate_size(returns)
