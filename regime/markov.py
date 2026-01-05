import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class RegimeModel:
    """
    3-State Hidden Markov Model (Simplified) for Market Regime Detection.
    States:
    0: BULL_QUIET (Low Vol, positive trend)
    1: BEAR_VOLATILE (High Vol, negative trend)
    2: PANIC (Extreme Vol, Crash)
    """
    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self.transition_matrix = np.array([
            [0.90, 0.09, 0.01], # Bull -> Bull, Bear, Panic
            [0.15, 0.80, 0.05], # Bear -> Bull, Bear, Panic
            [0.05, 0.20, 0.75]  # Panic -> Bull, Bear, Panic
        ])
        # Priors (Current belief state)
        self.current_probs = np.array([0.8, 0.15, 0.05]) 
        
    def update(self, returns: pd.Series, vix: float = 20.0) -> Dict[str, float]:
        """
        Updates belief state based on latest observation.
        Returns probability of PANIC state.
        """
        if returns.empty:
            return {"panic_prob": 0.0}
            
        # 1. Observation Likelihood (Gaussian approximation)
        # We estimate which state matches current data best
        
        # Bull: Positive ret, Low Vol (<15%)
        # Bear: Negative ret, High Vol (15-40%)
        # Panic: Extreme neg ret, Extreme Vol (>40%)
        
        # Recent realized vol (annualized)
        recent_ret = returns.iloc[-1]
        recent_vol = returns.tail(20).std() * np.sqrt(252)
        if np.isnan(recent_vol): recent_vol = 0.0
        
        likelihoods = np.zeros(3)
        
        # Likelihood 0 (Bull)
        l0_vol = max(0.01, (0.20 - recent_vol)) # Higher likelihood if vol low
        l0_ret = max(0.0, recent_ret + 0.01)
        likelihoods[0] = l0_vol + l0_ret
        
        # Likelihood 1 (Bear)
        l1_vol = max(0.01, 1 - abs(recent_vol - 0.25))
        l1_ret = max(0.0, -recent_ret)
        likelihoods[1] = l1_vol + l1_ret
        
        # Likelihood 2 (Panic)
        # VIX is a strong proxy for Panic state likelihood
        l2_vix = 1.0 if vix > 35 else (vix / 35.0)**2
        l2_ret = max(0.0, -recent_ret * 5) # Panic needs huge drop
        likelihoods[2] = l2_vix + l2_ret
        
        # Normalize Likelihoods
        if likelihoods.sum() == 0:
            likelihoods = np.array([0.33, 0.33, 0.33])
        else:
            likelihoods = likelihoods / likelihoods.sum()
            
        # 2. Bayesian Update with Transition Matrix
        # Predicted Prob = Current_Probs * Transition
        predicted_probs = np.dot(self.current_probs, self.transition_matrix)
        
        # 3. Posterior = Predicted * Likelihood
        posterior = predicted_probs * likelihoods
        self.current_probs = posterior / posterior.sum()
        
        return {
            "bull_prob": self.current_probs[0],
            "bear_prob": self.current_probs[1],
            "panic_prob": self.current_probs[2]
        }
