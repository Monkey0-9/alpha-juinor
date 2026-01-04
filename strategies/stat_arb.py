
import numpy as np
import pandas as pd
from typing import Tuple, Optional

class KalmanPairsTrader:
    """
    Kalman Filter Statistical Arbitrage Strategy.
    Estimates the hedge ratio (beta) between two assets (Y = beta * X + alpha) dynamically.
    Trades the spread (error) `e = Y - (beta * X + alpha)`.
    """
    
    def __init__(self, delta: float = 1e-4, vt: float = 1e-3):
        """
        delta: Transition covariance smoothing parameter (process noise).
        vt: Observation covariance (measurement noise).
        """
        self.delta = delta
        self.vt = vt
        
        # State: [hedge_ratio, intercept] (beta, alpha)
        # We start with prior 0.0 for both
        self.state_mean = np.zeros(2) 
        self.state_cov = np.zeros((2, 2))
        self.initialized = False
        
        # Track spread stats for z-score
        self.spread_history = []
        self.spread_std_window = 60
        
    def update(self, x_price: float, y_price: float) -> Tuple[float, float, float]:
        """
        Update filter with latest prices.
        Returns: (hedge_ratio, spread, z_score)
        Equation: Y = beta * X + alpha
        Observation Matrix H = [X, 1]
        """
        # 1. Prediction Step (Random Walk Prior)
        # x(t|t-1) = x(t-1|t-1)
        # P(t|t-1) = P(t-1|t-1) + Q
        # Process Noise Q: allows coefficients to drift
        Q = np.eye(2) * self.delta 
        
        if not self.initialized:
            self.state_mean = np.array([1.0, 0.0]) # Assume Beta=1 initially
            self.state_cov = np.eye(2)
            self.initialized = True
            
        pred_state = self.state_mean
        pred_cov = self.state_cov + Q
        
        # 2. Measurement Update
        # Observation H = [X, 1]
        H = np.array([x_price, 1.0])
        
        # Innovation (Error) y - H*x
        y_expected = np.dot(H, pred_state)
        error = y_price - y_expected
        
        # Innovation Variance S = H*P*H' + R
        S = np.dot(H, np.dot(pred_cov, H.T)) + self.vt
        
        # Kalman Gain K = P*H' / S
        K = np.dot(pred_cov, H.T) / S
        
        # New State x(t|t) = x(t|t-1) + K * error
        new_state = pred_state + K * error
        
        # New Cov P(t|t) = (I - K*H) * P(t|t-1)
        new_cov = np.dot((np.eye(2) - np.outer(K, H)), pred_cov)
        
        self.state_mean = new_state
        self.state_cov = new_cov
        
        hedge_ratio = new_state[0]
        # alpha = new_state[1] # Not usually traded, just accounts for bias
        
        # 3. Z-Score Calculation
        # Spread variance is S (from Kalman) or empirical rolling?
        # Kalman 'S' is the variance of the prediction error *step*, which is useful.
        # But for mean reversion trading, we often check z = error / sqrt(S)
        z_score = error / np.sqrt(S) if S > 0 else 0.0
        
        # Alternative: Empirical Z-score of spread history
        # self.spread_history.append(error)
        # if len(self.spread_history) > self.spread_std_window:
        #    self.spread_history.pop(0)
        # if len(self.spread_history) > 20:
        #    std = np.std(self.spread_history)
        #    z_score = error / std if std > 0 else 0.0
            
        return hedge_ratio, error, z_score
