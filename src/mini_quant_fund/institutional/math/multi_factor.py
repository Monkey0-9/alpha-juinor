import numpy as np
import pandas as pd

class MultiFactorEngine:
    """
    Institutional Multi-Factor Scoring Engine.
    Ranks stocks based on Momentum, Quality (Vol-adjusted returns), and Value.
    """
    def rank_stocks(self, signals, historical_data):
        scores = {}
        for symbol, alpha_strength in signals.items():
            if symbol not in historical_data or historical_data[symbol].empty:
                continue
            
            df = historical_data[symbol]
            # 1. Momentum Factor (3-day return)
            mom = df['close'].pct_change(3).iloc[-1]
            
            # 2. Quality Factor (Sharpe-like ratio)
            returns = df['close'].pct_change().dropna()
            quality = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # 3. Master Institutional Score
            # Weighting: 40% Alpha AI, 30% Momentum, 30% Quality
            master_score = (alpha_strength * 0.4) + (mom * 0.3) + (quality * 0.3)
            scores[symbol] = master_score
            
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
