"""
alpha_agents/fundamentals.py

Fundamental Alpha Agents.
Checks for valuation ratios (PE, PB) in input data or features.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict
from contracts import BaseAgent, AgentResult
from alpha_families.normalization import AlphaNormalizer

# Shared Normalizer
normalizer = AlphaNormalizer()

class ValueAgent(BaseAgent):
    """
    Value Factor Agent (PE/EBITDA/PB).
    Expects fundamental data in features or extra columns.
    """
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        # Check for fundamental columns
        # Assuming data might have 'pe_ratio' or similar if joined
        val_metric = None

        # 1. Try to find metric
        if 'pe_ratio' in data.columns:
            val_metric = data['pe_ratio']
        elif 'close' in data.columns and 'earnings' in data.columns:
            val_metric = data['close'] / data['earnings']

        if val_metric is None or len(val_metric) < 252:
             # Neutral result if no data
            return AgentResult(symbol, self.name, 0.0, 0.0, 0.0, {'reason': 'no_data'})

        # 2. Compute Signal (Inverted PE -> Earnings Yield)
        # Low PE = High Yield = Bullish
        current_pe = val_metric.iloc[-1]
        if current_pe <= 0:
             return AgentResult(symbol, self.name, 0.0, 0.0, 0.0, {'reason': 'negative_pe'})

        earnings_yield = 1.0 / val_metric
        current_yield = earnings_yield.iloc[-1]

        # 3. Normalize
        history = earnings_yield.iloc[-252:]

        z_score, confidence = normalizer.normalize_signal(
            raw_value=current_yield,
            history=history
        )

        # 4. Construct
        vol = data['Close'].pct_change().std() * np.sqrt(252)
        dist = normalizer.construct_distribution(z_score, confidence, vol)

        return AgentResult(symbol, self.name, dist['mu'], dist['sigma'], dist['confidence'], dist)
