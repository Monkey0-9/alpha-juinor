
import numpy as np
import pandas as pd
from typing import Any
from contracts import BaseAgent, AgentResult
from alpha_families.normalization import AlphaNormalizer

# Shared Normalizer
normalizer = AlphaNormalizer()

class MomentumAgent(BaseAgent):
    """
    Standard Time-Series Momentum (12-1 month).
    """
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        if len(data) < 252:
            return AgentResult(symbol, self.name, 0.0, 0.0, 0.0, {})

        prices = data['Close']

        # 1. Compute Raw Signal Series (Rolling 12m-1m)
        # We need historical signals for normalization
        # Note: 252d ~ 12m, 21d ~ 1m
        ret_12m = prices.pct_change(252)
        ret_1m = prices.pct_change(21)
        raw_signal_series = ret_12m - ret_1m

        # Current values
        current_signal = raw_signal_series.iloc[-1]

        if not np.isfinite(current_signal):
             return AgentResult(symbol, self.name, 0.0, 0.0, 0.0, {})

        # 2. Normalize
        # Use last 252 days of signal history for Z-score stats
        history = raw_signal_series.iloc[-252:]

        z_score, confidence = normalizer.normalize_signal(
            raw_value=current_signal,
            history=history,
            data_confidence=1.0
        )

        # 3. Construct Distribution
        vol = prices.pct_change().std() * np.sqrt(252)
        dist = normalizer.construct_distribution(z_score, confidence, vol)

        return AgentResult(symbol, self.name, dist['mu'], dist['sigma'], dist['confidence'], dist)

class MeanReversionAgent(BaseAgent):
    """
    Short-term Mean Reversion (RSI/Bollinger).
    """
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        if len(data) < 60:
            return AgentResult(symbol, self.name, 0.0, 0.0, 0.0)

        prices = data['Close']

        # 1. Compute Raw Signal Series (Rolling Z-Score of Price vs MA)
        ma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()

        # Raw signal = -1 * (Price - MA) / Std (Mean Reversion)
        # We invert it so positive signal = bullish (price below mean)
        raw_signal_series = -1.0 * (prices - ma_20) / std_20

        current_signal = raw_signal_series.iloc[-1]

        if not np.isfinite(current_signal):
             return AgentResult(symbol, self.name, 0.0, 0.0, 0.0)

        # 2. Normalize using history of the signal itself
        history = raw_signal_series.iloc[-252:]

        z_score, confidence = normalizer.normalize_signal(
            raw_value=current_signal,
            history=history
        )

        # 3. Construct Distribution
        vol = prices.pct_change().std() * np.sqrt(252)
        # Mean Reversion targets shorter term capture?
        # Normalizer uses standard scale.
        dist = normalizer.construct_distribution(z_score, confidence, vol)

        return AgentResult(symbol, self.name, dist['mu'], dist['sigma'], dist['confidence'], dist)

class VolatilityAgent(BaseAgent):
    """
    Volatility Carry / Short Vol.
    """
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        # Simplified: Short vol if VRP is positive (requires options data ideally)
        # Here: Short vol if realized vol is low and stable
        return AgentResult(symbol, self.name, 0.01, 0.05, 0.3)

class TrendFollowingAgent(BaseAgent):
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        if len(data) < 200: return AgentResult(symbol, self.name, 0.0, 0.0, 0.0)
        sma200 = data['Close'].rolling(200).mean().iloc[-1]
        price = data['Close'].iloc[-1]
        mu = 0.05 if price > sma200 else -0.05
        return AgentResult(symbol, self.name, mu, 0.1, 0.6)

class BreakoutAgent(BaseAgent):
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        if len(data) < 20: return AgentResult(symbol, self.name, 0.0, 0.0, 0.0)
        high_20 = data['High'].rolling(20).max().iloc[-1]
        price = data['Close'].iloc[-1]
        if price >= high_20:
             return AgentResult(symbol, self.name, 0.1, 0.2, 0.7)
        return AgentResult(symbol, self.name, 0.0, 0.0, 0.0)

class RSIDivergenceAgent(BaseAgent):
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        # Placeholder for complex divergence logic
        return AgentResult(symbol, self.name, 0.0, 0.0, 0.0)

class MACDCrossoverAgent(BaseAgent):
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        if len(data) < 26: return AgentResult(symbol, self.name, 0.0, 0.0, 0.0)
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()

        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return AgentResult(symbol, self.name, 0.05, 0.1, 0.6)
        return AgentResult(symbol, self.name, 0.0, 0.0, 0.0)

class BollingerBandwidthAgent(BaseAgent):
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        return AgentResult(symbol, self.name, 0.0, 0.0, 0.0)
