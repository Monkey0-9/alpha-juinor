# strategies/alpha.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import numpy as np


def _minmax_scale(s: pd.Series) -> pd.Series:
    """Scale series to 0..1 using min/max (safe)."""
    if s.empty:
        return s
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


class Alpha(ABC):
    """Abstract Alpha. Implement compute(prices) -> conviction series (0..1)."""

    @abstractmethod
    def compute(self, prices: pd.Series) -> pd.Series:
        pass


class TrendAlpha(Alpha):
    """
    Trend alpha: uses spread of short / long EMA and recent slope.
    Produces a smooth 0..1 conviction series (higher => stronger trend).
    """

    def __init__(self, short: int = 50, long: int = 200):
        self.short = short
        self.long = long

    def compute(self, prices: pd.Series) -> pd.Series:
        if len(prices) < self.long:
            return pd.Series(0.0, index=prices.index)

        ema_s = prices.ewm(span=self.short, adjust=False).mean()
        ema_l = prices.ewm(span=self.long, adjust=False).mean()

        # raw score = relative distance between short and long EMAs
        score = (ema_s - ema_l) / ema_l
        # also increase score when slope of short EMA is positive
        slope = ema_s.diff(5) / ema_s.shift(5)
        raw = score + 0.5 * slope

        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        return _minmax_scale(raw)


class MeanReversionAlpha(Alpha):
    """
    Mean reversion alpha:
    When price is stretched vs short MA (zscore), expect reversion.
    The signal is larger when price is far from short MA in standardized units.
    """

    def __init__(self, short: int = 5, lookback_std: int = 21):
        self.short = short
        self.lookback_std = lookback_std

    def compute(self, prices: pd.Series) -> pd.Series:
        if len(prices) < max(self.short, self.lookback_std):
            return pd.Series(0.0, index=prices.index)

        sma_short = prices.rolling(self.short).mean()
        dev = prices - sma_short
        vol = prices.pct_change().rolling(self.lookback_std).std().replace(0, np.nan)
        z = (dev / (prices * vol)).fillna(0)  # normalized stretch

        # mean reversion expects negative signal when price is far above SMA (we want to short),
        # but we want a long-only conviction scale. We'll use symmetric reversion magnitude.
        # Ideally, high z (overbought) -> sell signal (low conviction for long only layout)
        # But if we treat it as "reversion opportunity", maybe higher is better?
        # Standard approach:
        # If z < -2 (oversold) -> Buy (High Conviction)
        # If z > 2 (overbought) -> Sell (Low Conviction)
        
        # Invert z so that low z (oversold) becomes high value
        raw = -z 
        
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        return _minmax_scale(raw)


class RSIAlpha(Alpha):
    """
    RSI Alpha:
    RSI < 30 -> High Conviction (Oversold, potential bounce)
    RSI > 70 -> Low Conviction (Overbought)
    """
    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, prices: pd.Series) -> pd.Series:
        if len(prices) < self.period + 1:
            return pd.Series(0.0, index=prices.index)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)

        # We want Alpha 1.0 when RSI is low (oversold)
        # Alpha 0.0 when RSI is high (overbought)
        # Linear map: (100 - RSI) / 100
        raw = (100 - rsi) / 100.0
        return raw.clip(0, 1)


class MACDAlpha(Alpha):
    """
    MACD Crossover Alpha:
    Signal = MACD - SignalLine
    Positive -> Bullish Trend -> High Conviction
    """
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, prices: pd.Series) -> pd.Series:
        if len(prices) < self.slow + self.signal:
            return pd.Series(0.0, index=prices.index)

        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=self.signal, adjust=False).mean()

        hist = macd - sig
        # Higher histogram -> stronger bullish momentum
        return _minmax_scale(hist)


class BollingerBandAlpha(Alpha):
    """
    Bollinger Band Squeeze/Breakout Alpha.
    Here we focus on Mean Reversion: Price touching lower band -> Buy.
    """
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def compute(self, prices: pd.Series) -> pd.Series:
        if len(prices) < self.window:
            return pd.Series(0.0, index=prices.index)

        sma = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std()
        
        upper = sma + (std * self.num_std)
        lower = sma - (std * self.num_std)

        # %B indicator = (Price - Lower) / (Upper - Lower)
        # %B < 0 -> Price below lower band (Oversold) -> High Conviction
        # %B > 1 -> Price above upper band (Overbought) -> Low Conviction
        
        pct_b = (prices - lower) / (upper - lower).replace(0, np.nan)
        
        # Invert: 1 - %B
        # If pct_b is 0 (at lower band), score is 1.0
        # If pct_b is 1 (at upper band), score is 0.0
        score = 1.0 - pct_b
        return score.fillna(0.5).clip(0, 1)


class CompositeAlpha(Alpha):
    """
    Combine multiple Alpha objects with weights.
    """

    def __init__(self, alphas: List[Alpha], weights: Optional[List[float]] = None):
        self.alphas = alphas
        if weights is None:
            self.weights = [1.0] * len(alphas)
        else:
            if len(weights) != len(alphas):
                raise ValueError("weights length must match alphas")
            self.weights = list(weights)

    def compute(self, prices: pd.Series) -> pd.Series:
        # compute each alpha
        series_list = []
        for a in self.alphas:
            # Fix #4 & #8: No silent fallbacks. Fail if component fails.
            s = a.compute(prices).fillna(0)
            series_list.append(s)

        if not series_list:
             return pd.Series(0.0, index=prices.index)

        # align indices
        df = pd.concat(series_list, axis=1).fillna(0)
        
        # efficient weighted sum
        w = np.array(self.weights)
        w_sum = w.sum()
        if w_sum <= 0:
            return pd.Series(0.0, index=prices.index)
            
        w_norm = w / w_sum
        conv = (df * w_norm).sum(axis=1)

        return conv.clip(0, 1)
