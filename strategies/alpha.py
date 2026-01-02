# strategies/alpha.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np


def _minmax_scale(s: pd.Series) -> pd.Series:
    """Scale series to 0..1 using min/max (safe)."""
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
        sma_short = prices.rolling(self.short).mean()
        dev = prices - sma_short
        vol = prices.pct_change().rolling(self.lookback_std).std().replace(0, np.nan)
        z = (dev / (prices * vol)).fillna(0)  # normalized stretch

        # mean reversion expects negative signal when price is far above SMA (we want to short),
        # but we want a long-only conviction scale. We'll use symmetric reversion magnitude.
        raw = -z  # negative z (price above sma) -> negative raw; we'll take absolute magnitude
        raw = raw.abs()
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        return _minmax_scale(raw)


class CompositeAlpha(Alpha):
    """
    Combine multiple Alpha objects with weights (weights are optional).
    If weights omitted, equal weighting is used.
    """

    def __init__(self, alphas: List[Alpha], weights: List[float] | None = None):
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
            s = a.compute(prices).fillna(0)
            series_list.append(s)

        # align indices
        df = pd.concat(series_list, axis=1).fillna(0)
        df.columns = [f"a{i}" for i in range(df.shape[1])]

        # weighted average
        w = np.array(self.weights) / sum(self.weights)
        conv = (df * w).sum(axis=1)

        # final clamp
        return conv.clip(0, 1)
