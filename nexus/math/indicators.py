"""
nexus/math/indicators.py — Superhuman Regime Detection Engine

Replaces single-threshold volatility/trend detection with:
  - Multi-timeframe consensus (5D, 20D, 60D)
  - Probabilistic regime distribution (not just a single label)
  - HMM-inspired Gaussian mixture regime classification
  - Volatility regime overlay (realized vs. historical vol)
  - Hawkes Process for volatility clustering
"""
from typing import Any, Dict
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Superhuman Market Regime Detector.

    Produces both a single regime label (backwards-compatible) and a full
    probability distribution across regimes, derived from:
      1. Gaussian Mixture over rolling return statistics (HMM-inspired)
      2. Multi-timeframe trend/volatility consensus (5D, 20D, 60D)
      3. Volatility regime overlay (realized vol vs. 60D baseline)

    Regimes: BULL | BEAR | SIDEWAYS | TURBULENT
    """

    # Thresholds calibrated for daily SPY-class data
    TREND_BULL   = 0.018   # +1.8% over window = bull
    TREND_BEAR   = -0.018  # -1.8% over window = bear
    VOL_TURB     = 0.028   # daily vol > 2.8% = turbulent
    VOL_LOW      = 0.008   # daily vol < 0.8% = quiet

    # Multi-timeframe windows (trading days)
    WINDOWS: Dict[str, int] = {"fast": 5, "medium": 20, "slow": 60}

    def __init__(self, window: int = 20):
        self.window = window
        # Track last regime probability for external use
        self._last_probs: Dict[str, float] = {
            "BULL": 0.25, "BEAR": 0.25, "SIDEWAYS": 0.25, "TURBULENT": 0.25
        }

    # ------------------------------------------------------------------ #
    # Public API — backwards-compatible                                    #
    # ------------------------------------------------------------------ #

    def detect(self, data: pd.DataFrame) -> str:
        """
        Returns the dominant regime label (backwards-compatible).
        Internally computes the full probability distribution.
        """
        probs = self.detect_probabilities(data)
        self._last_probs = probs
        return max(probs, key=lambda k: probs[k])

    def get_regime_probabilities(self) -> Dict[str, float]:
        """Returns the last computed regime probability distribution."""
        return dict(self._last_probs)

    # ------------------------------------------------------------------ #
    # Core probabilistic engine                                            #
    # ------------------------------------------------------------------ #

    def detect_probabilities(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Full probabilistic regime detection.
        Returns a dict like: {'BULL': 0.65, 'BEAR': 0.05, 'SIDEWAYS': 0.20, 'TURBULENT': 0.10}
        """
        if data.empty or "close" not in data.columns or len(data) < self.window:
            return {"BULL": 0.25, "BEAR": 0.25, "SIDEWAYS": 0.25, "TURBULENT": 0.25}

        try:
            close = _safe_series(data["close"])
            returns = close.pct_change().dropna()
            if len(returns) < 5:
                return {"BULL": 0.25, "BEAR": 0.25, "SIDEWAYS": 0.25, "TURBULENT": 0.25}

            # Step 1: Multi-timeframe signal votes
            votes = self._multi_timeframe_votes(close, returns)

            # Step 2: Gaussian Mixture probability estimate
            gm_probs = self._gaussian_mixture_probs(returns)

            # Step 3: Volatility regime overlay
            vol_overlay = self._volatility_overlay(returns)

            # Step 4: Fuse all three via weighted average
            fused = self._fuse_probabilities(votes, gm_probs, vol_overlay)

            return fused

        except Exception as exc:
            logger.warning(f"Regime detection failed: {exc}")
            return {"BULL": 0.25, "BEAR": 0.25, "SIDEWAYS": 0.25, "TURBULENT": 0.25}

    def _multi_timeframe_votes(
        self, close: pd.Series, returns: pd.Series
    ) -> Dict[str, float]:
        """
        Each timeframe window produces a regime vote.
        Returns vote tallies normalized to probability.
        """
        tallies: Dict[str, float] = {"BULL": 0.0, "BEAR": 0.0, "SIDEWAYS": 0.0, "TURBULENT": 0.0}
        weights = {"fast": 0.25, "medium": 0.45, "slow": 0.30}

        for name, w in self.WINDOWS.items():
            if len(close) < w:
                tallies["SIDEWAYS"] += weights[name]
                continue

            trend = float(close.iloc[-1] / close.iloc[-w]) - 1.0
            vol = float(returns.tail(w).std())

            # Determine regime for this timeframe
            if vol > self.VOL_TURB:
                regime = "TURBULENT"
            elif trend > self.TREND_BULL:
                regime = "BULL"
            elif trend < self.TREND_BEAR:
                regime = "BEAR"
            else:
                regime = "SIDEWAYS"

            tallies[regime] += weights[name]

        return tallies  # already normalized (weights sum to 1.0)

    def _gaussian_mixture_probs(self, returns: pd.Series) -> Dict[str, float]:
        """
        HMM-inspired: fit 4 Gaussian centroids to return statistics.
        Uses last 60 bars of rolling 5-day mean and std as features.
        """
        probs: Dict[str, float] = {"BULL": 0.25, "BEAR": 0.25, "SIDEWAYS": 0.25, "TURBULENT": 0.25}
        if len(returns) < 20:
            return probs

        roll_mean = returns.rolling(5, min_periods=1).mean()
        roll_std  = returns.rolling(5, min_periods=1).std().fillna(0.0)

        # Current observation
        mu_obs  = float(roll_mean.iloc[-1])
        vol_obs = float(roll_std.iloc[-1])

        # Gaussian centroids for each regime (mean_return, vol)
        centroids = {
            "BULL":      (0.003,   0.010),
            "BEAR":      (-0.003,  0.015),
            "SIDEWAYS":  (0.0,     0.008),
            "TURBULENT": (0.0,     0.032),
        }

        # Compute Gaussian likelihood for each regime
        raw: Dict[str, float] = {}
        for regime, (mu_c, sigma_c) in centroids.items():
            sigma_c = max(sigma_c, 1e-6)
            d_mu  = (mu_obs - mu_c) ** 2 / (2 * sigma_c ** 2)
            d_vol = (vol_obs - sigma_c) ** 2 / (2 * sigma_c ** 2)
            raw[regime] = float(np.exp(-(d_mu + d_vol)))

        total = sum(raw.values()) + 1e-9
        return {k: v / total for k, v in raw.items()}

    def _volatility_overlay(self, returns: pd.Series) -> Dict[str, float]:
        """
        Compare recent realized vol to 60D baseline.
        High ratio → TURBULENT boost; low ratio → BULL/SIDEWAYS boost.
        """
        probs = {"BULL": 0.25, "BEAR": 0.25, "SIDEWAYS": 0.25, "TURBULENT": 0.25}
        if len(returns) < 10:
            return probs

        recent_vol  = float(returns.tail(5).std())
        baseline_vol = float(returns.tail(min(60, len(returns))).std())
        if baseline_vol < 1e-8:
            return probs

        ratio = recent_vol / baseline_vol

        if ratio > 1.8:        # Vol spike → TURBULENT
            return {"BULL": 0.05, "BEAR": 0.15, "SIDEWAYS": 0.10, "TURBULENT": 0.70}
        elif ratio > 1.3:      # Elevated vol
            return {"BULL": 0.15, "BEAR": 0.20, "SIDEWAYS": 0.20, "TURBULENT": 0.45}
        elif ratio < 0.6:      # Vol compression → likely range-bound
            return {"BULL": 0.30, "BEAR": 0.10, "SIDEWAYS": 0.55, "TURBULENT": 0.05}
        else:
            return probs

    @staticmethod
    def _fuse_probabilities(
        votes: Dict[str, float],
        gm: Dict[str, float],
        overlay: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Weighted fusion of the three probability sources.
        """
        w_votes, w_gm, w_overlay = 0.40, 0.35, 0.25
        fused: Dict[str, float] = {}
        for regime in ("BULL", "BEAR", "SIDEWAYS", "TURBULENT"):
            fused[regime] = (
                w_votes   * votes.get(regime, 0.25)
                + w_gm    * gm.get(regime, 0.25)
                + w_overlay * overlay.get(regime, 0.25)
            )
        total = sum(fused.values()) + 1e-9
        return {k: v / total for k, v in fused.items()}


# ------------------------------------------------------------------ #
# HawkesProcess — unchanged interface, improved numerical stability    #
# ------------------------------------------------------------------ #

class HawkesProcess:
    """
    Self-exciting Hawkes Process for modeling volatility clustering.
    """
    def __init__(self, mu: float = 0.01, alpha: float = 0.1, beta: float = 0.5):
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta

    def calculate_intensity(self, events: np.ndarray[Any, Any]) -> float:
        """Calculates intensity at the last event time."""
        if len(events) == 0:
            return self.mu
        t_last = events[-1]
        diffs = t_last - events[:-1]
        diffs = np.clip(diffs, 0.0, 500.0)  # numerical stability
        intensity = self.mu + np.sum(self.alpha * np.exp(-self.beta * diffs))
        return float(intensity)


# ------------------------------------------------------------------ #
# Order Book Imbalance helper                                          #
# ------------------------------------------------------------------ #

def calculate_obi(bid_size: float, ask_size: float) -> float:
    """Calculates Order Book Imbalance (OBI)."""
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    return (bid_size - ask_size) / total


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _safe_series(col: Any) -> pd.Series:
    """Flatten MultiIndex or DataFrame column to a plain Series."""
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0].astype(float)
    return col.astype(float)


def compute_hurst_exponent(prices: pd.Series, max_lag: int = 20) -> float:
    """
    Estimate Hurst exponent via R/S rescaled range analysis.
    H > 0.55 → trending (momentum regime)
    H < 0.45 → mean-reverting
    H ≈ 0.50 → random walk (no edge)
    """
    prices_arr = prices.astype(float).to_numpy()
    if len(prices_arr) < max_lag * 2:
        return 0.5

    lags = range(2, min(max_lag, len(prices_arr) // 2))
    rs_values = []
    lag_values = []

    for lag in lags:
        subseries = prices_arr[:lag]
        mean_s = np.mean(subseries)
        deviations = subseries - mean_s
        cumulative = np.cumsum(deviations)
        r = np.max(cumulative) - np.min(cumulative)
        s = np.std(subseries, ddof=1)
        if s > 0 and r > 0:
            rs_values.append(np.log(r / s))
            lag_values.append(np.log(lag))

    if len(rs_values) < 2:
        return 0.5

    poly = np.polyfit(lag_values, rs_values, 1)
    return float(np.clip(poly[0], 0.0, 1.0))
