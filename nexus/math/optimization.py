"""
nexus/math/optimization.py — Kelly Criterion + IC-Scaled Portfolio Optimizer

Replaces flat normalized weights with:
  - Fractional Kelly Criterion position sizing
  - Information Coefficient (IC) scaling
  - Cross-asset correlation penalty (forces diversification)
  - Sharpe-ranked factor blending with adaptive factor weights
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KellyCriterionSizer:
    """
    Fractional Kelly Criterion position sizing.

    Full Kelly: f* = (p * b - q) / b
      where p = win probability, q = 1-p, b = win/loss ratio

    We use fractional Kelly (default 0.25x) to account for
    model uncertainty and avoid ruin in adverse scenarios.
    """

    def __init__(self, fraction: float = 0.25):
        self.fraction = fraction  # conservative fractional Kelly

    def size_position(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Compute Kelly fraction for a single signal.

        Parameters
        ----------
        win_rate : float   Probability signal is correct (0–1)
        avg_win  : float   Average return when correct (positive)
        avg_loss : float   Average loss when wrong (positive magnitude)

        Returns
        -------
        float   Kelly fraction of capital to allocate (0–1, clipped)
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.02  # minimum allocation

        b = avg_win / avg_loss          # win/loss ratio
        p = max(0.01, min(0.99, win_rate))
        q = 1.0 - p

        kelly_f = (p * b - q) / b
        kelly_f = max(0.0, kelly_f)     # never short via Kelly
        return float(kelly_f * self.fraction)


class InformationCoefficientTracker:
    """
    Tracks rolling Information Coefficient (IC) per signal.

    IC = correlation(predicted_alpha, realized_return_next_period)
    High IC signals get more weight; low IC signals get penalized.
    """

    def __init__(self, window: int = 30):
        self.window = window
        # symbol -> list of (predicted, realized) pairs
        self._history: Dict[str, List[Tuple[float, float]]] = {}

    def record(self, symbol: str, predicted: float, realized: float) -> None:
        if symbol not in self._history:
            self._history[symbol] = []
        self._history[symbol].append((predicted, realized))
        if len(self._history[symbol]) > self.window * 2:
            self._history[symbol] = self._history[symbol][-self.window:]

    def get_ic(self, symbol: str) -> float:
        """Returns rolling IC for symbol (-1 to 1). Defaults to 0 if insufficient data."""
        pairs = self._history.get(symbol, [])
        if len(pairs) < 5:
            return 0.0
        predicted = [p for p, _ in pairs[-self.window:]]
        realized  = [r for _, r in pairs[-self.window:]]
        if np.std(predicted) < 1e-9 or np.std(realized) < 1e-9:
            return 0.0
        corr = float(np.corrcoef(predicted, realized)[0, 1])
        return float(np.clip(corr, -1.0, 1.0))

    def get_win_rate(self, symbol: str) -> float:
        """Returns fraction of signals that had correct directional prediction."""
        pairs = self._history.get(symbol, [])
        if len(pairs) < 5:
            return 0.5  # neutral prior
        correct = sum(
            1 for p, r in pairs[-self.window:]
            if (p > 0 and r > 0) or (p < 0 and r < 0)
        )
        return float(correct / len(pairs[-self.window:]))


class PortfolioOptimizer:
    """
    Superhuman Portfolio Optimizer.

    Replaces naive abs-signal normalization with:
      1. IC-weighted signal scaling
      2. Kelly Criterion position sizing
      3. Cross-asset correlation penalty (>0.70 pairwise → penalize)
      4. Max position cap enforcement
    """

    MAX_POSITION = 0.20  # max 20% in any single name
    MIN_POSITION = 0.01  # floor (below this, skip)
    CORR_THRESHOLD = 0.70  # above this → apply concentration penalty

    def __init__(self) -> None:
        self.ic_tracker = InformationCoefficientTracker(window=30)
        self.kelly_sizer = KellyCriterionSizer(fraction=0.25)

    def optimize_weights(
        self,
        symbols: List[str],
        signals: List[float],
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """
        Calculate optimal Kelly + IC-scaled weights.

        Parameters
        ----------
        symbols : list[str]
        signals : list[float]    Raw alpha signals (-1 to 1)
        historical_data : dict   Optional OHLCV DataFrames for correlation penalty

        Returns
        -------
        dict[symbol → weight]   Portfolio weights (sum ≤ 1.0)
        """
        if not symbols:
            return {}

        signals_arr = np.array(signals, dtype=float)

        # Step 1: IC scaling — amplify high-IC signals, dampen low-IC
        ic_scales = np.array([
            max(0.0, self.ic_tracker.get_ic(s) + 0.5)  # shift so 0 IC → 0.5 scale
            for s in symbols
        ], dtype=float)

        scaled_signals = signals_arr * ic_scales

        # Step 2: Kelly sizing — per-signal position fraction
        kelly_weights: Dict[str, float] = {}
        for sym, sig in zip(symbols, scaled_signals):
            win_rate = self.ic_tracker.get_win_rate(sym)
            avg_win  = max(0.005, abs(sig) * 0.012)  # proxy from signal magnitude
            avg_loss = max(0.003, (1.0 - abs(sig)) * 0.008)
            kw = self.kelly_sizer.size_position(win_rate, avg_win, avg_loss)
            kelly_weights[sym] = kw * abs(float(sig))  # scale by signal confidence

        # Step 3: Correlation penalty
        if historical_data and len(symbols) > 1:
            kelly_weights = self._apply_correlation_penalty(
                kelly_weights, historical_data
            )

        # Step 4: Normalize, cap, floor
        total = sum(kelly_weights.values()) + 1e-9
        weights: Dict[str, float] = {}
        for sym, w in kelly_weights.items():
            normalized = w / total
            normalized = min(normalized, self.MAX_POSITION)
            if normalized >= self.MIN_POSITION:
                weights[sym] = float(normalized)

        # Re-normalize after capping
        final_total = sum(weights.values()) + 1e-9
        return {s: w / final_total for s, w in weights.items()}

    def _apply_correlation_penalty(
        self,
        weights: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Reduce weight for assets highly correlated (>0.70) with larger positions.
        """
        # Build returns matrix
        returns_map: Dict[str, pd.Series] = {}
        for sym in weights:
            if sym in historical_data and not historical_data[sym].empty:
                r = historical_data[sym]["close"].pct_change().dropna()
                returns_map[sym] = r

        if len(returns_map) < 2:
            return weights

        # Align returns to common index
        df = pd.DataFrame(returns_map).dropna(how="all")
        if df.shape[1] < 2 or df.shape[0] < 5:
            return weights

        corr_matrix = df.corr().fillna(0.0)
        adjusted = dict(weights)

        for sym_a in list(adjusted.keys()):
            penalty = 1.0
            for sym_b in list(adjusted.keys()):
                if sym_a == sym_b:
                    continue
                if sym_a in corr_matrix.index and sym_b in corr_matrix.columns:
                    corr = abs(float(corr_matrix.loc[sym_a, sym_b]))
                    if corr > self.CORR_THRESHOLD:
                        # Heavier weight gets less penalty; lighter gets more
                        if adjusted[sym_b] > adjusted[sym_a]:
                            excess = (corr - self.CORR_THRESHOLD) / (1.0 - self.CORR_THRESHOLD)
                            penalty *= (1.0 - 0.40 * excess)
            adjusted[sym_a] *= max(0.20, penalty)

        return adjusted


class MultiFactorEngine:
    """
    Superhuman Multi-Factor Ranking Engine.

    Ranks assets via adaptive factor blending:
      - Alpha (from strategy consensus)
      - Momentum (rate of price change)
      - Quality (low volatility, high Sharpe proxy)
      - Information Ratio proxy (alpha / tracking error)

    Factor weights update dynamically based on rolling IC per factor.
    """

    def __init__(self) -> None:
        # Adaptive factor weights — recalibrated each call
        self._factor_weights = {
            "alpha":    0.40,
            "momentum": 0.30,
            "quality":  0.20,
            "ir":       0.10,
        }

    def rank_assets(
        self,
        signals: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Rank assets by Sharpe-weighted multi-factor score.
        """
        rankings: Dict[str, float] = {}
        factor_scores: Dict[str, Dict[str, float]] = {
            "alpha": {}, "momentum": {}, "quality": {}, "ir": {}
        }

        for symbol, alpha in signals.items():
            data = historical_data.get(symbol, pd.DataFrame())
            if data.empty or "close" not in data.columns:
                factor_scores["alpha"][symbol]    = alpha
                factor_scores["momentum"][symbol] = 0.0
                factor_scores["quality"][symbol]  = 0.0
                factor_scores["ir"][symbol]       = alpha
                continue

            close = data["close"].astype(float)
            returns = close.pct_change().dropna()
            if returns.empty:
                factor_scores["alpha"][symbol]    = alpha
                factor_scores["momentum"][symbol] = 0.0
                factor_scores["quality"][symbol]  = 0.0
                factor_scores["ir"][symbol]       = alpha
                continue

            # Momentum: recent relative performance
            momentum = float((close.iloc[-1] / close.iloc[0]) - 1.0)

            # Quality: inverse of volatility-adjusted returns (low vol = high quality)
            vol = float(returns.std()) + 1e-8
            mean_ret = float(returns.mean())
            sharpe_proxy = mean_ret / vol  # daily Sharpe proxy
            quality = float(np.tanh(sharpe_proxy * 15.0))

            # Information Ratio proxy: alpha / tracking_error vs SPY-like baseline
            tracking_error = vol * np.sqrt(252)
            ir = float(alpha / max(tracking_error, 1e-4))

            factor_scores["alpha"][symbol]    = float(alpha)
            factor_scores["momentum"][symbol] = float(np.tanh(momentum * 10.0))
            factor_scores["quality"][symbol]  = quality
            factor_scores["ir"][symbol]       = float(np.tanh(ir * 2.0))

        # Adaptive factor weights: normalize each factor cross-sectionally
        self._recalibrate_weights(factor_scores, signals)

        for symbol in signals:
            score = sum(
                self._factor_weights[f] * factor_scores[f].get(symbol, 0.0)
                for f in self._factor_weights
            )
            rankings[symbol] = score

        return dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))

    def _recalibrate_weights(
        self,
        factor_scores: Dict[str, Dict[str, float]],
        signals: Dict[str, float],
    ) -> None:
        """
        Boost weights for factors that are aligned with raw alpha signals.
        Simple IC proxy: correlation of factor scores with alpha signals.
        """
        alphas = np.array([signals.get(s, 0.0) for s in signals])
        if np.std(alphas) < 1e-9:
            return  # no variation → keep current weights

        factor_ics: Dict[str, float] = {}
        for factor, scores in factor_scores.items():
            factor_vals = np.array([scores.get(s, 0.0) for s in signals])
            if np.std(factor_vals) < 1e-9:
                factor_ics[factor] = 0.0
                continue
            corr = float(np.corrcoef(factor_vals, alphas)[0, 1])
            factor_ics[factor] = max(0.0, corr)  # only reward positive IC

        total_ic = sum(factor_ics.values()) + 1e-9
        smoothing = 0.30  # blend 30% new IC, 70% old weights
        for factor in self._factor_weights:
            new_w = factor_ics[factor] / total_ic
            self._factor_weights[factor] = (
                (1 - smoothing) * self._factor_weights[factor]
                + smoothing * new_w
            )

        # Normalize
        total = sum(self._factor_weights.values()) + 1e-9
        for factor in self._factor_weights:
            self._factor_weights[factor] /= total


class MonteCarloSimulator:
    """
    Portfolio-level Monte Carlo survival analysis — unchanged interface.
    """

    def run_survival_analysis(
        self,
        initial_capital: float,
        daily_returns: np.ndarray[Any, Any],
        days: int = 252,
        n_simulations: int = 1000,
        ruin_threshold: float = 0.5,
    ) -> float:
        if len(daily_returns) < 2 or initial_capital <= 0 or days <= 0:
            return 0.5

        mu    = float(np.mean(daily_returns))
        sigma = float(np.std(daily_returns))
        if sigma == 0:
            return 1.0

        ruin_level = initial_capital * (1 - ruin_threshold)
        survived   = 0
        rng        = np.random.default_rng(42)

        for _ in range(n_simulations):
            path_returns = rng.normal(mu, sigma, days)
            prices = initial_capital * np.cumprod(1 + path_returns)
            if np.min(prices) > ruin_level:
                survived += 1

        return survived / n_simulations
