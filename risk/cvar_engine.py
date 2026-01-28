"""
risk/cvar_engine.py

CVaR Compute Engine (Ticket 11)

Computes Conditional Value at Risk (CVaR/Expected Shortfall) using:
1. Extreme Value Theory (EVT) for tail modeling
2. Empirical quantiles (fallback)
3. Historical simulation

CVaR is the single most important tail risk metric.
All capital allocation must respect CVaR limits.
"""

import json
import logging
import math
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("CVAR_ENGINE")


@dataclass
class CVaRResult:
    """CVaR calculation result."""
    symbol: str
    confidence_level: float      # e.g., 0.95 for 95%
    var: float                   # Value at Risk (quantile)
    cvar: float                  # Conditional VaR (expected shortfall)
    method: str                  # "EVT", "EMPIRICAL", "PARAMETRIC"
    lookback_days: int
    computed_at: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['metadata'] = self.metadata or {}
        return d


@dataclass
class PortfolioCVaR:
    """Portfolio-level CVaR."""
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    max_loss_1d: float           # Maximum 1-day loss in sample
    gross_exposure: float
    net_exposure: float
    computed_at: str
    positions: Dict[str, float]  # Symbol -> CVaR contribution

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CVaREngine:
    """
    Conditional Value at Risk Computation Engine.

    Implements multiple methods:
    1. EVT (Extreme Value Theory) - Best for true tail estimation
    2. Empirical - Simple quantile-based
    3. Parametric - Assumes normal/t distribution

    Usage:
        engine = CVaREngine()
        result = engine.compute_cvar(returns, confidence=0.95)
    """

    # EVT parameters
    EVT_THRESHOLD_PERCENTILE = 0.10  # Use bottom 10% for GPD fit
    MIN_TAIL_OBSERVATIONS = 30

    def __init__(self, lookback_days: int = 252):
        """
        Initialize CVaREngine.

        Args:
            lookback_days: Default lookback for calculations
        """
        self.lookback_days = lookback_days
        self._cache: Dict[str, CVaRResult] = {}

    def compute_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        symbol: str = "PORTFOLIO",
        method: str = "AUTO"
    ) -> CVaRResult:
        """
        Compute CVaR for a return series.

        Args:
            returns: Daily return series
            confidence: Confidence level (0.95 = 95%)
            symbol: Symbol identifier
            method: "EVT", "EMPIRICAL", "PARAMETRIC", or "AUTO"

        Returns:
            CVaRResult with VaR and CVaR
        """
        now = datetime.utcnow().isoformat() + 'Z'

        # Clean returns
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) < 20:
            logger.warning(f"Insufficient data for CVaR: {len(returns)} points")
            return CVaRResult(
                symbol=symbol,
                confidence_level=confidence,
                var=-0.05,
                cvar=-0.08,
                method="DEFAULT",
                lookback_days=len(returns),
                computed_at=now,
                metadata={"warning": "Insufficient data, using defaults"}
            )

        # Select method
        if method == "AUTO":
            if len(returns) >= self.MIN_TAIL_OBSERVATIONS * 10:
                method = "EVT"
            else:
                method = "EMPIRICAL"

        # Compute based on method
        if method == "EVT":
            var, cvar, metadata = self._compute_evt(returns, confidence)
        elif method == "PARAMETRIC":
            var, cvar, metadata = self._compute_parametric(returns, confidence)
        else:  # EMPIRICAL
            var, cvar, metadata = self._compute_empirical(returns, confidence)

        result = CVaRResult(
            symbol=symbol,
            confidence_level=confidence,
            var=round(var, 6),
            cvar=round(cvar, 6),
            method=method,
            lookback_days=len(returns),
            computed_at=now,
            metadata=metadata
        )

        # Cache
        self._cache[f"{symbol}:{confidence}"] = result

        return result

    def _compute_empirical(
        self,
        returns: pd.Series,
        confidence: float
    ) -> Tuple[float, float, Dict]:
        """
        Empirical (historical) CVaR computation.

        Simple but robust - uses actual return distribution.
        """
        alpha = 1 - confidence  # e.g., 0.05 for 95%

        # Sort returns (smallest first)
        sorted_returns = returns.sort_values()

        # VaR = quantile at alpha
        var = sorted_returns.quantile(alpha)

        # CVaR = average of returns below VaR
        tail = sorted_returns[sorted_returns <= var]
        cvar = tail.mean() if len(tail) > 0 else var

        metadata = {
            "n_observations": len(returns),
            "n_tail_observations": len(tail),
            "min_return": float(sorted_returns.min()),
            "max_return": float(sorted_returns.max()),
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis())
        }

        return float(var), float(cvar), metadata

    def _compute_parametric(
        self,
        returns: pd.Series,
        confidence: float
    ) -> Tuple[float, float, Dict]:
        """
        Parametric CVaR using Student-t distribution.

        Better for fat tails than normal distribution.
        """
        alpha = 1 - confidence

        # Fit t-distribution
        params = stats.t.fit(returns)
        df, loc, scale = params

        # VaR from t-distribution quantile
        var = stats.t.ppf(alpha, df, loc, scale)

        # CVaR for t-distribution (analytical formula)
        x_alpha = stats.t.ppf(alpha, df)
        pdf_at_alpha = stats.t.pdf(x_alpha, df)
        cvar = loc + scale * (-pdf_at_alpha / alpha) * (df + x_alpha**2) / (df - 1)

        metadata = {
            "df": float(df),
            "loc": float(loc),
            "scale": float(scale),
            "distribution": "student_t"
        }

        return float(var), float(cvar), metadata

    def _compute_evt(
        self,
        returns: pd.Series,
        confidence: float
    ) -> Tuple[float, float, Dict]:
        """
        Extreme Value Theory CVaR using Generalized Pareto Distribution.

        Best method for true tail risk estimation.
        Fits GPD to exceedances above threshold.
        """
        # Use losses (negative returns)
        losses = -returns

        # Determine threshold (e.g., 90th percentile of losses)
        threshold = losses.quantile(1 - self.EVT_THRESHOLD_PERCENTILE)

        # Exceedances above threshold
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < self.MIN_TAIL_OBSERVATIONS:
            # Fall back to empirical
            logger.debug(f"EVT: insufficient exceedances ({len(exceedances)}), using empirical")
            return self._compute_empirical(returns, confidence)

        try:
            # Fit GPD to exceedances
            # GPD parameters: shape (xi), scale (sigma)
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

            # Probability of exceeding threshold
            n = len(losses)
            n_u = len(exceedances)
            p_exceed = n_u / n

            # Calculate VaR using EVT formula
            alpha = 1 - confidence  # e.g., 0.05
            q = alpha / p_exceed

            if shape != 0:
                var_excess = (scale / shape) * (q**(-shape) - 1)
            else:
                var_excess = scale * np.log(1 / q)

            var = -(threshold + var_excess)  # Convert back to return

            # CVaR for GPD
            if shape < 1:
                cvar_excess = var_excess / (1 - shape) + (scale - shape * threshold) / (1 - shape)
                cvar = -(threshold + cvar_excess)
            else:
                cvar = var * 1.2  # Approximation for heavy tails

            metadata = {
                "gpd_shape": round(float(shape), 4),
                "gpd_scale": round(float(scale), 4),
                "threshold": round(float(threshold), 4),
                "n_exceedances": int(n_u),
                "exceed_prob": round(float(p_exceed), 4)
            }

            return float(var), float(cvar), metadata

        except Exception as e:
            logger.warning(f"EVT fit failed: {e}, falling back to empirical")
            return self._compute_empirical(returns, confidence)

    def compute_portfolio_cvar(
        self,
        position_returns: Dict[str, pd.Series],
        weights: Dict[str, float]
    ) -> PortfolioCVaR:
        """
        Compute portfolio-level CVaR.

        Args:
            position_returns: Dict of symbol -> return series
            weights: Dict of symbol -> portfolio weight

        Returns:
            PortfolioCVaR with aggregated metrics
        """
        now = datetime.utcnow().isoformat() + 'Z'

        # Calculate weighted portfolio returns
        symbols = list(weights.keys())

        # Align all return series
        returns_df = pd.DataFrame({
            s: position_returns.get(s, pd.Series(dtype=float))
            for s in symbols
        }).dropna()

        if len(returns_df) < 20:
            logger.warning("Insufficient data for portfolio CVaR")
            return PortfolioCVaR(
                var_95=-0.05,
                cvar_95=-0.08,
                var_99=-0.08,
                cvar_99=-0.12,
                max_loss_1d=-0.10,
                gross_exposure=sum(abs(w) for w in weights.values()),
                net_exposure=sum(weights.values()),
                computed_at=now,
                positions={}
            )

        # Portfolio returns (weighted sum)
        portfolio_returns = sum(
            returns_df[s] * weights.get(s, 0)
            for s in symbols if s in returns_df.columns
        )

        # Compute CVaR at 95% and 99%
        result_95 = self.compute_cvar(portfolio_returns, confidence=0.95, symbol="PORTFOLIO")
        result_99 = self.compute_cvar(portfolio_returns, confidence=0.99, symbol="PORTFOLIO")

        # Per-symbol CVaR contribution
        position_cvar = {}
        for s in symbols:
            if s in returns_df.columns:
                r = self.compute_cvar(returns_df[s], confidence=0.95, symbol=s)
                contribution = r.cvar * abs(weights.get(s, 0))
                position_cvar[s] = round(contribution, 6)

        return PortfolioCVaR(
            var_95=result_95.var,
            cvar_95=result_95.cvar,
            var_99=result_99.var,
            cvar_99=result_99.cvar,
            max_loss_1d=float(portfolio_returns.min()) if len(portfolio_returns) > 0 else -0.10,
            gross_exposure=sum(abs(w) for w in weights.values()),
            net_exposure=sum(weights.values()),
            computed_at=now,
            positions=position_cvar
        )

    def get_cached_result(self, symbol: str, confidence: float = 0.95) -> Optional[CVaRResult]:
        """Get cached CVaR result."""
        return self._cache.get(f"{symbol}:{confidence}")

    def clear_cache(self):
        """Clear the result cache."""
        self._cache.clear()


# Singleton instance
_instance: Optional[CVaREngine] = None


def get_cvar_engine() -> CVaREngine:
    """Get singleton CVaREngine instance."""
    global _instance
    if _instance is None:
        _instance = CVaREngine()
    return _instance
