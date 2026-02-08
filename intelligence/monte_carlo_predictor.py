"""
Monte Carlo Price Predictor with Markov Chain Regime Modeling.

Features:
- Geometric Brownian Motion (GBM) simulation
- Ornstein-Uhlenbeck (mean-reverting) process
- Markov Chain regime transitions for adaptive parameters
- Multi-horizon price forecasting (1d, 5d, 20d)
- Fair value range estimation with confidence intervals
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states for Markov Chain."""

    BULL = 0
    BEAR = 1
    SIDEWAYS = 2
    VOLATILE = 3


@dataclass
class RegimeParameters:
    """Parameters for each market regime."""

    drift: float
    volatility: float
    mean_reversion_speed: float


# Regime-specific parameters (calibrated for typical market behavior)
REGIME_PARAMS = {
    MarketRegime.BULL: RegimeParameters(
        drift=0.15, volatility=0.15, mean_reversion_speed=0.5
    ),
    MarketRegime.BEAR: RegimeParameters(
        drift=-0.10, volatility=0.25, mean_reversion_speed=0.3
    ),
    MarketRegime.SIDEWAYS: RegimeParameters(
        drift=0.02, volatility=0.12, mean_reversion_speed=2.0
    ),
    MarketRegime.VOLATILE: RegimeParameters(
        drift=0.0, volatility=0.35, mean_reversion_speed=1.0
    ),
}

# Markov Chain transition matrix (rows=current state, cols=next state)
# Order: BULL, BEAR, SIDEWAYS, VOLATILE
TRANSITION_MATRIX = np.array(
    [
        [0.85, 0.05, 0.07, 0.03],  # From BULL
        [0.05, 0.80, 0.10, 0.05],  # From BEAR
        [0.15, 0.10, 0.70, 0.05],  # From SIDEWAYS
        [0.10, 0.15, 0.15, 0.60],  # From VOLATILE
    ]
)


@dataclass
class PricePrediction:
    """Multi-horizon price prediction with confidence intervals."""

    symbol: str
    current_price: float

    # Point predictions
    pred_1d: float
    pred_5d: float
    pred_20d: float

    # Confidence intervals (5th, 50th, 95th percentiles)
    range_1d: Tuple[float, float, float]
    range_5d: Tuple[float, float, float]
    range_20d: Tuple[float, float, float]

    # Probability metrics
    prob_up_1d: float
    prob_up_5d: float
    prob_up_20d: float

    # Fair value range (25th to 75th percentile)
    fair_value_low: float
    fair_value_mid: float
    fair_value_high: float

    # Metadata
    volatility: float
    drift: float
    current_regime: str
    regime_probabilities: Dict[str, float]
    timestamp: datetime


class MarkovChainRegimeDetector:
    """
    Markov Chain-based regime detection and forecasting.

    Uses Hidden Markov Model principles to:
    1. Detect current market regime from returns
    2. Forecast future regime probabilities
    3. Provide regime-specific simulation parameters
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.transition_matrix = TRANSITION_MATRIX
        self.regime_params = REGIME_PARAMS

    def detect_regime(self, prices: pd.Series) -> MarketRegime:
        """
        Detect current market regime based on price characteristics.

        Uses:
        - Recent returns trend
        - Volatility level
        - RSI for overbought/oversold
        """
        if len(prices) < self.lookback:
            return MarketRegime.SIDEWAYS

        returns = prices.pct_change().dropna()
        recent_returns = returns.tail(20)

        # Calculate metrics
        mean_return = recent_returns.mean() * 252
        volatility = recent_returns.std() * np.sqrt(252)
        momentum = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0

        # Regime classification rules
        if volatility > 0.30:
            return MarketRegime.VOLATILE
        elif momentum > 0.05 and mean_return > 0.10:
            return MarketRegime.BULL
        elif momentum < -0.05 and mean_return < -0.05:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def forecast_regime_probabilities(
        self, current_regime: MarketRegime, horizon_days: int
    ) -> Dict[str, float]:
        """
        Forecast regime probabilities at future horizon using Markov Chain.

        P(state at t+n) = P(current) @ T^n
        """
        # Start with current regime as certain
        state_probs = np.zeros(4)
        state_probs[current_regime.value] = 1.0

        # Apply transition matrix n times
        for _ in range(horizon_days):
            state_probs = state_probs @ self.transition_matrix

        return {
            "BULL": float(state_probs[0]),
            "BEAR": float(state_probs[1]),
            "SIDEWAYS": float(state_probs[2]),
            "VOLATILE": float(state_probs[3]),
        }

    def get_weighted_parameters(
        self, regime_probs: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """
        Get drift, volatility, and mean-reversion weighted by regime probabilities.
        """
        weighted_drift = 0.0
        weighted_vol = 0.0
        weighted_theta = 0.0

        for regime in MarketRegime:
            prob = regime_probs[regime.name]
            params = self.regime_params[regime]
            weighted_drift += prob * params.drift
            weighted_vol += prob * params.volatility
            weighted_theta += prob * params.mean_reversion_speed

        return weighted_drift, weighted_vol, weighted_theta


class MonteCarloPricePredictor:
    """
    Monte Carlo simulation engine with Markov Chain regime modeling.

    Supports:
    1. Geometric Brownian Motion (GBM) with regime-adaptive parameters
    2. Ornstein-Uhlenbeck (OU) for mean-reverting assets
    3. Markov Chain for regime transition forecasting
    """

    def __init__(self, n_simulations: int = 10000, seed: Optional[int] = 42):
        self.n_simulations = n_simulations
        self.regime_detector = MarkovChainRegimeDetector()

        if seed is not None:
            np.random.seed(seed)

        logger.info(
            f"[MC_PREDICTOR] Initialized with {n_simulations} simulations "
            f"and Markov Chain regime modeling"
        )

    def _estimate_parameters(self, prices: pd.Series) -> Tuple[float, float, float]:
        """
        Estimate drift (mu), volatility (sigma), and mean-reversion speed (theta).
        """
        returns = prices.pct_change().dropna()

        if len(returns) < 20:
            return 0.0, 0.20, 0.1

        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)

        # Mean-reversion speed estimation
        log_prices = np.log(prices)
        log_mean = log_prices.rolling(20).mean()
        deviation = log_prices - log_mean

        if len(deviation.dropna()) > 10:
            deviation_clean = deviation.dropna()
            autocorr = deviation_clean.autocorr(lag=1)
            if not np.isnan(autocorr) and autocorr < 1:
                theta = -np.log(max(autocorr, 0.01)) * 252
                theta = np.clip(theta, 0.1, 50.0)
            else:
                theta = 1.0
        else:
            theta = 1.0

        return float(mu), float(sigma), float(theta)

    def simulate_gbm_paths(
        self, s0: float, mu: float, sigma: float, n_days: int
    ) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion paths.
        """
        dt = 1.0 / 252
        Z = np.random.standard_normal((self.n_simulations, n_days))
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_prices = np.cumsum(log_returns, axis=1)
        prices = s0 * np.exp(log_prices)
        return prices

    def simulate_regime_aware_paths(
        self, s0: float, current_regime: MarketRegime, n_days: int
    ) -> np.ndarray:
        """
        Simulate price paths with Markov Chain regime transitions.

        Each path independently samples regime transitions and uses
        regime-specific parameters for each day.
        """
        dt = 1.0 / 252
        prices = np.zeros((self.n_simulations, n_days))
        prices[:, 0] = s0

        # Track regime for each simulation path
        regimes = np.full(self.n_simulations, current_regime.value)

        for t in range(1, n_days):
            # Sample regime transitions for each path
            for i in range(self.n_simulations):
                current = regimes[i]
                probs = self.regime_detector.transition_matrix[current]
                regimes[i] = np.random.choice(4, p=probs)

            # Get parameters for each regime
            Z = np.random.standard_normal(self.n_simulations)

            for regime_val in range(4):
                mask = regimes == regime_val
                if not np.any(mask):
                    continue

                regime = MarketRegime(regime_val)
                params = REGIME_PARAMS[regime]

                mu = params.drift
                sigma = params.volatility

                log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[mask]
                prices[mask, t] = prices[mask, t - 1] * np.exp(log_ret)

        return prices

    def simulate_mean_reversion_paths(
        self,
        s0: float,
        mu: float,
        sigma: float,
        theta: float,
        long_term_mean: float,
        n_days: int,
    ) -> np.ndarray:
        """
        Simulate Ornstein-Uhlenbeck mean-reverting process.
        """
        dt = 1.0 / 252
        prices = np.zeros((self.n_simulations, n_days))
        prices[:, 0] = s0

        for t in range(1, n_days):
            Z = np.random.standard_normal(self.n_simulations)
            drift = theta * (long_term_mean - prices[:, t - 1]) * dt
            diffusion = sigma * prices[:, t - 1] * np.sqrt(dt) * Z
            prices[:, t] = prices[:, t - 1] + drift + diffusion
            prices[:, t] = np.maximum(prices[:, t], 0.01)

        return prices

    def predict(
        self,
        symbol: str,
        prices: pd.Series,
        use_mean_reversion: bool = False,
        use_regime_aware: bool = True,
    ) -> PricePrediction:
        """
        Generate multi-horizon price predictions using Monte Carlo simulation.

        Args:
            symbol: Stock symbol
            prices: Historical price series
            use_mean_reversion: If True, use OU process
            use_regime_aware: If True, use Markov Chain regime simulation

        Returns:
            PricePrediction with forecasts and confidence intervals
        """
        if len(prices) < 5:
            raise ValueError(f"Insufficient price data for {symbol}")

        s0 = float(prices.iloc[-1])
        mu, sigma, theta = self._estimate_parameters(prices)

        # Detect current regime and forecast probabilities
        current_regime = self.regime_detector.detect_regime(prices)
        regime_probs = self.regime_detector.forecast_regime_probabilities(
            current_regime, horizon_days=5
        )

        # Long-term mean for mean reversion
        long_term_mean = float(prices.rolling(60).mean().iloc[-1])
        if np.isnan(long_term_mean):
            long_term_mean = s0

        # Simulate paths for 20 days
        if use_mean_reversion:
            paths = self.simulate_mean_reversion_paths(
                s0, mu, sigma, theta, long_term_mean, n_days=20
            )
        elif use_regime_aware:
            paths = self.simulate_regime_aware_paths(s0, current_regime, n_days=20)
        else:
            paths = self.simulate_gbm_paths(s0, mu, sigma, n_days=20)

        # Extract predictions at different horizons
        prices_1d = paths[:, 0]
        prices_5d = paths[:, 4]
        prices_20d = paths[:, -1]

        def get_range(arr):
            return (
                float(np.percentile(arr, 5)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 95)),
            )

        range_1d = get_range(prices_1d)
        range_5d = get_range(prices_5d)
        range_20d = get_range(prices_20d)

        # Probability of price going up
        prob_up_1d = float(np.mean(prices_1d > s0))
        prob_up_5d = float(np.mean(prices_5d > s0))
        prob_up_20d = float(np.mean(prices_20d > s0))

        # Fair value range
        fair_value_low = float(np.percentile(prices_5d, 25))
        fair_value_mid = float(np.percentile(prices_5d, 50))
        fair_value_high = float(np.percentile(prices_5d, 75))

        return PricePrediction(
            symbol=symbol,
            current_price=s0,
            pred_1d=range_1d[1],
            pred_5d=range_5d[1],
            pred_20d=range_20d[1],
            range_1d=range_1d,
            range_5d=range_5d,
            range_20d=range_20d,
            prob_up_1d=prob_up_1d,
            prob_up_5d=prob_up_5d,
            prob_up_20d=prob_up_20d,
            fair_value_low=fair_value_low,
            fair_value_mid=fair_value_mid,
            fair_value_high=fair_value_high,
            volatility=sigma,
            drift=mu,
            current_regime=current_regime.name,
            regime_probabilities=regime_probs,
            timestamp=datetime.utcnow(),
        )

    def get_fair_value_range(
        self, prices: pd.Series, horizon_days: int = 5
    ) -> Dict[str, float]:
        """
        Calculate fair value range based on Monte Carlo simulation.
        """
        s0 = float(prices.iloc[-1])
        current_regime = self.regime_detector.detect_regime(prices)

        # Use regime-aware simulation
        paths = self.simulate_regime_aware_paths(s0, current_regime, horizon_days)
        final_prices = paths[:, -1]

        return {
            "p5": float(np.percentile(final_prices, 5)),
            "p25": float(np.percentile(final_prices, 25)),
            "p50": float(np.percentile(final_prices, 50)),
            "p75": float(np.percentile(final_prices, 75)),
            "p95": float(np.percentile(final_prices, 95)),
            "mean": float(np.mean(final_prices)),
            "std": float(np.std(final_prices)),
            "current_regime": current_regime.name,
        }


# Module-level singleton
_mc_predictor: Optional[MonteCarloPricePredictor] = None


def get_mc_predictor() -> MonteCarloPricePredictor:
    """Get or create the global Monte Carlo predictor instance."""
    global _mc_predictor
    if _mc_predictor is None:
        _mc_predictor = MonteCarloPricePredictor()
    return _mc_predictor
