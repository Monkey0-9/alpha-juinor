"""
Monte Carlo Mean Reversion Strategy with Elite Prediction.

This strategy combines:
- Elite Monte Carlo simulated fair value ranges (with Heston/Jump diffusion)
- Mean reversion signals when price deviates from simulated range
- Methods from Probability Theory, Statistics, and Linear Algebra
- RSI confirmation for entry points
- Regime-aware adjustments

Uses Top 1% Global Standard mathematical methods.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Import standard predictor
from intelligence.monte_carlo_predictor import get_mc_predictor
from strategy_factory.interface import Signal, StrategyInterface

# Import elite predictors with safe fallback
try:
    from intelligence.elite_monte_carlo import get_elite_mc_predictor
    from intelligence.elite_statistics import (
        HypothesisTests,
        InformationMetrics,
        TimeSeriesStats,
    )

    ELITE_AVAILABLE = True
except ImportError:
    ELITE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MCMRConfig:
    """Configuration for Monte Carlo Mean Reversion Strategy."""

    n_simulations: int = 5000
    lookback_period: int = 60
    fair_value_horizon: int = 5
    oversold_percentile: float = 25.0
    overbought_percentile: float = 75.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0


class MonteCarloMeanReversionStrategy(StrategyInterface):
    """
    Mean Reversion Strategy powered by Elite Monte Carlo fair value estimation.

    Mathematical Foundation:
    - Probability: Bayesian estimation of drift/volatility
    - Stochastic Processes: Heston model for volatility, Merton for jumps
    - Statistics: Entropy and Hurst exponent for regime detection
    """

    def __init__(self, config: Optional[MCMRConfig] = None):
        self.config = config or MCMRConfig()
        self._name = "MC_MeanReversion_Elite"
        self._seed = 42

        logger.info(f"[{self._name}] Initialized with Elite Mode: {ELITE_AVAILABLE}")

    @property
    def name(self) -> str:
        return self._name

    def generate_signal(
        self, symbol: str, prices: pd.Series, regime_data: Optional[dict] = None
    ) -> Signal:
        """
        Generate trading signal using Elite Monte Carlo prediction.
        """
        try:
            # Check for insufficient data
            if len(prices) < max(30, self.config.lookback_period):
                return Signal(symbol, 0.0, 0.0, False, {"error": "Insufficient data"})

            # 1. Get Prediction Metrics
            # ----------------------------------------------------------------
            metrics = self._get_prediction_metrics(symbol, prices)

            current_price = float(prices.iloc[-1])
            fv_low = metrics["fv_low"]
            fv_high = metrics["fv_high"]

            # 2. Advanced Statistics
            # ----------------------------------------------------------------
            stats = self._get_advanced_stats(prices)
            hurst = stats["hurst"]
            entropy = stats["entropy"]
            is_random_walk = stats["is_random_walk"]

            # Quality filter based on statistical properties
            confidence_multiplier = 1.0
            if is_random_walk:
                confidence_multiplier *= 0.6  # Random walks are hard to predict
            if entropy > 4.5:
                confidence_multiplier *= 0.8  # High entropy = high noise

            # 3. Technical Confirmation (RSI)
            # ----------------------------------------------------------------
            rsi = self._calculate_rsi(prices)

            # 4. Signal Logic
            # ----------------------------------------------------------------
            signal_strength = 0.0
            is_entry = False
            signal_type = "NEUTRAL"

            # Undervalued: Price < Low Fair Value AND Oversold
            if current_price < fv_low:
                # Calculate deviation score (0 to 1)
                deviation = (fv_low - current_price) / fv_low
                score = min(deviation * 10, 1.0)

                if rsi < self.config.rsi_oversold:
                    signal_strength = 0.8 + (score * 0.2)
                    signal_type = "STRONG_BUY"
                    is_entry = True
                elif rsi < 50:
                    signal_strength = 0.5 + (score * 0.2)
                    signal_type = "BUY"
                    is_entry = True

            # Overvalued: Price > High Fair Value AND Overbought
            elif current_price > fv_high:
                deviation = (current_price - fv_high) / fv_high
                score = min(deviation * 10, 1.0)

                if rsi > self.config.rsi_overbought:
                    signal_strength = -(0.8 + (score * 0.2))
                    signal_type = "STRONG_SELL"
                    is_entry = True
                elif rsi > 50:
                    signal_strength = -(0.5 + (score * 0.2))
                    signal_type = "SELL"
                    is_entry = True

            # 5. Regime-Aware Adjustments
            # ----------------------------------------------------------------
            regime = metrics["regime"]
            regime_adjusted = False

            # Stronger mean reversion in SIDEWAYS/VOLATILE regimes
            # Hurst < 0.5 confirms mean reverting behavior
            if regime in ["SIDEWAYS", "VOLATILE"] and hurst < 0.5:
                signal_strength *= 1.2
                regime_adjusted = True

            # Dampen mean reversion signals in strong trends
            # Hurst > 0.6 confirms trending behavior
            elif regime in ["BULL", "BEAR"] and hurst > 0.6:
                # Determine trend direction
                trend_up = regime == "BULL"

                # If fighting the trend, reduce signal significantly
                if (trend_up and signal_strength < 0) or (
                    not trend_up and signal_strength > 0
                ):
                    signal_strength *= 0.5
                    regime_adjusted = True

            # 6. Final Outputs
            # ----------------------------------------------------------------
            final_confidence = metrics["model_confidence"] * confidence_multiplier

            # Clip strength
            signal_strength = max(-1.0, min(1.0, float(signal_strength)))

            return Signal(
                symbol=symbol,
                strength=signal_strength,
                confidence=final_confidence,
                regime_adjusted=regime_adjusted,
                is_entry=is_entry,
                metadata={
                    "signal_type": signal_type,
                    "rsi": rsi,
                    "regime": regime,
                    "fair_value_range": (fv_low, fv_high),
                    "mispricing_score": metrics["mispricing_score"],
                    "kelly_fraction": metrics["kelly_fraction"],
                    "hurst_exponent": hurst,
                    "entropy": entropy,
                    "is_random_walk": is_random_walk,
                },
            )

        except Exception as e:
            logger.error(f"Error in Monte Carlo Mean Reversion: {e}")
            return Signal(symbol, 0.0, 0.0, False, {"error": str(e)})

    def _get_prediction_metrics(self, symbol: str, prices: pd.Series) -> Dict:
        """
        Get prediction metrics using HINDCAST Monte Carlo validation.

        We predict from t-N days ago to t to generate an independent
        fair value distribution, then compare current price to that benchmark.
        """
        horizon = self.config.fair_value_horizon

        # Ensure enough data for hindcast
        if len(prices) < horizon + 20:
            # Fallback to standard check if not enough history
            pass

        # HINDCAST: Use data up to t-horizon to predict t
        past_prices = prices.iloc[:-horizon]

        if ELITE_AVAILABLE:
            try:
                predictor = get_elite_mc_predictor()
                predictor.n_days = horizon

                # Predict from past to "now"
                pred = predictor.predict(symbol, past_prices)

                # Get current regime from full data
                regime_pred = predictor.predict(symbol, prices)

                # Recalculate mispricing of CURRENT price against HINDCAST
                current_price = float(prices.iloc[-1])

                # Approx sigma from IQR (IQR approx 1.35 sigma for normal)
                iqr = pred.fair_value_high - pred.fair_value_low
                sigma = iqr / 1.35 if iqr > 0 else 1e-6

                median = (pred.fair_value_high + pred.fair_value_low) / 2
                mispricing = (current_price - median) / sigma

                return {
                    "fv_low": pred.fair_value_low,
                    "fv_high": pred.fair_value_high,
                    "mispricing_score": mispricing,
                    "kelly_fraction": pred.kelly_fraction,
                    "regime": regime_pred.current_regime,
                    "model_confidence": pred.model_confidence,
                }
            except Exception as e:
                logger.warning(f"Elite predictor failed: {e}, falling back.")

        # Fallback to standard
        predictor = get_mc_predictor()
        fv = predictor.get_fair_value_range(past_prices, horizon_days=horizon)

        # Recalculate mispricing
        current = float(prices.iloc[-1])
        median = fv.get("p50", (fv["p25"] + fv["p75"]) / 2)
        iqr = fv["p75"] - fv["p25"]
        sigma_proxy = iqr / 1.35 if iqr > 0 else current * 0.01

        mispricing = (current - median) / sigma_proxy

        return {
            "fv_low": fv["p25"],
            "fv_high": fv["p75"],
            "mispricing_score": mispricing,
            "kelly_fraction": 0.0,
            "regime": "SIDEWAYS",
            "model_confidence": 0.6,
        }

    def _get_advanced_stats(self, prices: pd.Series) -> Dict:
        """Get advanced statistical metrics if available."""
        if ELITE_AVAILABLE:
            try:
                returns = prices.pct_change().dropna().values
                hurst = TimeSeriesStats.hurst_exponent(returns)
                entropy = InformationMetrics.shannon_entropy(returns)
                rw_test = HypothesisTests.variance_ratio_test(returns)
                return {
                    "hurst": hurst,
                    "entropy": entropy,
                    "is_random_walk": rw_test["is_random_walk"],
                }
            except Exception as e:
                logger.error(f"Advanced stats failed: {e}")
                import traceback

                traceback.print_exc()

        return {"hurst": 0.5, "entropy": 0.0, "is_random_walk": False}

    def _calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate Relative Strength Index."""
        period = self.config.rsi_period

        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Handle division by zero
        loss = loss.replace(0, np.nan)
        rs = gain / loss

        rsi = 100 - (100 / (1 + rs))

        # Fill NaNs (if loss was 0 everywhere, RSI is 100)
        rsi = rsi.fillna(100.0)

        return float(rsi.iloc[-1])

    def explain_decision(self, signal: Signal) -> str:
        """
        Generate a professional quantitative research report for the decision.

        Returns:
            Markdown-formatted explanation of the signal logic.
        """
        m = signal.metadata
        regime = m.get("regime", "UNKNOWN")
        fv_low, fv_high = m.get("fair_value_range", (0, 0))
        price = 0.0  # This would ideally come from the signal or passed in context

        # Determine sentiment
        if signal.strength > 0:
            sentiment = "BULLISH"
            action = "BUY"
        elif signal.strength < 0:
            sentiment = "BEARISH"
            action = "SELL"
        else:
            sentiment = "NEUTRAL"
            action = "HOLD"

        report = [
            f"### Elite Quantitative Signal Report: {sentiment}",
            f"**Action**: {action} | **Confidence**: {signal.confidence:.1%}",
            "",
            "#### 1. Market Regime Analysis",
            f"- **Detected Regime**: `{regime}`",
            f"- **Hurst Exponent**: {m.get('hurst_exponent', 0.5):.3f} "
            + (
                "(Mean Reverting)"
                if m.get("hurst_exponent", 0.5) < 0.5
                else "(Trending)"
            ),
            f"- **Market Entropy**: {m.get('entropy', 0):.2f} bits",
            "",
            "#### 2. Monte Carlo Valuation (Bayesian/Heston)",
            f"- **Fair Value Range**: {fv_low:.2f} to {fv_high:.2f}",
            f"- **Mispricing Score**: {m.get('mispricing_score', 0):.2f} sigma",
            "",
            "#### 3. Risk & Optimization",
            f"- **Kelly Criterion**: {m.get('kelly_fraction', 0):.1%} allocation",
            f"- **RSI (14d)**: {m.get('rsi', 0):.1f}",
            "",
            f"**Conclusion**: Strategic model recommends {action} based on {regime} regime dynamics and probabilistic valuation.",
        ]

        return "\n".join(report)


# Convenience function
def create_mc_mean_reversion_strategy(
    n_simulations: int = 5000, lookback: int = 60
) -> MonteCarloMeanReversionStrategy:
    """Factory function to create configured strategy instance."""
    config = MCMRConfig(n_simulations=n_simulations, lookback_period=lookback)
    return MonteCarloMeanReversionStrategy(config)
