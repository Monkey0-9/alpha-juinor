"""
Elite Alpha Generator - 2026 Peak Intelligence
===============================================

Generates high-quality alpha signals from multiple sources.

Alpha Sources:
1. Statistical Arbitrage (pairs, mean reversion)
2. Momentum Factors (cross-sectional, time-series)
3. Fundamental Value (earnings, quality)
4. Alternative Data (sentiment, flow)
5. ML Patterns (learned alpha)

Target: Information Ratio > 1.5
"""

import logging
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AlphaSignal:
    """High-quality alpha signal."""
    symbol: str
    alpha_value: float  # Z-score normalized
    alpha_type: str
    confidence: float
    decay_half_life: int  # Days
    icr: float  # Information Coefficient Ratio


class EliteAlphaGenerator:
    """
    Elite alpha generation engine.

    Combines multiple alpha sources with dynamic weighting.
    """

    def __init__(self):
        # Alpha family weights (learned from IC)
        self.alpha_weights = {
            "momentum_12m_1m": 0.15,
            "momentum_vol_adj": 0.12,
            "mean_reversion_5d": 0.10,
            "quality_score": 0.08,
            "sentiment_composite": 0.12,
            "flow_signal": 0.10,
            "ml_pattern": 0.18,
            "regime_alpha": 0.15
        }

        # IC tracking for each alpha
        self.ic_history = {k: [] for k in self.alpha_weights}

        logger.info("[ALPHA_GEN] Elite generator initialized")

    def generate_alpha(
        self,
        symbol: str,
        features: Dict[str, float],
        returns_1m: float,
        returns_12m: float,
        volatility: float,
        regime: str
    ) -> AlphaSignal:
        """
        Generate composite alpha signal.
        """
        alphas = {}

        # 1. 12-1 Momentum (skip last month)
        mom_12_1 = returns_12m - returns_1m
        alphas["momentum_12m_1m"] = self._zscore(mom_12_1, 0.15)

        # 2. Vol-adjusted momentum
        if volatility > 0:
            vol_adj_mom = returns_1m / volatility
        else:
            vol_adj_mom = 0
        alphas["momentum_vol_adj"] = self._zscore(vol_adj_mom, 1.0)

        # 3. Mean reversion (5-day)
        ret_5d = features.get("return_5d", 0)
        mr_alpha = -ret_5d  # Negative of recent returns
        alphas["mean_reversion_5d"] = self._zscore(mr_alpha, 0.05)

        # 4. Quality score (from features)
        roe = features.get("roe", 0.15)
        debt_ratio = features.get("debt_ratio", 0.5)
        quality = roe - debt_ratio * 0.5
        alphas["quality_score"] = self._zscore(quality, 0.2)

        # 5. Sentiment composite
        news_sent = features.get("news_sentiment", 0)
        social_sent = features.get("social_sentiment", 0)
        sentiment = 0.6 * news_sent + 0.4 * social_sent
        alphas["sentiment_composite"] = sentiment

        # 6. Flow signal
        etf_flow = features.get("etf_flow", 0)
        inst_flow = features.get("institutional_flow", 0)
        flow = 0.5 * etf_flow + 0.5 * inst_flow
        alphas["flow_signal"] = self._zscore(flow, 0.1)

        # 7. ML pattern
        ml_pred = features.get("ml_prediction", 0)
        alphas["ml_pattern"] = ml_pred

        # 8. Regime alpha
        regime_alpha = self._regime_alpha(regime, returns_1m)
        alphas["regime_alpha"] = regime_alpha

        # Combine with weights
        composite = 0.0
        for alpha_name, alpha_val in alphas.items():
            weight = self.alpha_weights.get(alpha_name, 0.1)
            composite += weight * alpha_val

        # Calculate confidence
        confidence = self._calculate_confidence(alphas, regime)

        # Determine primary alpha type
        max_contrib = max(alphas.items(), key=lambda x: abs(x[1] * self.alpha_weights.get(x[0], 0.1)))

        return AlphaSignal(
            symbol=symbol,
            alpha_value=np.clip(composite, -3, 3),
            alpha_type=max_contrib[0],
            confidence=confidence,
            decay_half_life=5,
            icr=0.08  # Target IC
        )

    def _zscore(self, value: float, typical_std: float) -> float:
        """Convert to z-score."""
        if typical_std > 0:
            return value / typical_std
        return 0.0

    def _regime_alpha(self, regime: str, recent_return: float) -> float:
        """Generate regime-conditional alpha."""
        if regime == "BULL":
            # Favor momentum
            return recent_return * 0.5
        elif regime == "BEAR":
            # Favor defensives (negative correlation with recent)
            return -recent_return * 0.3
        elif regime == "VOLATILE":
            # No strong signal
            return 0.0
        else:
            return recent_return * 0.2

    def _calculate_confidence(
        self,
        alphas: Dict[str, float],
        regime: str
    ) -> float:
        """Calculate signal confidence."""
        # Agreement of alpha sources
        signs = [np.sign(a) for a in alphas.values()]
        agreement = abs(sum(signs)) / len(signs)

        # Regime penalty
        regime_factor = {
            "VOLATILE": 0.7,
            "CRISIS": 0.5,
            "NORMAL": 1.0,
            "BULL": 1.1,
            "BEAR": 0.9
        }.get(regime, 1.0)

        return np.clip(agreement * regime_factor, 0.3, 0.95)

    def update_ic(self, alpha_name: str, ic_value: float):
        """Update IC history for an alpha."""
        if alpha_name in self.ic_history:
            self.ic_history[alpha_name].append(ic_value)
            if len(self.ic_history[alpha_name]) > 252:
                self.ic_history[alpha_name].pop(0)

            # Update weight based on IC
            avg_ic = np.mean(self.ic_history[alpha_name])
            self.alpha_weights[alpha_name] = max(0.02, min(0.30, avg_ic * 3))

    def get_portfolio_alphas(
        self,
        symbols: List[str],
        features_map: Dict[str, Dict],
        returns_map: Dict[str, Dict],
        regime: str,
        top_n: int = 20
    ) -> Dict[str, AlphaSignal]:
        """
        Generate alphas for all symbols and return top N.
        """
        alphas = {}

        for symbol in symbols:
            features = features_map.get(symbol, {})
            returns = returns_map.get(symbol, {})

            ret_1m = returns.get("1m", 0)
            ret_12m = returns.get("12m", 0)
            vol = features.get("volatility", 0.02)

            alpha = self.generate_alpha(
                symbol, features, ret_1m, ret_12m, vol, regime
            )
            alphas[symbol] = alpha

        # Sort by alpha value * confidence
        sorted_symbols = sorted(
            alphas.keys(),
            key=lambda s: abs(alphas[s].alpha_value) * alphas[s].confidence,
            reverse=True
        )[:top_n]

        return {s: alphas[s] for s in sorted_symbols}


# Singleton
_alpha_gen = None


def get_alpha_generator() -> EliteAlphaGenerator:
    global _alpha_gen
    if _alpha_gen is None:
        _alpha_gen = EliteAlphaGenerator()
    return _alpha_gen
