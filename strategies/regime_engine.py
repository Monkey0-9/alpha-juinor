"""
Regime Detection Engine for Institutional Trading.

Classifies market regimes: Risk-On/Risk-Off, Volatility Expansion/Compression,
Liquidity Stress/Abundance, Trend vs Mean-Reversion.

Outputs: regime_tag, regime_confidence, vol_target_multiplier.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from enum import Enum


class RegimeTag(Enum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    VOL_EXPANSION = "VOL_EXPANSION"
    VOL_COMPRESSION = "VOL_COMPRESSION"
    LIQUIDITY_STRESS = "LIQUIDITY_STRESS"
    LIQUIDITY_ABUNDANCE = "LIQUIDITY_ABUNDANCE"
    TREND = "TREND"
    MEAN_REVERSION = "MEAN_REVERSION"


class RegimeEngine:
    """
    Multi-dimensional regime detection for institutional alpha generation.
    """

    def __init__(self):
        self.vol_threshold = 1.5  # Volatility expansion threshold
        self.liquidity_threshold = 0.8  # Liquidity stress threshold
        self.trend_threshold = 0.6  # Trend strength threshold

    def detect_regime(self, market_data: pd.DataFrame, macro_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect multi-dimensional market regime.
        """
        # ---- PASSIVE SAFETY NET ----
        if isinstance(market_data, pd.Series):
            market_data = market_data.to_frame(name="Close")
        # -----------------------------

        # Extract market data (assume SPY as proxy for broad market)
        if isinstance(market_data.columns, pd.MultiIndex) and 'SPY' in market_data.columns.get_level_values(0):
            data = market_data['SPY'].dropna()
        elif 'SPY' in market_data.columns:
            data = market_data['SPY'].dropna()
        else:
            # Use first available ticker or the input itself if it's a single-ticker OHLCV
            if not isinstance(market_data.columns, pd.MultiIndex) and 'Close' in market_data.columns:
                data = market_data
            elif isinstance(market_data.columns, pd.MultiIndex):
                ticker = market_data.columns.get_level_values(0)[0]
                data = market_data[ticker].dropna()
            else:
                data = market_data.iloc[:, 0].dropna()

        if len(data) < 60:
            return {
                'regime_tag': RegimeTag.RISK_OFF.value,
                'regime_confidence': 0.5,
                'vol_target_multiplier': 0.5
            }

        # 1. Risk-On/Risk-Off (based on VIX and trend)
        risk_regime = self._detect_risk_regime(data, macro_context)

        # 2. Volatility Expansion/Compression
        vol_regime = self._detect_vol_regime(data)

        # 3. Liquidity Stress/Abundance
        liq_regime = self._detect_liquidity_regime(data)

        # 4. Trend vs Mean-Reversion
        trend_regime = self._detect_trend_regime(data)

        # Combine into primary regime (weighted voting)
        regimes = [risk_regime, vol_regime, liq_regime, trend_regime]
        regime_scores = {}
        for r in regimes:
            regime_scores[r['tag']] = regime_scores.get(r['tag'], 0) + r['confidence']

        primary_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[primary_regime] / len(regimes)

        # Vol target multiplier based on regime
        vol_multiplier = self._get_vol_multiplier(primary_regime)

        return {
            'regime_tag': primary_regime,
            'regime_confidence': confidence,
            'vol_target_multiplier': vol_multiplier
        }

    def _detect_risk_regime(self, data: pd.DataFrame, macro_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Detect Risk-On/Risk-Off regime."""
        # Use VIX if available
        vix = macro_context.get('VIX', 20.0) if macro_context else 20.0

        # Trend strength
        # Trend strength
        col = 'Close' if 'Close' in data.columns else data.columns[0]
        returns = data[col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) < 20:
             return {'tag': RegimeTag.RISK_OFF.value, 'confidence': 0.5}

        trend_strength = abs(returns.rolling(20).mean()) / returns.rolling(20).std()
        ts_last = trend_strength.iloc[-1] if not trend_strength.empty else 0.0

        if vix < 20 and ts_last > 0.5:
            return {'tag': RegimeTag.RISK_ON.value, 'confidence': 0.8}
        elif vix > 30 or ts_last < 0.2:
            return {'tag': RegimeTag.RISK_OFF.value, 'confidence': 0.9}
        else:
            return {'tag': RegimeTag.RISK_ON.value, 'confidence': 0.5}

    def _detect_vol_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Volatility Expansion/Compression."""
        col = 'Close' if 'Close' in data.columns else data.columns[0]
        returns = data[col].pct_change(fill_method=None).dropna()
        short_vol = returns.tail(10).std() * np.sqrt(252) if len(returns) >= 10 else 0.02
        long_vol = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else 0.02

        if long_vol == 0:
            ratio = 1.0
        else:
            ratio = short_vol / long_vol

        if ratio > self.vol_threshold:
            return {'tag': RegimeTag.VOL_EXPANSION.value, 'confidence': min(ratio / self.vol_threshold, 1.0)}
        else:
            return {'tag': RegimeTag.VOL_COMPRESSION.value, 'confidence': 1.0 - min(ratio / self.vol_threshold, 1.0)}

    def _detect_liquidity_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Liquidity Stress/Abundance."""
        # Use volume relative to moving average
        if 'Volume' not in data.columns:
            return {'tag': RegimeTag.LIQUIDITY_ABUNDANCE.value, 'confidence': 0.5}

        volume_ma = data['Volume'].rolling(20).mean()
        volume_ratio = data['Volume'] / volume_ma

        avg_ratio = volume_ratio.tail(5).mean() if not volume_ratio.empty else 1.0

        if avg_ratio < self.liquidity_threshold:
            return {'tag': RegimeTag.LIQUIDITY_STRESS.value, 'confidence': 1.0 - avg_ratio}
        else:
            return {'tag': RegimeTag.LIQUIDITY_ABUNDANCE.value, 'confidence': avg_ratio}

    def _detect_trend_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Trend vs Mean-Reversion."""
        # Use Hurst exponent approximation
        returns = data['Close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 100:
            return {'tag': RegimeTag.MEAN_REVERSION.value, 'confidence': 0.5}

        # Simple trend strength measure
        col = 'Close' if 'Close' in data.columns else data.columns[0]
        ma_short = data[col].rolling(10).mean()
        ma_long = data[col].rolling(50).mean()

        if ma_long.empty or pd.isna(ma_long.iloc[-1]):
             return {'tag': RegimeTag.MEAN_REVERSION.value, 'confidence': 0.5}

        trend_strength = abs(ma_short - ma_long).iloc[-1] / data[col].iloc[-1]

        if trend_strength > self.trend_threshold:
            return {'tag': RegimeTag.TREND.value, 'confidence': trend_strength}
        else:
            return {'tag': RegimeTag.MEAN_REVERSION.value, 'confidence': 1.0 - trend_strength}

    def _get_vol_multiplier(self, regime: str) -> float:
        """Get volatility target multiplier for regime."""
        multipliers = {
            RegimeTag.RISK_ON.value: 1.0,
            RegimeTag.RISK_OFF.value: 0.5,
            RegimeTag.VOL_EXPANSION.value: 0.7,
            RegimeTag.VOL_COMPRESSION.value: 1.2,
            RegimeTag.LIQUIDITY_STRESS.value: 0.6,
            RegimeTag.LIQUIDITY_ABUNDANCE.value: 1.1,
            RegimeTag.TREND.value: 1.0,
            RegimeTag.MEAN_REVERSION.value: 0.8
        }
        return multipliers.get(regime, 1.0)
