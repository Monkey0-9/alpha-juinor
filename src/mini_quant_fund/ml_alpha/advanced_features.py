"""
Advanced Feature Engineering System
Generates sophisticated market microstructure and technical features
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Generate advanced features for ML models."""

    def __init__(self):
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.computed_features: List[str] = []

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all available features."""
        features = pd.DataFrame(index=df.index)

        # Basic OHLCV features
        features = pd.concat([features, self._compute_ohlcv_features(df)], axis=1)

        # Volatility features
        features = pd.concat([features, self._compute_volatility_features(df)], axis=1)

        # Trend features
        features = pd.concat([features, self._compute_trend_features(df)], axis=1)

        # Momentum features
        features = pd.concat([features, self._compute_momentum_features(df)], axis=1)

        # Mean reversion features
        features = pd.concat(
            [features, self._compute_mean_reversion_features(df)], axis=1
        )

        # Microstructure features
        features = pd.concat(
            [features, self._compute_microstructure_features(df)], axis=1
        )

        # Statistical features
        features = pd.concat([features, self._compute_statistical_features(df)], axis=1)

        # Regime features
        features = pd.concat([features, self._compute_regime_features(df)], axis=1)

        # Drop NaN rows
        features = features.dropna()
        self.feature_cache["all_features"] = features

        logger.info(f"[FEATURES] Generated {features.shape[1]} features")
        return features

    def _compute_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic OHLCV features."""
        features = pd.DataFrame(index=df.index)

        features["returns"] = df["close"].pct_change()
        features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        features["hl_ratio"] = df["high"] / (df["low"] + 1e-8)
        features["oc_ratio"] = df["close"] / (df["open"] + 1e-8)
        features["cc_ratio"] = df["close"] / df["close"].shift(1)
        features["volume_sma_20"] = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / (
            df["volume"].rolling(20).mean() + 1e-8
        )

        return features

    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility and variance features."""
        features = pd.DataFrame(index=df.index)

        returns = df["close"].pct_change()

        features["volatility_20"] = returns.rolling(20).std()
        features["volatility_60"] = returns.rolling(60).std()
        features["volatility_ratio"] = features["volatility_20"] / (
            features["volatility_60"] + 1e-8
        )

        features["parkinson_vol"] = self._parkinson_volatility(df, window=20)
        features["garman_klass_vol"] = self._garman_klass_volatility(df, window=20)

        features["high_low_volatility"] = (df["high"] - df["low"]).rolling(20).std()

        return features

    def _compute_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend and direction features."""
        features = pd.DataFrame(index=df.index)

        close = df["close"]

        # Moving averages
        features["sma_10"] = close.rolling(10).mean()
        features["sma_20"] = close.rolling(20).mean()
        features["sma_50"] = close.rolling(50).mean()
        features["sma_200"] = close.rolling(200).mean()

        features["ema_12"] = close.ewm(span=12).mean()
        features["ema_26"] = close.ewm(span=26).mean()

        # Trend indicators
        features["price_trend_20"] = (close - close.rolling(20).min()) / (
            close.rolling(20).max() - close.rolling(20).min() + 1e-8
        )
        features["price_trend_50"] = (close - close.rolling(50).min()) / (
            close.rolling(50).max() - close.rolling(50).min() + 1e-8
        )

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features["macd"] = ema12 - ema26
        features["macd_signal"] = features["macd"].ewm(span=9).mean()
        features["macd_hist"] = features["macd"] - features["macd_signal"]

        return features

    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum features."""
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        returns = close.pct_change()

        # RSI
        features["rsi_14"] = self._calculate_rsi(close, 14)
        features["rsi_20"] = self._calculate_rsi(close, 20)

        # Rate of Change
        features["roc_10"] = close.pct_change(10)
        features["roc_20"] = close.pct_change(20)

        # Momentum
        features["momentum_10"] = close - close.shift(10)
        features["momentum_20"] = close - close.shift(20)

        # Stochastic
        features["stoch_k"], features["stoch_d"] = self._calculate_stochastic(df, 14)

        return features

    def _compute_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean reversion features."""
        features = pd.DataFrame(index=df.index)

        close = df["close"]

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features["bb_upper"] = sma20 + (std20 * 2)
        features["bb_lower"] = sma20 - (std20 * 2)
        features["bb_position"] = (close - features["bb_lower"]) / (
            features["bb_upper"] - features["bb_lower"] + 1e-8
        )
        features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / (
            sma20 + 1e-8
        )

        # Keltner Channels
        features["atr"] = self._calculate_atr(df, 14)
        features["kc_position"] = self._keltner_channel_position(df, close)

        # Distance from SMA
        features["dist_sma_20"] = (close - close.rolling(20).mean()) / (
            close.rolling(20).std() + 1e-8
        )
        features["dist_sma_50"] = (close - close.rolling(50).mean()) / (
            close.rolling(50).std() + 1e-8
        )

        return features

    def _compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features."""
        features = pd.DataFrame(index=df.index)

        # Volume profile
        features["volume_trend"] = df["volume"].rolling(20).mean() / (
            df["volume"].rolling(60).mean() + 1e-8
        )

        # Price-Volume correlation
        returns = df["close"].pct_change()
        vol_changes = df["volume"].pct_change()
        features["pv_correlation"] = returns.rolling(20).corr(vol_changes)

        # High-Low Range
        features["hl_range"] = (df["high"] - df["low"]) / df["close"]
        features["hl_range_ma"] = features["hl_range"].rolling(20).mean()

        # Volume-weighted features
        vwap_num = (df["close"] * df["volume"]).rolling(20).sum()
        vwap_den = df["volume"].rolling(20).sum()
        features["vwap"] = vwap_num / (vwap_den + 1e-8)
        features["price_vwap_ratio"] = df["close"] / (features["vwap"] + 1e-8)

        # Order imbalance proxy
        features["volume_ma_ratio"] = df["volume"].rolling(5).mean() / (
            df["volume"].rolling(20).mean() + 1e-8
        )

        return features

    def _compute_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features."""
        features = pd.DataFrame(index=df.index)

        returns = df["close"].pct_change()

        # Skewness and Kurtosis
        features["returns_skew_20"] = returns.rolling(20).skew()
        features["returns_kurt_20"] = returns.rolling(20).kurt()

        # Autocorrelation
        features["returns_acf_1"] = returns.rolling(20).apply(
            lambda x: x.autocorr(1) if len(x) > 1 else 0
        )

        # Distribution features
        features["returns_mean_20"] = returns.rolling(20).mean()
        features["returns_std_20"] = returns.rolling(20).std()

        return features

    def _compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime features."""
        features = pd.DataFrame(index=df.index)

        returns = df["close"].pct_change()
        volatility = returns.rolling(30).std()

        # Regime indicators
        features["high_vol_regime"] = (
            volatility > volatility.rolling(60).quantile(0.75)
        ).astype(int)
        features["low_vol_regime"] = (
            volatility < volatility.rolling(60).quantile(0.25)
        ).astype(int)
        features["trending_regime"] = (
            abs(df["close"].rolling(20).mean() - df["close"].rolling(50).mean())
            / df["close"]
            > 0.02
        ).astype(int)

        return features

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_stochastic(
        df: pd.DataFrame, period: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic indicator."""
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-8)
        d = k.rolling(3).mean()
        return k, d

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Parkinson volatility."""
        hl_ratio = np.log(df["high"] / df["low"])
        return (1 / (4 * np.log(2))) * (hl_ratio**2).rolling(window).mean() ** 0.5

    @staticmethod
    def _garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility."""
        hl = np.log(df["high"] / df["low"])
        cc = np.log(df["close"] / df["close"].shift(1))

        gk = 0.5 * (hl**2) - (2 * np.log(2) - 1) * (cc**2)
        return gk.rolling(window).mean() ** 0.5

    @staticmethod
    def _keltner_channel_position(df: pd.DataFrame, close: pd.Series) -> pd.Series:
        """Calculate position within Keltner Channel."""
        sma = close.rolling(20).mean()
        atr = AdvancedFeatureEngineer._calculate_atr(df, 14)
        top = sma + (2 * atr)
        bottom = sma - (2 * atr)
        return (close - bottom) / (top - bottom + 1e-8)


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to create all advanced features."""
    engineer = AdvancedFeatureEngineer()
    return engineer.compute_all_features(df)
