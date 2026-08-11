import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Standardized pipeline for calculating technical indicators,
    order book imbalances, and statistical factors.
    """

    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adds RSI, MACD, and Rate of Change."""
        df = df.copy()

        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ROC
        df['roc_10'] = df['close'].pct_change(periods=10)

        return df

    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adds Bollinger Bands, ATR, and historical volatility."""
        df = df.copy()

        # Bollinger Bands
        df['bb_mean'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mean'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_mean'] - (df['bb_std'] * 2)

        # ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

        return df

    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        df = FeatureEngineer.add_momentum_features(df)
        df = FeatureEngineer.add_volatility_features(df)
        df.dropna(inplace=True)
        return df
