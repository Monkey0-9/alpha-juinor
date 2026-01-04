# strategies/features.py
import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureEngineer:
    """
    Generates technical features for ML models.
    Scales features to be suitable for tree-based models (scaling not strictly required but helpful).
    """

    def __init__(self, use_technical: bool = True, use_lags: bool = True):
        self.use_technical = use_technical
        self.use_lags = use_lags

    def compute_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Expects a DataFrame with columns: Open, High, Low, Close, Volume.
        Returns a DataFrame of features (X).
        """
        df = prices.copy()
        features = pd.DataFrame(index=df.index)

        # 1. Base Returns (Target proxy)
        # We predict forward returns, but for features we use past returns
        features["ret_1d"] = df["Close"].pct_change()
        features["ret_5d"] = df["Close"].pct_change(5)
        features["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

        # 2. Volatility
        features["vol_21d"] = features["log_ret"].rolling(21).std() * np.sqrt(252)
        features["vol_63d"] = features["log_ret"].rolling(63).std() * np.sqrt(252)

        # 3. Momentum / Trend
        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        features["macd"] = ema_12 - ema_26
        features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
        features["macd_hist"] = features["macd"] - features["macd_signal"]

        # Bollinger Bands %B
        sma_20 = df["Close"].rolling(20).mean()
        std_20 = df["Close"].rolling(20).std()
        upper = sma_20 + 2 * std_20
        lower = sma_20 - 2 * std_20
        features["bb_width"] = (upper - lower) / sma_20
        features["bb_pct_b"] = (df["Close"] - lower) / (upper - lower).replace(0, np.nan)

        # 4. Volume
        # Relative Volume (Vol / Avg Vol)
        features["vol_rel_20"] = df["Volume"] / df["Volume"].rolling(20).mean()
        # Volume Z-Score (Institutional Liquidity Signal)
        features["vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std().replace(0, np.nan)

        # 5. Volatility-Adjusted Momentum (Institutional "Price Momentum / Vol")
        # Captures how 'clean' the trend is
        features["mom_vol_adj"] = features["ret_21d"] / (features["vol_21d"] + 1e-6) if "ret_21d" in features else (df["Close"].pct_change(21) / features["vol_21d"])

        # 6. Intraday Dynamics
        features["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
        features["hl_vol"] = features["high_low_range"].rolling(20).mean()

        # 5. Lagged Features (Critical for ML)
        if self.use_lags:
            cols_to_lag = ["ret_1d", "rsi_14", "vol_21d", "bb_pct_b"]
            for col in cols_to_lag:
                for lag in [1, 2, 3, 5]:
                    features[f"{col}_lag_{lag}"] = features[col].shift(lag)

        # Drop NaNs created by windows/lags
        # We don't drop here to allow alignment, but caller should handle it.
        return features

    def compute_target(self, prices: pd.DataFrame, forward_window: int = 1) -> pd.Series:
        """
        Compute forward returns as target variable.
        """
        # Close to Close return 'forward_window' days ahead
        # shift(-k) brings future value to current row
        target = prices["Close"].pct_change(forward_window).shift(-forward_window)
        return target
