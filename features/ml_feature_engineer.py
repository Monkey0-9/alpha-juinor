"""
ML Feature Engineer
Calculates predictive features from raw market data, focusing on Smart Money Concepts (SMC).
"""

import numpy as np
import pandas as pd


def calculate_smc_features(
    ticker_data: pd.DataFrame, return_full_history: bool = False
) -> pd.DataFrame:
    """
    Calculates predictive features from raw market data.

    Args:
        ticker_data: DataFrame with OHLCV data.
        return_full_history: If True, returns features for all rows.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = ticker_data.copy()

    # Ensure standard lowercase columns
    df.columns = [c.lower() for c in df.columns]

    features = {}

    # Validate required columns
    required_cols = ["volume", "close", "high", "low"]
    # buy/sell volume might be missing if source doesn't provide it
    has_order_flow = "buy_volume" in df.columns and "sell_volume" in df.columns

    if not all(col in df.columns for col in required_cols):
        # Return empty with correct columns if data is missing, to prevent pipeline crashes
        return pd.DataFrame(
            [{"ofi": 0.0, "vwap_deviation": 0.0, "liquidity_hunt_score": 0.0}]
        )

    # -----------------------------------------------------
    # 1. Order Flow Imbalance (Smart Money Proxy)
    # -----------------------------------------------------
    if has_order_flow:
        # Avoid division by zero
        vol_safe = df["volume"].replace(0, 1)
        # We take the rolling average OFI of the last 3 periods to smooth noise
        ofi_series = (df["buy_volume"] - df["sell_volume"]) / vol_safe
        df["ofi"] = ofi_series.rolling(3).mean()
    else:
        df["ofi"] = 0.0

    # -----------------------------------------------------
    # 2. Volume-Weighted Price Deviation
    # -----------------------------------------------------
    # -----------------------------------------------------
    # 2. Volume-Weighted Price Deviation
    # -----------------------------------------------------
    cum_pv = (df["close"] * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    df["vwap"] = cum_pv / cum_vol.replace(0, 1)

    # Vectorized Deviation
    df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, 1)

    # -----------------------------------------------------
    # 3. Liquidity Hunt Signal (Vectorized)
    # -----------------------------------------------------
    body_size = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]

    # Lower wick size: min(open, close) - low
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    # 20-period avg volume
    avg_vol = df["volume"].rolling(20).mean()

    # Condition A: Long lower wick (> 50% of range)
    cond_wick = (total_range > 0) & ((lower_wick / total_range) > 0.5)

    # Condition B: High Volume (> 1.5x avg)
    cond_vol = df["volume"] > (avg_vol * 1.5)

    # Condition C: Bullish Close
    cond_bullish = df["close"] > df["open"]

    # Combined Score (Vectorized sum)
    df["liquidity_hunt_score"] = (
        (cond_wick.astype(float) * 0.4)
        + (cond_vol.astype(float) * 0.4)
        + (cond_bullish.astype(float) * 0.2)
    )

    # Fill NaNs (start of rolling windows)
    df["liquidity_hunt_score"] = df["liquidity_hunt_score"].fillna(0.0)
    df["ofi"] = df.get("ofi", pd.Series(0, index=df.index)).fillna(0.0)
    df["vwap_deviation"] = df["vwap_deviation"].fillna(0.0)

    # Return only the requested features
    feature_cols = ["ofi", "vwap_deviation", "liquidity_hunt_score"]

    # Return logic
    if return_full_history:
        return df[feature_cols]
    else:
        # Inference mode: Return last row as DataFrame (preserving col names)
        return df[feature_cols].iloc[[-1]]
