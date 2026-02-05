"""
ML Feature Engineer
Calculates predictive features from raw market data, focusing on Smart Money Concepts (SMC).
"""
import numpy as np
import pandas as pd


def calculate_smc_features(ticker_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Smart Money Concept (SMC) features for ML prediction.
    Features:
    1. Order Flow Imbalance (OFI)
    2. VWAP Deviation
    3. Liquidity Hunt Score (Rejection/Wick analysis)

    Args:
        ticker_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume']
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = ticker_data.copy()

    # Ensure standard lowercase columns
    df.columns = [c.lower() for c in df.columns]

    features = {}

    # Validate required columns
    required_cols = ['volume', 'close', 'high', 'low']
    # buy/sell volume might be missing if source doesn't provide it
    has_order_flow = 'buy_volume' in df.columns and 'sell_volume' in df.columns

    if not all(col in df.columns for col in required_cols):
        # Return empty with correct columns if data is missing, to prevent pipeline crashes
        return pd.DataFrame([{
            'ofi': 0.0,
            'vwap_deviation': 0.0,
            'liquidity_hunt_score': 0.0
        }])

    # -----------------------------------------------------
    # 1. Order Flow Imbalance (Smart Money Proxy)
    # -----------------------------------------------------
    if has_order_flow:
        # Avoid division by zero
        vol_safe = df['volume'].replace(0, 1)
        # We take the rolling average OFI of the last 3 periods to smooth noise
        ofi_series = (df['buy_volume'] - df['sell_volume']) / vol_safe
        features['ofi'] = ofi_series.iloc[-3:].mean()
    else:
        features['ofi'] = 0.0

    # -----------------------------------------------------
    # 2. Volume-Weighted Price Deviation
    # -----------------------------------------------------
    # Calculate VWAP for the loaded window
    # Formula: Cumulative(Price * Volume) / Cumulative(Volume)
    cum_pv = (df['close'] * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum()

    # Handle zero volume start
    vwap_series = cum_pv / cum_vol.replace(0, 1)

    current_close = df['close'].iloc[-1]
    current_vwap = vwap_series.iloc[-1]

    if current_vwap != 0:
        features['vwap_deviation'] = (current_close - current_vwap) / current_vwap
    else:
        features['vwap_deviation'] = 0.0

    # -----------------------------------------------------
    # 3. Liquidity Hunt Signal (Simplified)
    # -----------------------------------------------------
    # Logic: Look for price dipping below recent support (lows) and then rejecting (wick).
    # We analyze the last candle.
    last_candle = df.iloc[-1]

    # Need at least 20 rows for meaningful averages
    if len(df) > 20:
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    else:
        avg_vol = df['volume'].mean()

    # Calculate wick metrics
    body_size = abs(last_candle['close'] - last_candle['open'])
    total_range = last_candle['high'] - last_candle['low']
    lower_wick = min(last_candle['close'], last_candle['open']) - last_candle['low']

    hunt_score = 0.0

    # Condition A: Long lower wick (Rejection of lower prices) => 'Stop Hunt'
    if total_range > 0 and (lower_wick / total_range) > 0.5:
        hunt_score += 0.4

    # Condition B: High Volume on Rejection (Smart Money Absorption)
    if avg_vol > 0 and last_candle['volume'] > (avg_vol * 1.5):
        hunt_score += 0.4

    # Condition C: Close > Open (Bullish Rejection)
    if last_candle['close'] > last_candle['open']:
        hunt_score += 0.2

    features['liquidity_hunt_score'] = hunt_score

    return pd.DataFrame([features])
