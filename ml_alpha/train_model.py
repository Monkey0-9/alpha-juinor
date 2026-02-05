"""
Train Predictive Model
Trains a LightGBM classifier to predict short-term price movements (+1% return in 24h).
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed. Install with: pip install lightgbm")
    exit(1)
import logging
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_training_features(df):
    """
    Vectorized feature calculation for training (calculates features for ALL rows).
    This mirrors logic in ml_feature_engineer.py but for full history.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # 1. OFI (Rolling 3 periods)
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        ofi = (df["buy_volume"] - df["sell_volume"]) / df["volume"].replace(0, 1)
        df["feature_ofi"] = ofi.rolling(3).mean()
    else:
        df["feature_ofi"] = 0.0

    # 2. VWAP Deviation
    cum_pv = (df["close"] * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    df["vwap"] = cum_pv / cum_vol.replace(0, 1)
    df["feature_vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, 1)

    # 3. Liquidity Hunt (Simplified vector)
    # Replicating "wick > body * 2" and "vol > avg * 1.5" logic
    body = (df["close"] - df["open"]).abs()
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    avg_vol = df["volume"].rolling(20).mean()

    # Boolean conditions converted to float
    cond_wick = (lower_wick > body * 2).astype(float)
    cond_vol = (df["volume"] > avg_vol * 1.5).astype(float)
    cond_bullish = (df["close"] > df["open"]).astype(float)

    df["feature_liquidity_hunt_score"] = (
        (cond_wick * 0.4) + (cond_vol * 0.4) + (cond_bullish * 0.2)
    )

    return df.dropna()


def create_target(df, horizon_hours=24, threshold=0.01):
    """Target: 1 if future return > threshold, else 0"""
    # Assuming hourly data? Adjust shift based on data frequency.
    # If data is minute, 24h is 1440 rows. If daily, 1 row.
    # We'll assume hourly for now as default (User said "next 24 hours").
    # Ideally should detect frequency.
    future_close = df["close"].shift(-horizon_hours)
    ret = (future_close - df["close"]) / df["close"]
    df["target"] = (ret > threshold).astype(int)
    return df.dropna()


def train(data_path, model_output, test_size=0.15):
    logger.info(f"Starting pipeline. Loading data from {data_path}")

    # Mock Data Generation if path empty or file not found
    if not os.path.exists(data_path) or not os.listdir(data_path):
        logger.warning("Data path invalid or empty. PROCESS SIMULATION MODE.")
        # Create synthetic data for verification of pipeline
        dates = pd.date_range("2024-01-01", periods=5000, freq="h")

        df = pd.DataFrame(
            {
                "open": 100 + np.random.randn(5000).cumsum(),
                "volume": np.random.randint(100, 10000, 5000),
                "buy_volume": np.random.randint(50, 5000, 5000),
                "sell_volume": np.random.randint(50, 5000, 5000),
            }
        )
        df["high"] = df["open"] + np.random.rand(5000)
        df["low"] = df["open"] - np.random.rand(5000)
        df["close"] = df["open"] + np.random.randn(5000) * 0.5
        dfs = [df]
    else:
        # Load real data
        pass  # (Implementation skipped for brevity in this step, focusing on pipeline structure)
        dfs = []  # Placeholder

    # Process
    processed_frames = []
    for df in dfs:
        df_feat = calculate_training_features(df)
        df_target = create_target(df_feat, horizon_hours=24)
        processed_frames.append(df_target)

    if not processed_frames:
        return

    full_df = pd.concat(processed_frames)

    features = ["feature_ofi", "feature_vwap_deviation", "feature_liquidity_hunt_score"]
    X = full_df[features]
    y = full_df["target"]

    # Split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Training on {len(X_train)} samples, Testing on {len(X_test)}")

    # Train
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05)
    clf.fit(X_train, y_train)

    # Eval
    score = clf.score(X_test, y_test)
    logger.info(f"Model Accuracy: {score:.4f}")

    # Save
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    with open(model_output, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved to {model_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/feature_store/")
    parser.add_argument("--model_output", default="./models/lgbm_predictor_v1.pkl")
    parser.add_argument("--test_size", type=float, default=0.15)
    args = parser.parse_args()

    train(args.data_path, args.model_output, args.test_size)
