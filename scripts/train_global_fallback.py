"""
scripts/train_global_fallback.py
Train a global fallback ML model on all available high-quality data.
The resulting model is saved to ml/models/return_model.pkl with feature_names_in_.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from database.manager import DatabaseManager
from data.processors.features import compute_features_for_symbol

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    db = DatabaseManager()
    model_dir = Path("ml/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    symbols = db.get_active_symbols()
    logger.info(f"Loading data for {len(symbols)} symbols...")

    all_X = []
    all_y = []

    for symbol in symbols:
        try:
            df = db.get_daily_prices(symbol)
            if df.empty or len(df) < 500:
                continue

            # Canonical column mapping
            df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            }, inplace=True)

            # Target: next-day return
            df['target'] = df['Close'].pct_change(fill_method=None).shift(-1)
            df.dropna(subset=['target'], inplace=True)

            # Compute features
            X = compute_features_for_symbol(df, contract_name="ml_v1")
            if X is None or X.empty:
                continue

            # Align
            common_idx = X.index.intersection(df.index)
            X = X.loc[common_idx]
            y = df.loc[common_idx, 'target']

            # Drop NaN
            valid = ~y.isna() & ~X.isna().any(axis=1)
            X = X[valid]
            y = y[valid]

            if len(X) > 100:
                all_X.append(X)
                all_y.append(y)
                logger.info(f"  {symbol}: {len(X)} samples")

        except Exception as e:
            logger.warning(f"  {symbol}: Failed - {e}")

    if not all_X:
        logger.error("No training data collected. Aborting.")
        return

    # Combine all data
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)

    logger.info(f"Total training samples: {len(X_combined)}")
    logger.info(f"Features: {list(X_combined.columns)}")

    # Train GradientBoosting (same as legacy)
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_combined, y_combined)

    # CRITICAL: Attach feature names for alignment
    model.feature_names_in_ = np.array(X_combined.columns)

    # Save
    model_path = model_dir / "return_model.pkl"
    joblib.dump(model, model_path)

    logger.info(f"✓ Global fallback model saved to {model_path}")
    logger.info(f"  Features: {len(model.feature_names_in_)}")
    logger.info(f"  feature_names_in_: {list(model.feature_names_in_)}")

if __name__ == "__main__":
    main()
