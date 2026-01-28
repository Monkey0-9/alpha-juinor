"""
scripts/train_ml_alpha.py
Offline ML Training Pipeline for Institutional Quant Fund.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager
from alpha_families.ml_alpha import MLAlpha
from data.quality import validate_data_for_ml
from features.contract import load_feature_contract
import sklearn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MLTraining")

def get_git_commit_hash() -> str:
    """Get current git commit hash for model versioning."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def train_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    model_dir: str,
    feature_contract: Dict[str, Any],
    model_type: str = "huber",
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1
):
    """Train and save model for a single symbol."""
    logger.info(f"Starting training for {symbol} (model={model_type}, estimators={n_estimators})")

    if df.empty:
        logger.warning(f"No history found for {symbol}")
        return False

    df.sort_index(inplace=True)

    # 2. Institutional Validation
    is_ready, reasons = validate_data_for_ml(df, min_rows=1000, min_quality=0.6)
    if not is_ready:
        logger.warning(f"Symbol {symbol} rejected for ML training: {reasons}")
        return False

    # 3. Feature Extraction & Training using Contract
    try:
        # Prepare target (e.g., next day's return)
        df['target'] = df['Close'].pct_change(fill_method=None).shift(-1)
        # ONLY drop rows where target is missing at this stage
        df = df.dropna(subset=['target'])

        if len(df) < 252:
            logger.warning(f"Insufficient samples for training {symbol} after target cleanup: {len(df)}")
            return False


        # Extract features using contract (institutional spec)
        from data.processors.features import compute_features_for_symbol
        try:
            X = compute_features_for_symbol(df, contract_name="ml_v1")
        except Exception as e:
            logger.error(f"Feature computation failed for {symbol}: {e}")
            return False

        if X is None or X.empty:
            logger.warning(f"Feature extraction yielded empty X for {symbol}")
            return False

        # Align target with features
        # Ensure indices are compatible (datetime objects for intersection)
        if not isinstance(X.index, pd.DatetimeIndex):
            X.index = pd.to_datetime(X.index)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        common_idx = X.index.intersection(df.index)
        logger.info(f"Alignment: X={len(X)}, df={len(df)}, intersection={len(common_idx)}")

        X = X.loc[common_idx]
        y = df.loc[common_idx, 'target']

        # Drop any remaining NaNs in target or features
        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 100:
            logger.warning(f"Insufficient samples for training {symbol} after cleanup: {len(X)}")
            return False

        # Model Selection
        if model_type == "gbr":
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
        elif model_type == "linear":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        else: # huber (default)
            from sklearn.linear_model import HuberRegressor
            model = HuberRegressor(max_iter=min(2000, n_estimators * 10))

        model.fit(X, y)

        # 4. Persistence with Model Metadata (Institutional Spec)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_subdir = Path(model_dir) / f"{symbol}_v1_{timestamp}"
        model_subdir.mkdir(parents=True, exist_ok=True)

        model_pkl_path = model_subdir / "model.pkl"
        meta_json_path = model_subdir / "model_meta.json"

        # Atomic write: model pickle
        joblib.dump(model, model_pkl_path)

        # Prepare metadata
        training_start = df.index.min().isoformat() if hasattr(df.index.min(), 'isoformat') else str(df.index.min())
        training_end = df.index.max().isoformat() if hasattr(df.index.max(), 'isoformat') else str(df.index.max())

        model_meta = {
            "model_id": f"{symbol}_ml_v1",
            "version": "1.0",
            "features": list(X.columns),  # EXACT order used for training
            "n_features": len(X.columns),
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "scikit_version": sklearn.__version__,
            "git_commit": get_git_commit_hash(),
            "training_data_period": {
                "start": training_start,
                "end": training_end,
                "n_samples": len(X)
            },
            "symbol": symbol,
            "model_type": model.__class__.__name__,
            "params": {
                "model_type": model_type,
                "n_estimators": n_estimators if model_type == "gbr" else None,
                "max_depth": max_depth if model_type == "gbr" else None,
                "learning_rate": learning_rate if model_type == "gbr" else None
            },
            "contract_name": "ml_v1"
        }

        # Atomic write: metadata
        import tempfile
        import shutil
        import json
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump(model_meta, tmp, indent=2)
            tmp_path = tmp.name
        shutil.move(tmp_path, str(meta_json_path))

        # Update latest link (old style for compatibility)
        legacy_model_path = Path(model_dir) / f"{symbol}_v1_{timestamp}.joblib"
        joblib.dump({
            'model': model,
            'features': list(X.columns),
            'timestamp': timestamp,
            'metrics': {'samples': len(X)},
            'metadata': model_meta
        }, legacy_model_path)

        latest_link = Path(model_dir) / f"{symbol}_latest.joblib"
        joblib.dump({'path': str(legacy_model_path)}, latest_link)

        logger.info(f"Successfully trained and saved {model_type} model for {symbol}")
        return True

    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline ML Training Pipeline")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols to train")
    parser.add_argument("--model-type", type=str, choices=["huber", "linear", "gbr"], default="huber")
    parser.add_argument("--estimators", type=int, default=100, help="Number of trees (for GBR) or iterations (for Huber)")
    parser.add_argument("--epochs", type=int, help="Alias for --estimators")
    parser.add_argument("--max-depth", type=int, default=3, help="Max depth for GBR")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for GBR")
    args = parser.parse_args()

    # Alias --epochs to --estimators
    n_estimators = args.epochs if args.epochs is not None else args.estimators

    # Load feature contract
    logger.info("Loading ML v1 feature contract...")
    feature_contract = load_feature_contract("ml_v1")

    db = DatabaseManager()
    model_dir = "models/ml_alpha"

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = db.get_active_symbols()

    logger.info(f"Targeting {len(symbols)} symbols using {args.model_type} model")

    success_count = 0
    for symbol in symbols:
        df = db.get_daily_prices(symbol)
        if not df.empty:
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            if train_for_symbol(
                symbol, df, model_dir, feature_contract,
                model_type=args.model_type,
                n_estimators=n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate
            ):
                success_count += 1
        else:
            logger.warning(f"Empty data for {symbol}")

    logger.info(f"ML Training Campaign Complete. Successes: {success_count}/{len(symbols)}")

if __name__ == "__main__":
    main()
