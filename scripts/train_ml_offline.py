
import sqlite3
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Setup
DB = "runtime/institutional_trading.db"
MODEL_DIR = Path("runtime/models/ml_alpha")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRAIN_OFFLINE")

def load_all_training_data():
    """Load price history for all ACTIVE symbols."""
    con = sqlite3.connect(DB)
    # Get active symbols
    active_symbols = [r[0] for r in con.execute("SELECT symbol FROM trading_eligibility WHERE state='ACTIVE'").fetchall()]

    all_data = []

    for sym in active_symbols:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM price_history WHERE symbol=? ORDER BY date ASC",
            con, params=(sym,)
        )
        if len(df) > 252:
            df['symbol'] = sym
            all_data.append(df)

    con.close()
    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data)

def extract_features(df):
    """Simple feature extraction matching MLAlpha expectations significantly simplified."""
    # We need to calculate features PER SYMBOL
    # Group by symbol

    df = df.sort_values(['symbol', 'date'])

    # Calculate returns
    df['returns'] = df.groupby('symbol')['close'].pct_change()

    # Momentum
    for p in [5, 10, 20]:
        df[f'momentum_{p}'] = df.groupby('symbol')['close'].pct_change(p)

    # Target: Next 5 day return direction (1 if > 0, 0 else)
    df['target_return'] = df.groupby('symbol')['close'].pct_change(5).shift(-5)
    df['target'] = (df['target_return'] > 0).astype(int)

    # Drop NaNs
    df = df.dropna()

    feature_cols = [c for c in df.columns if 'momentum' in c or c == 'returns']
    return df[feature_cols], df['target']

def train_model():
    logger.info("Loading training data...")
    df = load_all_training_data()
    if df.empty:
        logger.error("No data found for training.")
        return

    logger.info(f"Loaded {len(df)} rows. Extracting features...")
    X, y = extract_features(df)

    if len(X) < 1000:
        logger.warning(f"Insufficient samples for training: {len(X)}")
        return

    logger.info(f"Training on {len(X)} samples with {X.shape[1]} features...")

    # Train/Test logic
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])

    pipeline.fit(X, y)

    logger.info("Saving model...")
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
