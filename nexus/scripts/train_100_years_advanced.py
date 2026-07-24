import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from nexus.data.historical_data_loader import HistoricalDataLoader
from nexus.core.strategies import StrategyFactory
from nexus.core.ml_brain import AdvancedMLBrain
from nexus.math.indicators import compute_hurst_exponent
from nexus.math.models import FractalEngine
from nexus.math.risk import RiskEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRAINING_SYMBOLS = [
    "^GSPC",  # S&P 500 (1927+)
    "^DJI",   # Dow Jones
    "^IXIC",  # NASDAQ Composite
    "^RUT",   # Russell 2000
    "SPY",    # S&P 500 ETF (1993+)
    "QQQ",    # NASDAQ ETF (1999+)
    "IWM",    # Russell 2000 ETF (2000+)
    "TLT",    # Long-Term Treasury (2002+)
    "GLD",    # Gold ETF (2004+)
    "XLF",    # Financial Sector (1998+)
    "XLE",    # Energy Sector
    "XLK",    # Tech Sector
    "^VIX",   # Volatility Index (1990+)
    "^FTSE",  # FTSE 100
    "^N225",  # Nikkei 225
    "DX-Y.NYB", # US Dollar Index
]

def generate_enhanced_features(df, lookback=100):
    features_list = []
    targets = []
    strategies = StrategyFactory.all_strategies()

    for i in range(lookback, len(df) - 5):
        window = df.iloc[i - lookback:i]
        future_return = np.sum(df["returns"].iloc[i:i+5].values)

        feature_row = {}

        # Regime probabilities
        from nexus.math.indicators import RegimeDetector
        detector = RegimeDetector()
        probs = detector.detect_probabilities(window)
        for regime, prob in probs.items():
            feature_row[f"regime_{regime.lower()}"] = prob

        # Multi-timeframe returns
        for period in [1, 5, 10, 21, 63]:
            if i >= period:
                ret = df["close"].iloc[i] / df["close"].iloc[i - period] - 1
                feature_row[f"ret_{period}d"] = float(ret)

        # Volatility features
        for period in [5, 10, 21, 63]:
            vol = df["returns"].iloc[i - period:i].std() * np.sqrt(252)
            feature_row[f"vol_{period}d"] = float(vol) if not np.isnan(vol) else 0.0

        # Technical indicators
        close = window["close"].values
        sma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
        feature_row["sma_crossover"] = float(sma_20 / sma_50 - 1)

        if len(close) >= 14:
            delta = np.diff(close)
            gains = delta[delta > 0].mean() if len(delta[delta > 0]) > 0 else 0
            losses = -delta[delta < 0].mean() if len(delta[delta < 0]) > 0 else 0
            rs = gains / max(losses, 1e-10)
            rsi = 100 - 100 / (1 + rs)
            feature_row["rsi"] = float(rsi)

        # Hurst exponent
        price_series = pd.Series(close)
        hurst = compute_hurst_exponent(price_series)
        feature_row["hurst"] = float(hurst)

        # Fractal dimension
        fe = FractalEngine()
        fd = fe.calculate_dimension(close)
        feature_row["fractal_dim"] = float(fd)

        # Strategy signals
        current_regime = "BULL" if probs.get("BULL", 0) > 0.3 else "BEAR"
        for strat in strategies:
            score = strat.score("^GSPC", 0.0, window, current_regime)
            feature_row[f"strat_{strat.name}"] = float(score)

        # Risk metrics
        returns_arr = df["returns"].iloc[i - min(252, i):i].values
        if len(returns_arr) > 20:
            risk_engine = RiskEngine()
            risk = risk_engine.assess_risk(returns_arr)
            for k, v in risk.items():
                feature_row[f"risk_{k}"] = float(v)

        features_list.append(feature_row)
        targets.append(float(np.tanh(future_return * 20)))

    return pd.DataFrame(features_list), pd.Series(targets)

def train_100_years_advanced():
    loader = HistoricalDataLoader()

    all_features = []
    all_targets = []

    primary_symbols = ["^GSPC", "^DJI", "^IXIC", "^RUT"]

    for symbol in primary_symbols:
        logger.info(f"Processing {symbol}...")
        df = loader.load_macro_data(symbol)
        if df.empty or len(df) < 500:
            logger.warning(f"Skipping {symbol}: insufficient data ({len(df)} rows)")
            continue

        logger.info(f"  {symbol}: {len(df)} days from {df.index.min().date()} to {df.index.max().date()}")
        features, targets = generate_enhanced_features(df)

        if len(features) > 0:
            all_features.append(features)
            all_targets.append(targets)

    if not all_features:
        logger.error("No training data generated from any symbol")
        return

    combined_features = pd.concat(all_features, ignore_index=True).fillna(0.0)
    combined_targets = pd.concat(all_targets, ignore_index=True)
    logger.info(f"Total training samples: {len(combined_features)}")

    split = int(len(combined_features) * 0.8)
    train_features = combined_features.iloc[:split]
    train_targets = combined_targets.iloc[:split]
    test_features = combined_features.iloc[split:]
    test_targets = combined_targets.iloc[split:]

    ml_brain = AdvancedMLBrain(model_path="nexus/models/xgboost_brain_advanced.json")
    ml_brain.model.set_params(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1
    )
    ml_brain.train(train_features, train_targets)

    from sklearn.metrics import mean_squared_error
    train_pred = ml_brain.model.predict(train_features)
    test_pred = ml_brain.model.predict(test_features)
    train_rmse = np.sqrt(mean_squared_error(train_targets, train_pred))
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_pred))
    train_ic = np.corrcoef(train_pred, train_targets)[0, 1]
    test_ic = np.corrcoef(test_pred, test_targets)[0, 1]

    logger.info("XGBoost Advanced Training Results:")
    logger.info(f"  Train RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
    logger.info(f"  Train IC: {train_ic:.4f}, Test IC: {test_ic:.4f}")

    ml_brain.model.save_model("nexus/models/xgboost_brain_advanced.json")
    logger.info("Advanced XGBoost model saved")

    importances = ml_brain.model.feature_importances_
    top_features = pd.DataFrame({
        "feature": train_features.columns,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)
    logger.info(f"Top 10 features:\n{top_features.head(10).to_string()}")

    ml_brain.model.save_model("nexus/models/xgboost_brain.json")
    logger.info("Production XGBoost model updated")

    logger.info("100-Year Advanced Training Complete!")

if __name__ == "__main__":
    train_100_years_advanced()