# strategies/ml_alpha.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from typing import Optional

from data.processors.features import FeatureEngineer
from strategies.alpha import Alpha

class MLAlpha(Alpha):
    """
    Machine Learning Alpha.
    Trains a Random Forest models to predict forward returns.
    """

    def __init__(self, feature_engineer: FeatureEngineer, train_window: int = 500):
        self.fe = feature_engineer
        self.model = RandomForestRegressor(
            n_estimators=50, 
            max_depth=5, 
            min_samples_leaf=10, 
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.ml_ready = False
        self.train_window = train_window
        self.retrain_interval = 63 # Retrain every quarter (~63 business days)
        self.last_train_date: Optional[pd.Timestamp] = None
        self.last_features: Optional[pd.DataFrame] = None

    def train(self, prices_history: pd.DataFrame):
        """
        Train the model on historical data.
        prices_history: DataFrame with Open, High, Low, Close, Volume
        """
        if len(prices_history) < self.train_window:
            print(f"[MLAlpha] Insufficient history to train. Need {self.train_window}, got {len(prices_history)}.")
            return

        # Institutional Alignment: We want to ensure features and target use the SAME raw data subset
        # before they start dropping rows internally.
        print(f"[MLAlpha] Generating features and 5d forward targets for {len(prices_history)} bars...")
        X = self.fe.compute_features(prices_history)
        y = self.fe.compute_target(prices_history, forward_window=5) # Predict 5-day return

        # Align X and y
        # y has NaNs at end (forward look), X has NaNs at start (lags)
        data = pd.concat([X, y.rename("target")], axis=1).dropna()
        
        if data.empty:
            print("[MLAlpha] No valid data after alignment.")
            return

        # Train/Test logic (simplified: train on all valid history for this "mini" fund)
        X_train = data[X.columns]
        y_train = data["target"]

        print(f"[MLAlpha] Training Random Forest on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.ml_ready = True
        self.last_features = X.iloc[[-1]] # Cache last features for live check if needed
        
        # Feature importance debug
        # importances = pd.Series(self.model.feature_importances_, index=X.columns).sort_values(ascending=False)
        # print("Top Features:\n", importances.head(3))

    def compute(self, prices: pd.Series) -> pd.Series:
        """
        Alpha interface requires input as Series (Close prices).
        BUT ML model needs full OHLV frame. 
        In this specific architecture, 'compute' is usually called with just a Series by the Engine.
        
        CRITICAL ADAPTATION:
        The engine in `main.py` strategy_fn extracts history cache. 
        We need to refactor `main.py` or the engine to pass the full dataframe if possible,
        OR we rely on the fact that `MLAlpha` might need to conduct its own lookups 
        or we change the signature in `main.py` before calling this.
        
        For now, `compute` will return 0.5 (neutral) if it can't run, 
        but strictly, `predict` should be called explicitly in main.py with full DataFrame.
        """
        return pd.Series(0.0, index=prices.index)

    def predict_conviction(self, full_ohlcv: pd.DataFrame) -> float:
        """
        Custom method for ML prediction using full DataFrame.
        Returns float 0..1
        """
        if not self.is_trained:
            # Default to Neutral (0.5) if not trained, allowing other alphas to drive.
            return 0.5

        # Compute features for the LATEST bar only
        # We need a window of history to calculate features (e.g. Rolling 63d)
        # So we pass the tail of the dataframe
        
        # Optimization: We assume full_ohlcv contains enough history
        features_full = self.fe.compute_features(full_ohlcv)
        
        if features_full.empty:
            return 0.5

        current_features = features_full.iloc[[-1]]
        
        # Check for NaNs (if not enough history for lags)
        if current_features.isnull().values.any():
            return 0.5

        pred_ret = self.model.predict(current_features)[0]
        
        # Map predicted return to conviction 0..1
        # E.g. +1% return -> 0.6, +5% -> 0.9, -1% -> 0.4
        # Sigmoid or linear scaling
        
        # Simple linear scaling:
        # -0.05 (-5%) -> 0.0
        # +0.05 (+5%) -> 1.0
        
        score = (pred_ret + 0.05) / 0.10
        return float(np.clip(score, 0.0, 1.0))

    def should_retrain(self, current_date: pd.Timestamp) -> bool:
        """
        Check if the model is due for institutional walk-forward re-training.
        """
        if not self.is_trained:
            return True
        if self.last_train_date is None:
            return True
        
        days_since = (current_date - self.last_train_date).days
        return days_since >= self.retrain_interval

    def walk_forward_update(self, current_date: pd.Timestamp, prices_history: pd.DataFrame):
        """
        Triggered re-training as part of a walk-forward process.
        Ensures model stays fresh and regime-aware.
        """
        if self.should_retrain(current_date):
            print(f"[MLAlpha] Walk-forward trigger at {current_date.date()}. Retraining...")
            # Use a rolling window of history for training to focus on recent regimes
            training_data = prices_history.tail(self.train_window)
            self.train(training_data)
            self.last_train_date = current_date
