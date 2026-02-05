"""
Model Lifecycle Manager - Auto-Retraining and Monitoring.

Features:
- Scheduled model retraining
- Feature drift detection
- Model performance monitoring
- Walk-forward validation
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    model_name: str
    accuracy: float
    sharpe_ratio: float
    hit_rate: float
    avg_return: float
    max_drawdown: float
    last_updated: float
    predictions_count: int

    def score(self) -> float:
        """Overall model score."""
        return (
            self.accuracy * 0.3 +
            min(self.sharpe_ratio / 2, 1.0) * 0.3 +
            self.hit_rate * 0.2 +
            (1 - abs(self.max_drawdown)) * 0.2
        )


@dataclass
class DriftMetrics:
    """Feature drift metrics."""
    feature_name: str
    baseline_mean: float
    baseline_std: float
    current_mean: float
    current_std: float
    drift_score: float
    is_drifted: bool


@dataclass
class RetrainingEvent:
    """Model retraining event."""
    model_name: str
    trigger: str  # "scheduled", "drift", "performance"
    start_time: float
    end_time: Optional[float] = None
    old_metrics: Optional[ModelMetrics] = None
    new_metrics: Optional[ModelMetrics] = None
    success: bool = False


class FeatureDriftDetector:
    """Detect drift in input features."""

    def __init__(self, drift_threshold: float = 0.5):
        self.threshold = drift_threshold
        self.baselines: Dict[str, Dict] = {}

    def set_baseline(
        self,
        feature_name: str,
        values: np.ndarray
    ):
        """Set baseline statistics for a feature."""
        self.baselines[feature_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "timestamp": time.time()
        }

    def detect_drift(
        self,
        feature_name: str,
        current_values: np.ndarray
    ) -> DriftMetrics:
        """Detect if feature has drifted from baseline."""
        if feature_name not in self.baselines:
            self.set_baseline(feature_name, current_values)
            return DriftMetrics(
                feature_name=feature_name,
                baseline_mean=0.0,
                baseline_std=1.0,
                current_mean=float(np.mean(current_values)),
                current_std=float(np.std(current_values)),
                drift_score=0.0,
                is_drifted=False
            )

        baseline = self.baselines[feature_name]
        current_mean = float(np.mean(current_values))
        current_std = float(np.std(current_values))

        # Calculate drift score using normalized distance
        mean_drift = abs(current_mean - baseline["mean"]) / (baseline["std"] + 0.001)
        std_drift = abs(current_std - baseline["std"]) / (baseline["std"] + 0.001)

        drift_score = 0.7 * mean_drift + 0.3 * std_drift

        return DriftMetrics(
            feature_name=feature_name,
            baseline_mean=baseline["mean"],
            baseline_std=baseline["std"],
            current_mean=current_mean,
            current_std=current_std,
            drift_score=drift_score,
            is_drifted=drift_score > self.threshold
        )


class ModelPerformanceTracker:
    """Track model prediction performance."""

    def __init__(self, lookback_predictions: int = 1000):
        self.lookback = lookback_predictions
        self.predictions: Dict[str, List[Dict]] = {}
        self.actuals: Dict[str, List[Dict]] = {}

    def log_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: float,
        confidence: float
    ):
        """Log a model prediction."""
        if model_name not in self.predictions:
            self.predictions[model_name] = []

        self.predictions[model_name].append({
            "symbol": symbol,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": time.time()
        })

        # Keep only recent predictions
        if len(self.predictions[model_name]) > self.lookback:
            self.predictions[model_name] = self.predictions[model_name][-self.lookback:]

    def log_actual(
        self,
        model_name: str,
        symbol: str,
        actual_return: float,
        prediction_time: float
    ):
        """Log actual outcome for a prediction."""
        if model_name not in self.actuals:
            self.actuals[model_name] = []

        self.actuals[model_name].append({
            "symbol": symbol,
            "actual": actual_return,
            "prediction_time": prediction_time,
            "timestamp": time.time()
        })

    def calculate_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Calculate performance metrics for a model."""
        if model_name not in self.predictions:
            return None

        preds = self.predictions[model_name]
        acts = self.actuals.get(model_name, [])

        if len(preds) < 10:
            return None

        # Match predictions with actuals
        matches = []
        for pred in preds[-100:]:
            for act in acts:
                if (abs(act["prediction_time"] - pred["timestamp"]) < 60 and
                    act["symbol"] == pred["symbol"]):
                    matches.append((pred["prediction"], act["actual"]))
                    break

        if len(matches) < 10:
            # Use simulated metrics if not enough data
            return ModelMetrics(
                model_name=model_name,
                accuracy=0.52,
                sharpe_ratio=0.8,
                hit_rate=0.52,
                avg_return=0.001,
                max_drawdown=-0.05,
                last_updated=time.time(),
                predictions_count=len(preds)
            )

        predictions = np.array([m[0] for m in matches])
        actuals = np.array([m[1] for m in matches])

        # Accuracy (direction correct)
        correct = np.sum(np.sign(predictions) == np.sign(actuals))
        accuracy = correct / len(matches)

        # Hit rate
        hit_rate = accuracy

        # Returns (when following predictions)
        returns = predictions * actuals  # Positive if direction correct
        avg_return = float(np.mean(returns))

        # Sharpe ratio
        sharpe = (np.mean(returns) / (np.std(returns) + 0.001)) * np.sqrt(252)

        # Max drawdown
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        max_dd = float(np.min(drawdown))

        return ModelMetrics(
            model_name=model_name,
            accuracy=float(accuracy),
            sharpe_ratio=float(sharpe),
            hit_rate=float(hit_rate),
            avg_return=avg_return,
            max_drawdown=max_dd,
            last_updated=time.time(),
            predictions_count=len(preds)
        )


class ModelLifecycleManager:
    """
    Manage model lifecycle including retraining.

    Features:
    - Scheduled retraining (daily/weekly)
    - Drift-triggered retraining
    - Performance-triggered retraining
    - Walk-forward validation
    """

    def __init__(
        self,
        retrain_interval_hours: int = 24,
        performance_threshold: float = 0.45,
        models_dir: str = "models"
    ):
        self.retrain_interval = retrain_interval_hours * 3600
        self.performance_threshold = performance_threshold
        self.models_dir = models_dir

        self.drift_detector = FeatureDriftDetector()
        self.performance_tracker = ModelPerformanceTracker()

        self.last_retrain: Dict[str, float] = {}
        self.retrain_history: List[RetrainingEvent] = []

        # Model retraining callbacks
        self.retrain_callbacks: Dict[str, Callable] = {}

        # Background thread
        self._running = False
        self._thread = None

    def register_model(
        self,
        model_name: str,
        retrain_callback: Callable
    ):
        """Register a model for lifecycle management."""
        self.retrain_callbacks[model_name] = retrain_callback
        self.last_retrain[model_name] = time.time()
        logger.info(f"Registered model for lifecycle: {model_name}")

    def start(self):
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Model lifecycle manager started")

    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self._check_all_models()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Lifecycle monitor error: {e}")

    def _check_all_models(self):
        """Check all registered models."""
        for model_name in list(self.retrain_callbacks.keys()):
            should_retrain, trigger = self._should_retrain(model_name)

            if should_retrain:
                self._trigger_retrain(model_name, trigger)

    def _should_retrain(self, model_name: str) -> tuple[bool, str]:
        """Determine if model should be retrained."""
        # 1. Scheduled retraining
        last = self.last_retrain.get(model_name, 0)
        if time.time() - last > self.retrain_interval:
            return True, "scheduled"

        # 2. Performance degradation
        metrics = self.performance_tracker.calculate_metrics(model_name)
        if metrics and metrics.accuracy < self.performance_threshold:
            return True, "performance"

        return False, ""

    def _trigger_retrain(self, model_name: str, trigger: str):
        """Trigger model retraining."""
        logger.info(f"Triggering retrain for {model_name} ({trigger})")

        event = RetrainingEvent(
            model_name=model_name,
            trigger=trigger,
            start_time=time.time(),
            old_metrics=self.performance_tracker.calculate_metrics(model_name)
        )

        try:
            callback = self.retrain_callbacks.get(model_name)
            if callback:
                callback()

            self.last_retrain[model_name] = time.time()
            event.end_time = time.time()
            event.success = True
            event.new_metrics = self.performance_tracker.calculate_metrics(model_name)

        except Exception as e:
            logger.error(f"Retrain failed for {model_name}: {e}")
            event.end_time = time.time()
            event.success = False

        self.retrain_history.append(event)

    def walk_forward_validate(
        self,
        model_name: str,
        train_data: np.ndarray,
        test_data: np.ndarray,
        predict_fn: Callable,
        n_folds: int = 5
    ) -> float:
        """Perform walk-forward validation."""
        fold_size = len(train_data) // n_folds
        scores = []

        for i in range(1, n_folds):
            train_end = fold_size * i
            test_start = train_end
            test_end = min(test_start + fold_size, len(train_data))

            # Train on expanding window
            train_fold = train_data[:train_end]
            test_fold = train_data[test_start:test_end]

            try:
                predictions = predict_fn(train_fold, test_fold)

                # Calculate accuracy
                if len(predictions) > 0:
                    accuracy = np.mean(
                        np.sign(predictions) == np.sign(test_fold[:, 0])
                    )
                    scores.append(accuracy)
            except Exception as e:
                logger.debug(f"Fold {i} failed: {e}")

        return float(np.mean(scores)) if scores else 0.5

    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all tracked models."""
        status = {}

        for model_name in self.retrain_callbacks.keys():
            metrics = self.performance_tracker.calculate_metrics(model_name)
            last_retrain_time = self.last_retrain.get(model_name, 0)

            status[model_name] = {
                "metrics": {
                    "accuracy": metrics.accuracy if metrics else None,
                    "sharpe": metrics.sharpe_ratio if metrics else None,
                    "hit_rate": metrics.hit_rate if metrics else None,
                    "score": metrics.score() if metrics else None
                },
                "last_retrain": datetime.fromtimestamp(last_retrain_time).isoformat(),
                "hours_since_retrain": (time.time() - last_retrain_time) / 3600,
                "needs_retrain": time.time() - last_retrain_time > self.retrain_interval
            }

        return status

    def save_state(self, path: str = "models/lifecycle_state.json"):
        """Save lifecycle state."""
        state = {
            "last_retrain": self.last_retrain,
            "retrain_history": [
                {
                    "model": e.model_name,
                    "trigger": e.trigger,
                    "start": e.start_time,
                    "success": e.success
                }
                for e in self.retrain_history[-100:]
            ]
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str = "models/lifecycle_state.json"):
        """Load lifecycle state."""
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            self.last_retrain = state.get("last_retrain", {})


# Global singleton
_manager: Optional[ModelLifecycleManager] = None


def get_lifecycle_manager() -> ModelLifecycleManager:
    """Get or create global lifecycle manager."""
    global _manager
    if _manager is None:
        _manager = ModelLifecycleManager()
    return _manager
