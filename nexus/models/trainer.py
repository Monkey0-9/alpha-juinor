import os
import json
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from nexus.data.features import FeatureEngineer
from nexus.models.zoo.time_series import GradientBoostedTimeSeriesModel

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.path.join("models", "registry.json")


class ModelRegistry:
    """
    Manages active production models vs candidate models.
    Validates candidates out-of-sample and deploys ONLY if performance strictly improves.
    """

    def __init__(self, registry_file: str = REGISTRY_PATH):
        self.registry_file = registry_file
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        self.active_version = "v1.0.0"
        self.active_metrics: Dict[str, float] = {"accuracy": 0.50, "directional_f1": 0.50}
        self.active_model: Optional[Any] = None
        self._load_registry()

    def _load_registry(self) -> None:
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)
                    self.active_version = data.get("active_version", "v1.0.0")
                    self.active_metrics = data.get("active_metrics", self.active_metrics)
            except Exception as e:
                logger.warning("Failed to load model registry: %s", e)

    def _save_registry(self) -> None:
        try:
            with open(self.registry_file, "w") as f:
                json.dump({
                    "active_version": self.active_version,
                    "active_metrics": self.active_metrics,
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }, f, indent=2)
        except Exception as e:
            logger.error("Failed to save model registry: %s", e)

    def evaluate_candidate(
        self, candidate_model: Any, validation_data: pd.DataFrame
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Evaluates candidate model on out-of-sample validation data.
        Returns (is_promoted, candidate_metrics)
        """
        if validation_data.empty or len(validation_data) < 20:
            return False, {}

        features = FeatureEngineer.add_all_features(validation_data)
        if len(features) < 15:
            return False, {}

        correct = 0
        total = 0

        # Run directional out-of-sample predictions
        for i in range(len(features) - 1):
            window = features.iloc[:i+1]
            next_realized_ret = features['close'].iloc[i+1] - features['close'].iloc[i]

            pred_signal = candidate_model.predict(window)
            if pred_signal != 0:
                if (pred_signal * next_realized_ret) > 0:
                    correct += 1
                total += 1

        acc = (correct / total) if total > 0 else 0.50
        candidate_metrics = {"accuracy": round(acc, 4), "total_signals": total}

        # Deploy only if out-of-sample accuracy exceeds current active accuracy + 2% margin
        min_required = self.active_metrics.get("accuracy", 0.50) + 0.02
        if acc >= min_required and total >= 10:
            major, minor, patch = self.active_version.replace("v", "").split(".")
            new_version = f"v{major}.{minor}.{int(patch) + 1}"

            self.active_version = new_version
            self.active_metrics = candidate_metrics
            self.active_model = candidate_model
            self._save_registry()
            logger.info(
                "NEW MODEL PROMOTED TO PRODUCTION: %s (Accuracy: %.2f%% vs previous %.2f%%)",
                new_version, acc * 100, min_required * 100
            )
            return True, candidate_metrics

        logger.info(
            "Candidate model rejected (Accuracy: %.2f%% < required %.2f%%)",
            acc * 100, min_required * 100
        )
        return False, candidate_metrics


class ContinuousLearner:
    """
    Automated continuous retraining pipeline.
    Periodically retrains model on incoming bar data and submits to registry.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.last_retrain_time = 0.0
        self.retrain_interval = 3600.0 * 6  # Retrain every 6 hours

    def step_retrain(self, historical_bars: pd.DataFrame) -> bool:
        now = time.time()
        if now - self.last_retrain_time < self.retrain_interval:
            return False

        if historical_bars.empty or len(historical_bars) < 100:
            return False

        logger.info("Executing continuous retraining cycle on %d bars...", len(historical_bars))

        # Walk-forward split: 70% train, 30% out-of-sample validation
        split_idx = int(len(historical_bars) * 0.70)
        train_df = historical_bars.iloc[:split_idx]
        val_df = historical_bars.iloc[split_idx:]

        # Feature engineering
        train_features = FeatureEngineer.add_all_features(train_df)

        candidate = GradientBoostedTimeSeriesModel()
        if not candidate.fit(train_features):
            return False

        promoted, metrics = self.registry.evaluate_candidate(candidate, val_df)
        self.last_retrain_time = now
        return promoted
