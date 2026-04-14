#!/usr/bin/env python3
"""
ML ONLINE LEARNING PIPELINE
============================

Real-time model retraining with continuous learning capabilities.

Features:
- Incremental learning (no full retraining required)
- Online feature adaptation
- Drift detection and model refreshing
- A/B testing framework for model versions
- Automated rollback on performance degradation

Author: MiniQuantFund ML Engineering
"""

import os
import sys
import json
import logging
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# ML libraries
try:
    from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from mini_quant_fund.core.production_config import config_manager
from mini_quant_fund.monitoring.production_monitor import get_production_monitor

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status."""
    TRAINING = "training"
    VALIDATING = "validating"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


class DriftType(Enum):
    """Types of data/concept drift."""
    NO_DRIFT = "no_drift"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"


@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    model_name: str
    created_at: datetime
    status: ModelStatus
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    training_samples: int
    validation_samples: int
    deployed_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    rollback_reason: Optional[str] = None


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning."""
    model_name: str = "alpha_predictor"
    batch_size: int = 1000
    learning_rate: float = 0.01
    max_buffer_size: int = 10000
    min_samples_for_update: int = 100
    validation_split: float = 0.2
    drift_threshold: float = 0.1
    performance_threshold: float = 0.05
    checkpoint_interval: int = 1000
    
    # Retraining schedule
    retrain_interval_minutes: int = 60
    full_retrain_interval_hours: int = 24


class IncrementalModel:
    """
    Wrapper for incremental learning models.
    Supports online updates without full retraining.
    """
    
    def __init__(self, model_name: str, model_type: str = "sgd"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying model."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available")
            return
        
        if self.model_type == "sgd":
            self.model = SGDRegressor(
                loss='squared_error',
                penalty='l2',
                alpha=0.0001,
                learning_rate='adaptive',
                eta0=0.01,
                max_iter=1000,
                tol=1e-3,
                random_state=42
            )
        elif self.model_type == "passive_aggressive":
            self.model = PassiveAggressiveRegressor(
                C=1.0,
                max_iter=1000,
                tol=1e-3,
                random_state=42
            )
        elif self.model_type == "hgb":
            self.model = HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Incrementally fit the model with new data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        if self.model is None:
            return
        
        # Scale features
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Partial fit
        try:
            self.model.partial_fit(X_scaled, y)
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Partial fit failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted or self.model is None:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        if not self.is_fitted:
            return 0.0
        
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'model_type': self.model_type
            }, f)
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_fitted = data['is_fitted']
            self.model_type = data['model_type']


class DriftDetector:
    """
    Detect data and concept drift in real-time.
    
    Uses statistical tests and performance monitoring.
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.reference_distribution: Optional[np.ndarray] = None
        self.performance_history: deque = deque(maxlen=1000)
    
    def set_reference(self, X: np.ndarray):
        """Set reference distribution for drift detection."""
        # Store mean and std of each feature
        self.reference_distribution = np.column_stack([
            np.mean(X, axis=0),
            np.std(X, axis=0)
        ])
    
    def detect_drift(self, X: np.ndarray, 
                    predictions: np.ndarray,
                    actual: np.ndarray) -> DriftType:
        """
        Detect drift in data or model performance.
        
        Returns:
            Type of drift detected
        """
        drift_detected = False
        
        # Check data drift (feature distribution shift)
        if self.reference_distribution is not None:
            current_mean = np.mean(X, axis=0)
            current_std = np.std(X, axis=0)
            
            # Calculate normalized distance from reference
            mean_distance = np.abs(
                current_mean - self.reference_distribution[:, 0]
            ) / (self.reference_distribution[:, 1] + 1e-8)
            
            if np.max(mean_distance) > 3.0:  # 3 sigma threshold
                drift_detected = True
                return DriftType.DATA_DRIFT
        
        # Check performance drift
        mse = mean_squared_error(actual, predictions)
        self.performance_history.append(mse)
        
        if len(self.performance_history) >= 100:
            recent_mse = np.mean(list(self.performance_history)[-50:])
            older_mse = np.mean(list(self.performance_history)[:50])
            
            if recent_mse > older_mse * (1 + self.threshold):
                return DriftType.PERFORMANCE_DRIFT
        
        return DriftType.NO_DRIFT


class OnlineLearningPipeline:
    """
    Production online learning pipeline.
    
    Manages model lifecycle:
    1. Continuous data ingestion
    2. Incremental training
    3. Validation and A/B testing
    4. Automated deployment
    5. Drift detection
    6. Rollback on degradation
    """
    
    def __init__(self, config: Optional[OnlineLearningConfig] = None):
        self.config = config or OnlineLearningConfig()
        self.models: Dict[str, IncrementalModel] = {}
        self.versions: Dict[str, List[ModelVersion]] = {}
        self.drift_detectors: Dict[str, DriftDetector] = {}
        
        # Data buffers
        self.feature_buffer: deque = deque(maxlen=self.config.max_buffer_size)
        self.target_buffer: deque = deque(maxlen=self.config.max_buffer_size)
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Metrics
        self.update_count = 0
        self.last_retrain = datetime.utcnow()
        
        # Initialize MLflow
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri("sqlite:///mlruns/tracking.db")
    
    def register_model(self, model_name: str, model_type: str = "sgd"):
        """Register a new model for online learning."""
        with self._lock:
            self.models[model_name] = IncrementalModel(model_name, model_type)
            self.versions[model_name] = []
            self.drift_detectors[model_name] = DriftDetector(
                self.config.drift_threshold
            )
            
            logger.info(f"Registered model: {model_name} (type: {model_type})")
    
    def ingest_sample(self, 
                      model_name: str,
                      features: Dict[str, float],
                      target: float,
                      timestamp: Optional[datetime] = None):
        """
        Ingest a new training sample.
        
        Args:
            model_name: Name of the model to update
            features: Feature dictionary
            target: Target value
            timestamp: Optional timestamp
        """
        with self._lock:
            self.feature_buffer.append(features)
            self.target_buffer.append(target)
            
            # Trigger update if buffer is full enough
            if len(self.feature_buffer) >= self.config.batch_size:
                self._trigger_update(model_name)
    
    def _trigger_update(self, model_name: str):
        """Trigger incremental model update."""
        if model_name not in self.models:
            return
        
        # Convert buffers to arrays
        if len(self.feature_buffer) < self.config.min_samples_for_update:
            return
        
        # Extract features
        feature_keys = list(self.feature_buffer[0].keys())
        X = np.array([[f[k] for k in feature_keys] for f in self.feature_buffer])
        y = np.array(list(self.target_buffer))
        
        # Clear buffers
        self.feature_buffer.clear()
        self.target_buffer.clear()
        
        # Split for validation
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Update model
        model = self.models[model_name]
        model.partial_fit(X_train, y_train)
        
        # Validate
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val) if len(X_val) > 0 else 0.0
        
        # Check drift
        predictions = model.predict(X_val) if len(X_val) > 0 else y_train
        actual = y_val if len(X_val) > 0 else y_train
        
        drift_detector = self.drift_detectors[model_name]
        drift_type = drift_detector.detect_drift(
            X_val if len(X_val) > 0 else X_train,
            predictions,
            actual
        )
        
        # Log metrics
        self.update_count += 1
        
        if self.update_count % self.config.checkpoint_interval == 0:
            self._save_checkpoint(model_name, {
                'train_score': train_score,
                'val_score': val_score,
                'samples': len(X),
                'drift': drift_type.value
            })
        
        # Alert on drift
        if drift_type != DriftType.NO_DRIFT:
            logger.warning(f"Drift detected for {model_name}: {drift_type.value}")
            self._handle_drift(model_name, drift_type)
        
        logger.debug(f"Model {model_name} updated: train_score={train_score:.4f}, "
                    f"val_score={val_score:.4f}")
    
    def _handle_drift(self, model_name: str, drift_type: DriftType):
        """Handle detected drift."""
        # Rollback to previous version if performance degraded
        if drift_type == DriftType.PERFORMANCE_DRIFT:
            versions = self.versions.get(model_name, [])
            production_versions = [v for v in versions 
                                  if v.status == ModelStatus.PRODUCTION]
            
            if len(production_versions) >= 2:
                # Rollback to previous production version
                previous = production_versions[-2]
                self.rollback_model(model_name, previous.version_id)
    
    def _save_checkpoint(self, model_name: str, metrics: Dict):
        """Save model checkpoint."""
        version_id = hashlib.md5(
            f"{model_name}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Save model
        checkpoint_path = f"models/{model_name}/{version_id}.pkl"
        self.models[model_name].save(checkpoint_path)
        
        # Create version record
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            created_at=datetime.utcnow(),
            status=ModelStatus.VALIDATING,
            metrics=metrics,
            feature_importance={},
            training_samples=metrics.get('samples', 0),
            validation_samples=0
        )
        
        self.versions[model_name].append(version)
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"{model_name}_{version_id}"):
                mlflow.log_params({
                    'model_name': model_name,
                    'version': version_id
                })
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(checkpoint_path)
    
    def deploy_model(self, model_name: str, version_id: str):
        """Deploy specific model version to production."""
        with self._lock:
            versions = self.versions.get(model_name, [])
            
            for v in versions:
                if v.version_id == version_id:
                    v.status = ModelStatus.PRODUCTION
                    v.deployed_at = datetime.utcnow()
                    logger.info(f"Deployed {model_name} v{version_id} to production")
                    return True
            
            return False
    
    def rollback_model(self, model_name: str, version_id: str):
        """Rollback to previous version."""
        with self._lock:
            versions = self.versions.get(model_name, [])
            
            # Deprecate current production
            for v in versions:
                if v.status == ModelStatus.PRODUCTION:
                    v.status = ModelStatus.DEPRECATED
                    v.deprecated_at = datetime.utcnow()
            
            # Deploy rollback version
            for v in versions:
                if v.version_id == version_id:
                    v.status = ModelStatus.PRODUCTION
                    v.rollback_reason = "Automatic rollback due to drift"
                    
                    # Load model weights
                    checkpoint_path = f"models/{model_name}/{version_id}.pkl"
                    if Path(checkpoint_path).exists():
                        self.models[model_name].load(checkpoint_path)
                    
                    logger.warning(f"Rolled back {model_name} to v{version_id}")
                    return True
            
            return False
    
    def predict(self, model_name: str, features: Dict[str, float]) -> float:
        """Make prediction with current model."""
        if model_name not in self.models:
            return 0.0
        
        X = np.array([[features.get(k, 0.0) for k in features.keys()]])
        prediction = self.models[model_name].predict(X)
        return float(prediction[0])
    
    def get_model_status(self, model_name: str) -> Dict:
        """Get comprehensive model status."""
        with self._lock:
            model = self.models.get(model_name)
            versions = self.versions.get(model_name, [])
            
            if not model:
                return {"error": "Model not found"}
            
            production = [v for v in versions if v.status == ModelStatus.PRODUCTION]
            
            return {
                "model_name": model_name,
                "is_fitted": model.is_fitted,
                "update_count": self.update_count,
                "versions_count": len(versions),
                "production_version": production[-1].version_id if production else None,
                "last_retrain": self.last_retrain.isoformat(),
                "buffer_size": len(self.feature_buffer)
            }
    
    def start(self):
        """Start background monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Online learning pipeline started")
    
    def stop(self):
        """Stop pipeline."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self._executor.shutdown(wait=False)
        logger.info("Online learning pipeline stopped")
    
    def _monitor_loop(self):
        """Background monitoring for periodic retraining."""
        while self._running:
            # Check if full retrain needed
            elapsed = datetime.utcnow() - self.last_retrain
            
            if elapsed >= timedelta(hours=self.config.full_retrain_interval_hours):
                for model_name in self.models:
                    self._schedule_full_retrain(model_name)
                self.last_retrain = datetime.utcnow()
            
            time.sleep(60)  # Check every minute
    
    def _schedule_full_retrain(self, model_name: str):
        """Schedule full model retraining."""
        logger.info(f"Scheduling full retrain for {model_name}")
        # Would trigger full retrain on historical data
        # Implementation depends on data storage backend


# Global pipeline instance
_online_learning_pipeline: Optional[OnlineLearningPipeline] = None


def get_online_learning_pipeline() -> OnlineLearningPipeline:
    """Get global online learning pipeline."""
    global _online_learning_pipeline
    if _online_learning_pipeline is None:
        _online_learning_pipeline = OnlineLearningPipeline()
    return _online_learning_pipeline


if __name__ == "__main__":
    # Test online learning pipeline
    print("Testing Online Learning Pipeline...")
    
    pipeline = OnlineLearningPipeline()
    pipeline.register_model("alpha_predictor", "sgd")
    
    # Simulate data ingestion
    for i in range(500):
        features = {
            'rsi': np.random.uniform(0, 100),
            'macd': np.random.uniform(-1, 1),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'price_momentum': np.random.uniform(-0.1, 0.1)
        }
        target = features['price_momentum'] * 10 + np.random.normal(0, 0.01)
        
        pipeline.ingest_sample("alpha_predictor", features, target)
    
    # Check status
    status = pipeline.get_model_status("alpha_predictor")
    print(f"\nModel Status: {json.dumps(status, indent=2)}")
    
    # Test prediction
    test_features = {
        'rsi': 50.0,
        'macd': 0.5,
        'volume_ratio': 1.5,
        'price_momentum': 0.05
    }
    prediction = pipeline.predict("alpha_predictor", test_features)
    print(f"\nPrediction: {prediction:.4f}")
