import os
import numpy as np
import pandas as pd
import xgboost as xgb
import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(f"PyTorch is not available or failed to load: {e}. Deep Learning modules will be disabled.")
    TORCH_AVAILABLE = False
from typing import Dict, Any, Union, List, Optional  # noqa: E402

class AdvancedMLBrain:
    """
    Institutional Machine Learning Brain utilizing XGBoost (C++ accelerated under the hood).
    Includes TimeSeriesSplit K-Fold Cross-Validation, Optuna Hyperparameter Optimization,
    Feature Importance analysis, and Model Versioning metadata.
    """

    MODEL_VERSION = "3.0.0"

    def __init__(self, model_path: str = "models/xgboost_brain.json"):
        self.model_path = model_path
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=42
        )
        self.is_trained = False
        self.feature_importances_: Dict[str, float] = {}

        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if os.path.exists(self.model_path):
            try:
                self.model.load_model(self.model_path)
                self.is_trained = True
                logger.info(f"Loaded pre-trained XGBoost Brain (version {self.MODEL_VERSION}).")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")

    def train_with_cv(self, features: pd.DataFrame, targets: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """Train using TimeSeriesSplit Cross-Validation for temporal data stability."""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error

        if features.empty or targets.empty:
            logger.warning("Empty dataset for CV training.")
            return {}

        logger.info(f"Running {n_splits}-Fold TimeSeriesSplit Cross-Validation...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
            X_tr, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_tr, y_val = targets.iloc[train_idx], targets.iloc[val_idx]

            fold_model = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=5, n_jobs=-1, random_state=42
            )
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            cv_scores.append(rmse)
            logger.debug(f"Fold {fold+1}/{n_splits} Validation RMSE: {rmse:.4f}")

        # Final fit on full dataset
        self.train(features, targets)
        return {"mean_cv_rmse": float(np.mean(cv_scores)), "std_cv_rmse": float(np.std(cv_scores))}

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train model on historical features & save model with feature importance."""
        if features.empty or targets.empty:
            logger.warning("Empty data provided for ML training.")
            return

        logger.info(f"Training XGBoost ML Brain (v{self.MODEL_VERSION}) on {len(features)} samples...")
        self.model.fit(features, targets)
        self.is_trained = True

        if hasattr(self.model, "feature_importances_") and hasattr(features, "columns"):
            importances = self.model.feature_importances_
            self.feature_importances_ = {
                col: float(imp) for col, imp in zip(features.columns, importances)
            }

        try:
            self.model.save_model(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def optimize_hyperparams(self, features: pd.DataFrame, targets: pd.Series, n_trials: int = 20) -> Dict[str, Any]:
        """Optuna Hyperparameter Optimization for XGBoost parameters."""
        try:
            import optuna

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                    'max_depth': trial.suggest_int('max_depth', 3, 9),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'n_jobs': -1,
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)
                model.fit(features, targets)
                preds = model.predict(features)
                return float(np.mean((targets - preds) ** 2))

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, timeout=60)
            best_params = study.best_params
            logger.info(f"Optuna Best Hyperparameters: {best_params}")
            self.model = xgb.XGBRegressor(**best_params)
            self.train(features, targets)
            return best_params
        except ImportError:
            logger.warning("Optuna not installed. Skipping hyperparameter optimization.")
            return {}

    def get_feature_importance(self) -> Dict[str, float]:
        """Returns feature importance ranking sorted by importance descending."""
        if not self.feature_importances_:
            return {}
        return dict(sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True))

    def predict(self, current_features: Dict[str, float]) -> float:
        if not self.is_trained:
            return np.mean(list(current_features.values())) if current_features else 0.0

        df = pd.DataFrame([current_features])
        try:
            pred = self.model.predict(df)[0]
            return float(np.clip(pred, -1.0, 1.0))
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0

    def batch_predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or features_df.empty:
            return np.zeros(len(features_df))

        return np.clip(self.model.predict(features_df), -1.0, 1.0)


# ------------------------------------------------------------------ #
# ONNX Deep Learning Brain (C++ / TensorRT Accelerated Inference)     #
# ------------------------------------------------------------------ #

if TORCH_AVAILABLE:
    BaseModule = nn.Module
else:
    class BaseModule:
        pass


class ONNXBrain:
    """
    Advanced Deep Learning Brain utilizing ONNX Runtime for TensorRT / CUDA / C++ execution.
    Loads Transformer/LSTM/PPO models exported from PyTorch.
    """
    def __init__(self, model_path: str = "models/transformer_brain.onnx", input_dim: int = 15):
        self.model_path = model_path
        self.input_dim = input_dim
        self.session = None

        # Priority execution providers: TensorRT -> CUDA -> CPU
        self.providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not os.path.exists(self.model_path):
            self._export_initial_model()

        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            self.input_name = self.session.get_inputs()[0].name
            logger.info("Loaded C++ ONNX Inference Session for TransformerBrain.")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")

    def _export_initial_model(self):
        """Creates an untrained PyTorch model and exports it to ONNX for the C++ runtime."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping ONNX model generation.")
            return

        logger.info(f"Exporting initial PyTorch Transformer to ONNX: {self.model_path}")
        model = TransformerFeatureExtractor(input_dim=self.input_dim)
        model.eval()
        
        dummy_input = torch.randn(1, self.input_dim)
        torch.onnx.export(
            model, 
            dummy_input, 
            self.model_path, 
            export_params=True,
            opset_version=14, 
            do_constant_folding=True,
            input_names=['features'],
            output_names=['signal'],
            dynamic_axes={'features': {0: 'batch_size'}, 'signal': {0: 'batch_size'}}
        )

    def predict(self, current_features: Any) -> float:
        if self.session is None or current_features is None:
            return 0.0

        if isinstance(current_features, dict):
            vals = list(current_features.values())
        elif isinstance(current_features, (list, tuple)):
            vals = list(current_features)
        elif isinstance(current_features, np.ndarray):
            vals = current_features.flatten().tolist()
        else:
            vals = []

        if len(vals) < self.input_dim:
            vals = vals + [0.0] * (self.input_dim - len(vals))
        elif len(vals) > self.input_dim:
            vals = vals[:self.input_dim]

        input_data = np.array([vals], dtype=np.float32)

        # Check expected rank of ONNX input tensor
        try:
            expected_shape = self.session.get_inputs()[0].shape
            if len(expected_shape) == 3:
                # Add sequence dimension for 3D inputs (batch_size, seq_len, input_dim)
                input_data = np.expand_dims(input_data, axis=1)

            ort_outs = self.session.run(None, {self.input_name: input_data})
            out = ort_outs[0]
            pred = float(np.ravel(out)[0])
            return float(np.clip(pred, -1.0, 1.0))
        except Exception as e:
            logger.error(f"ONNX prediction failed: {e}")
            return 0.0

    def batch_predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if self.session is None or features_df.empty:
            return np.zeros(len(features_df))

        input_data = features_df.to_numpy(dtype=np.float32)
        if input_data.shape[1] < self.input_dim:
            pad = np.zeros((input_data.shape[0], self.input_dim - input_data.shape[1]), dtype=np.float32)
            input_data = np.hstack([input_data, pad])
        elif input_data.shape[1] > self.input_dim:
            input_data = input_data[:, :self.input_dim]

        try:
            expected_shape = self.session.get_inputs()[0].shape
            if len(expected_shape) == 3:
                input_data = np.expand_dims(input_data, axis=1)

            ort_outs = self.session.run(None, {self.input_name: input_data})
            preds = np.ravel(ort_outs[0])
            return np.clip(preds, -1.0, 1.0)
        except Exception as e:
            logger.error(f"ONNX batch prediction failed: {e}")
            return np.zeros(len(features_df))
