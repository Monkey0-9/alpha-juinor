import os
import numpy as np
import pandas as pd
import xgboost as xgb
import onnxruntime as ort
import logging
import json
import tempfile
from typing import Dict, Any, Union, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(f"PyTorch is not available or failed to load: {e}. Deep Learning modules will be disabled.")
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class AdvancedMLBrain:
    def __init__(self, model_path: str = "models/xgboost_brain.json", version: str = "2.0.0"):
        self.model_path = model_path
        self.version = version
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric=['rmse', 'mae'],
        )
        self.is_trained = False
        self.feature_importance = {}
        self.training_history = []
        self.best_score = float('inf')
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if os.path.exists(self.model_path):
            try:
                self.model.load_model(self.model_path)
                self.is_trained = True
                logger.info("Loaded pre-trained XGBoost Brain v%s", self.version)
            except Exception as e:
                logger.warning("Could not load ML model: %s", e)

    def hyperopt_tune(self, features: pd.DataFrame, targets: pd.Series, n_trials: int = 50):
        if not OPTUNA_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("Optuna or sklearn not available, skipping hyperparameter tuning")
            return
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            }
            model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, features, targets, cv=tscv, scoring='neg_mean_squared_error')
            return scores.mean()
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        logger.info("Optuna best params: %s (score=%.4f)", best_params, study.best_value)
        self.model.set_params(**best_params)

    def train(self, features: pd.DataFrame, targets: pd.Series, eval_set=None) -> Dict[str, float]:
        if features.empty or targets.empty:
            logger.warning("Empty data provided for ML training.")
            return {}
        logger.info("Training XGBoost ML Brain on %s samples with %s features...", len(features), len(features.columns))
        if SKLEARN_AVAILABLE:
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(self.model, features, targets, cv=tscv, scoring='neg_mean_squared_error')
            logger.info("CV RMSE scores: mean=%.4f, std=%.4f", np.sqrt(-cv_scores.mean()), np.sqrt(-cv_scores).std())
        if eval_set is not None:
            self.model.fit(features, targets, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(features, targets)
        self.is_trained = True
        train_pred = self.model.predict(features)
        train_rmse = float(np.sqrt(mean_squared_error(targets, train_pred)))
        train_mae = float(mean_absolute_error(targets, train_pred))
        train_ic = float(np.corrcoef(train_pred, targets)[0, 1]) if np.std(train_pred) > 1e-9 else 0.0
        self.best_score = min(self.best_score, train_rmse)
        self.feature_importance = dict(zip(features.columns, self.model.feature_importances_.tolist()))
        metrics = {'rmse': train_rmse, 'mae': train_mae, 'ic': train_ic}
        self.training_history.append({'timestamp': datetime.now().isoformat(), 'samples': len(features), **metrics})
        logger.info("Training complete: RMSE=%.6f, MAE=%.6f, IC=%.4f", train_rmse, train_mae, train_ic)
        try:
            self.model.save_model(self.model_path)
            meta_path = self.model_path.replace('.json', '_meta.json')
            with open(meta_path, 'w') as f:
                json.dump({'version': self.version, 'feature_importance': self.feature_importance, 'training_history': self.training_history, 'best_score': self.best_score}, f, indent=2)
            logger.info("Model saved to %s with metadata", self.model_path)
        except Exception as e:
            logger.error("Failed to save model: %s", e)
        return metrics

    def predict(self, current_features: Dict[str, float]) -> float:
        if not self.is_trained:
            return np.mean(list(current_features.values())) if current_features else 0.0
        df = pd.DataFrame([current_features])
        missing = set(self.model.get_booster().feature_names) - set(df.columns)
        for col in missing:
            df[col] = 0.0
        df = df[[c for c in self.model.get_booster().feature_names if c in df.columns]]
        try:
            pred = self.model.predict(df)[0]
            return float(np.clip(pred, -1.0, 1.0))
        except Exception as e:
            logger.error("Prediction failed: %s", e)
            return 0.0

    def batch_predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or features_df.empty:
            return np.zeros(len(features_df))
        return np.clip(self.model.predict(features_df), -1.0, 1.0)

    def get_top_features(self, n: int = 10) -> List[tuple]:
        sorted_imp = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_imp[:n]


if TORCH_AVAILABLE:
    BaseModule = nn.Module
else:
    class BaseModule:
        pass

class TransformerFeatureExtractor(BaseModule):
    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.output(x)


class ONNXBrain:
    def __init__(self, model_path: str = "models/transformer.onnx", input_dim: int = 15):
        self.model_path = model_path
        self.input_dim = input_dim
        self.session = None
        self.providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(self.model_path):
            self._export_initial_model()
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=self.providers)
            self.input_name = self.session.get_inputs()[0].name
            logger.info("Loaded ONNX Inference Session for %s", model_path)
        except Exception as e:
            logger.error("Failed to load ONNX model: %s", e)

    def _export_initial_model(self):
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping ONNX model generation.")
            return
        logger.info("Exporting initial PyTorch Transformer to ONNX: %s", self.model_path)
        model = TransformerFeatureExtractor(input_dim=self.input_dim)
        model.eval()
        dummy_input = torch.randn(1, self.input_dim)
        torch.onnx.export(
            model, dummy_input, self.model_path,
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=['features'], output_names=['signal'],
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
        try:
            expected_shape = self.session.get_inputs()[0].shape
            if len(expected_shape) == 3:
                input_data = np.expand_dims(input_data, axis=1)
            ort_outs = self.session.run(None, {self.input_name: input_data})
            out = ort_outs[0]
            pred = float(np.ravel(out)[0])
            return float(np.clip(pred, -1.0, 1.0))
        except Exception as e:
            logger.error("ONNX prediction failed: %s", e)
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
            logger.error("ONNX batch prediction failed: %s", e)
            return np.zeros(len(features_df))