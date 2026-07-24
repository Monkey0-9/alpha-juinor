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
    Advanced Machine Learning Brain utilizing XGBoost (C++ accelerated under the hood).
    Trains on historical signals and macro regimes to predict 1-minute to daily returns.
    """
    
    def __init__(self, model_path: str = "models/xgboost_brain.json"):
        self.model_path = model_path
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,  # use all cores
            random_state=42
        )
        self.is_trained = False
        
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        if os.path.exists(self.model_path):
            try:
                self.model.load_model(self.model_path)
                self.is_trained = True
                logger.info("Loaded pre-trained XGBoost Brain.")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")
                
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """
        Train the model on historical data. 
        Features should include macro regime, alpha, momentum, quality.
        """
        if features.empty or targets.empty:
            logger.warning("Empty data provided for ML training.")
            return
            
        logger.info(f"Training XGBoost ML Brain on {len(features)} samples...")
        self.model.fit(features, targets)
        self.is_trained = True
        
        try:
            self.model.save_model(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            
    def predict(self, current_features: Dict[str, float]) -> float:
        """
        Predict the future return / alpha score for a single asset given its features.
        """
        if not self.is_trained:
            # Fallback to simple mean if not trained
            return np.mean(list(current_features.values())) if current_features else 0.0
            
        # Convert to DataFrame for XGBoost
        df = pd.DataFrame([current_features])
        try:
            pred = self.model.predict(df)[0]
            # Clip output to [-1.0, 1.0] for signal bounds
            return float(np.clip(pred, -1.0, 1.0))
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0

    def batch_predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict multiple assets at once.
        """
        if not self.is_trained or features_df.empty:
            return np.zeros(len(features_df))
            
        return np.clip(self.model.predict(features_df), -1.0, 1.0)


# ------------------------------------------------------------------ #
# ONNX Deep Learning Brain (C++ Accelerated Inference)                 #
# ------------------------------------------------------------------ #

if TORCH_AVAILABLE:
    BaseModule = nn.Module
else:
    class BaseModule:
        pass

class TransformerFeatureExtractor(BaseModule):
    """
    Simple Transformer for temporal sequence embedding.
    """
    def __init__(self, input_dim: int, embed_dim: int=32, num_heads: int=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=64, batch_first=True
        )
        self.fc = nn.Linear(embed_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch_size, input_dim) -> unsqueeze to (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return self.tanh(x)


class ONNXBrain:
    """
    Advanced Deep Learning Brain utilizing ONNX Runtime for C++ execution speeds.
    Loads Transformer/LSTM models exported from PyTorch.
    """
    def __init__(self, model_path: str = "models/transformer_brain.onnx", input_dim: int = 15):
        self.model_path = model_path
        self.input_dim = input_dim
        self.session = None
        
        # Configure C++ execution provider (CPU or CUDA)
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
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
