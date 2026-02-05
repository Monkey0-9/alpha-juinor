"""
Deep Learning Ensemble - LSTM + Transformer + CNN.

Ensemble of deep learning models for price prediction:
- LSTM for sequence modeling
- Transformer attention for regime detection
- CNN for pattern recognition
- Meta-learner for ensemble weighting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Ensemble model prediction."""
    symbol: str
    prediction: float  # -1 to 1
    confidence: float

    # Individual model predictions
    lstm_pred: float
    transformer_pred: float
    cnn_pred: float

    # Model weights
    model_weights: Dict[str, float]


class LSTMModel:
    """
    Simple LSTM implementation (NumPy only).

    For production, this would use PyTorch/TensorFlow.
    This is a simplified version for demonstration.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 32,
        output_dim: int = 1,
        sequence_length: int = 20
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length

        # Initialize weights (simplified)
        scale = 0.1
        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wc = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wy = np.random.randn(hidden_dim, output_dim) * scale

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X: np.ndarray) -> float:
        """Forward pass through LSTM."""
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        for t in range(min(len(X), self.sequence_length)):
            x = X[t]
            if len(x) != self.input_dim:
                x = np.zeros(self.input_dim)
                x[:min(len(X[t]), self.input_dim)] = X[t][:self.input_dim]

            concat = np.concatenate([x, h])

            f = self.sigmoid(concat @ self.Wf)
            i = self.sigmoid(concat @ self.Wi)
            c_tilde = np.tanh(concat @ self.Wc)
            o = self.sigmoid(concat @ self.Wo)

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

        y = h @ self.Wy
        return float(np.tanh(y[0]))  # Bounded output


class TransformerModel:
    """
    Simplified Transformer attention (NumPy only).
    """

    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 32,
        n_heads: int = 4,
        sequence_length: int = 20
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.sequence_length = sequence_length
        self.d_k = d_model // n_heads

        scale = 0.1
        self.Wq = np.random.randn(input_dim, d_model) * scale
        self.Wk = np.random.randn(input_dim, d_model) * scale
        self.Wv = np.random.randn(input_dim, d_model) * scale
        self.Wo = np.random.randn(d_model, 1) * scale

    def attention(self, Q, K, V):
        """Scaled dot-product attention."""
        scores = Q @ K.T / np.sqrt(self.d_k)
        weights = np.exp(scores - np.max(scores))
        weights = weights / (weights.sum() + 1e-10)
        return weights @ V

    def forward(self, X: np.ndarray) -> float:
        """Forward pass."""
        seq_len = min(len(X), self.sequence_length)

        # Pad/truncate input
        X_padded = np.zeros((seq_len, self.input_dim))
        for i in range(seq_len):
            if i < len(X):
                X_padded[i, :min(len(X[i]), self.input_dim)] = X[i][:self.input_dim]

        Q = X_padded @ self.Wq
        K = X_padded @ self.Wk
        V = X_padded @ self.Wv

        attn_out = self.attention(Q, K, V)

        # Pool and output
        pooled = attn_out.mean(axis=0)
        y = pooled @ self.Wo

        return float(np.tanh(y[0]))


class CNNModel:
    """
    Simple 1D CNN for pattern detection (NumPy only).
    """

    def __init__(
        self,
        input_dim: int = 10,
        n_filters: int = 16,
        kernel_size: int = 3,
        sequence_length: int = 20
    ):
        self.input_dim = input_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sequence_length = sequence_length

        scale = 0.1
        self.conv_weights = np.random.randn(
            n_filters, input_dim, kernel_size
        ) * scale
        self.fc_weights = np.random.randn(n_filters, 1) * scale

    def forward(self, X: np.ndarray) -> float:
        """Forward pass."""
        seq_len = min(len(X), self.sequence_length)

        # Pad input
        X_padded = np.zeros((seq_len, self.input_dim))
        for i in range(seq_len):
            if i < len(X):
                X_padded[i, :min(len(X[i]), self.input_dim)] = X[i][:self.input_dim]

        # Convolution
        conv_out = np.zeros((seq_len - self.kernel_size + 1, self.n_filters))
        for i in range(seq_len - self.kernel_size + 1):
            for f in range(self.n_filters):
                window = X_padded[i:i+self.kernel_size]
                conv_out[i, f] = np.sum(window * self.conv_weights[f].T)

        # ReLU and global max pooling
        conv_out = np.maximum(0, conv_out)
        pooled = conv_out.max(axis=0)

        # Output
        y = pooled @ self.fc_weights
        return float(np.tanh(y[0]))


class DeepEnsemble:
    """
    Ensemble of LSTM, Transformer, and CNN models.

    Uses simple averaging with performance-based weighting.
    """

    def __init__(
        self,
        input_dim: int = 10,
        sequence_length: int = 20
    ):
        self.input_dim = input_dim
        self.sequence_length = sequence_length

        # Initialize models
        self.lstm = LSTMModel(input_dim, sequence_length=sequence_length)
        self.transformer = TransformerModel(input_dim, sequence_length=sequence_length)
        self.cnn = CNNModel(input_dim, sequence_length=sequence_length)

        # Model weights (performance-based, starts equal)
        self.model_weights = {
            "lstm": 0.33,
            "transformer": 0.34,
            "cnn": 0.33
        }

        # Track model performance
        self.model_errors: Dict[str, List[float]] = {
            "lstm": [],
            "transformer": [],
            "cnn": []
        }

        # MC Dropout settings for uncertainty quantification
        self.mc_samples = 30
        self.dropout_rate = 0.2

    def predict_with_uncertainty(
        self,
        features: np.ndarray,
        symbol: str = "UNKNOWN",
        n_samples: int = None
    ) -> Dict[str, any]:
        """
        Generate ensemble prediction with full uncertainty quantification.

        Returns both epistemic (model) and aleatoric (data) uncertainty using
        Monte Carlo Dropout and ensemble disagreement.

        Args:
            features: Input features
            symbol: Symbol name
            n_samples: Number of MC samples (default: self.mc_samples)

        Returns:
            Dictionary with:
            - mean: Expected prediction
            - epistemic_std: Model uncertainty (reducible with more data)
            - aleatoric_std: Data uncertainty (irreducible)
            - total_std: Combined uncertainty
            - confidence_interval: [lower, upper] 95% CI
            - ensemble_prediction: Full EnsemblePrediction object
        """
        if n_samples is None:
            n_samples = self.mc_samples

        # Collect MC samples
        predictions = []

        for _ in range(n_samples):
            # Apply dropout to features (MC Dropout)
            if np.random.rand() > self.dropout_rate:
                features_dropout = features * (np.random.rand(*features.shape) > self.dropout_rate)
            else:
                features_dropout = features

            # Get individual predictions with dropout
            lstm_pred = self.lstm.forward(features_dropout)
            transformer_pred = self.transformer.forward(features_dropout)
            cnn_pred = self.cnn.forward(features_dropout)

            # Weighted ensemble
            ensemble_pred = (
                self.model_weights["lstm"] * lstm_pred +
                self.model_weights["transformer"] * transformer_pred +
                self.model_weights["cnn"] * cnn_pred
            )

            predictions.append(ensemble_pred)

        predictions = np.array(predictions)

        # Calculate uncertainties
        mean_pred = float(np.mean(predictions))
        total_std = float(np.std(predictions))

        # Epistemic uncertainty (model disagreement)
        base_pred = self.predict(features, symbol)
        model_preds = np.array([
            base_pred.lstm_pred,
            base_pred.transformer_pred,
            base_pred.cnn_pred
        ])
        epistemic_std = float(np.std(model_preds))

        # Aleatoric uncertainty (data noise)
        # Approximate as residual after accounting for epistemic
        aleatoric_std = float(np.sqrt(max(0, total_std**2 - epistemic_std**2)))

        # Confidence interval (95%)
        ci_lower = mean_pred - 1.96 * total_std
        ci_upper = mean_pred + 1.96 * total_std

        return {
            'mean': mean_pred,
            'std': total_std,
            'epistemic_std': epistemic_std,
            'aleatoric_std': aleatoric_std,
            'confidence_interval': (ci_lower, ci_upper),
            'sharpe_ratio': abs(mean_pred) / (total_std + 1e-10),
            'ensemble_prediction': base_pred,
            'mc_samples': n_samples
        }


    def predict(
        self,
        features: np.ndarray,
        symbol: str = "UNKNOWN"
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction.
        """
        # Get individual predictions
        lstm_pred = self.lstm.forward(features)
        transformer_pred = self.transformer.forward(features)
        cnn_pred = self.cnn.forward(features)

        # Weighted ensemble
        ensemble_pred = (
            self.model_weights["lstm"] * lstm_pred +
            self.model_weights["transformer"] * transformer_pred +
            self.model_weights["cnn"] * cnn_pred
        )

        # Confidence based on agreement
        preds = [lstm_pred, transformer_pred, cnn_pred]
        agreement = 1 - np.std(preds)  # Higher agreement = higher confidence
        confidence = float(np.clip(agreement, 0, 1))

        return EnsemblePrediction(
            symbol=symbol,
            prediction=float(ensemble_pred),
            confidence=confidence,
            lstm_pred=lstm_pred,
            transformer_pred=transformer_pred,
            cnn_pred=cnn_pred,
            model_weights=self.model_weights.copy()
        )

    def update_weights(
        self,
        actual_return: float,
        prediction: EnsemblePrediction
    ):
        """Update model weights based on prediction accuracy."""
        # Calculate errors
        errors = {
            "lstm": abs(prediction.lstm_pred - np.sign(actual_return)),
            "transformer": abs(prediction.transformer_pred - np.sign(actual_return)),
            "cnn": abs(prediction.cnn_pred - np.sign(actual_return))
        }

        for model, error in errors.items():
            self.model_errors[model].append(error)
            if len(self.model_errors[model]) > 100:
                self.model_errors[model] = self.model_errors[model][-100:]

        # Update weights inversely proportional to error
        if all(len(v) >= 10 for v in self.model_errors.values()):
            avg_errors = {
                k: np.mean(v) for k, v in self.model_errors.items()
            }
            inv_errors = {k: 1 / (v + 0.01) for k, v in avg_errors.items()}
            total = sum(inv_errors.values())
            self.model_weights = {k: v / total for k, v in inv_errors.items()}

    def get_signal(
        self,
        prediction: EnsemblePrediction
    ) -> Tuple[str, float]:
        """Convert prediction to trading signal."""
        pred = prediction.prediction
        conf = prediction.confidence

        if pred > 0.2 and conf > 0.3:
            return "BUY", conf
        elif pred < -0.2 and conf > 0.3:
            return "SELL", conf
        else:
            return "HOLD", conf


# Global singleton
_deep_ensemble: Optional[DeepEnsemble] = None


def get_deep_ensemble() -> DeepEnsemble:
    """Get or create global deep ensemble."""
    global _deep_ensemble
    if _deep_ensemble is None:
        _deep_ensemble = DeepEnsemble()
    return _deep_ensemble
