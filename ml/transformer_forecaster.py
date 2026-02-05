"""
Temporal Fusion Transformer for Time Series Forecasting
========================================================

State-of-the-art attention-based forecasting model.

Based on:
- Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Vaswani, A., et al. (2017). "Attention Is All You Need"

Features:
- Multi-horizon forecasting
- Interpretable attention mechanisms
- Variable selection
- Quantile regression for uncertainty
- Static and time-varying covariates
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    lookback: int = 60
    forecast_horizon: int = 5
    quantiles: List[float] = None

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VariableSelectionNetwork(nn.Module):
    """Variable selection with gating mechanism."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.grn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.grn(x)
        return x * weights, weights


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) for feature transformation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, use_context: bool = False):
        super().__init__()

        self.use_context = use_context

        self.fc1 = nn.Linear(input_dim + (hidden_dim if use_context else 0), hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.gate_fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.layer_norm = nn.LayerNorm(output_dim)

        # Skip connection
        if input_dim != output_dim:
            self.skip_layer = nn.Linear(input_dim, output_dim)
        else:
            self.skip_layer = None

    def forward(self, x, context=None):
        # Concatenate context if provided
        if self.use_context and context is not None:
            x_with_context = torch.cat([x, context], dim=-1)
        else:
            x_with_context = x

        # GRN transformation
        a = self.fc1(x_with_context)
        a = self.elu(a)
        a = self.dropout(a)
        a = self.fc2(a)

        # Gating
        gate = self.sigmoid(self.gate_fc(self.elu(self.fc1(x_with_context))))
        gated = a * gate

        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x

        # Layer norm
        out = self.layer_norm(gated + skip)

        return out


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon forecasting.

    Architecture:
    1. Variable selection for static and temporal features
    2. LSTM encoder for sequential processing
    3. Multi-head self-attention
    4. Gated residual networks
    5. Quantile output heads for uncertainty
    """

    def __init__(self, config: TransformerConfig, n_features: int):
        super().__init__()

        self.config = config
        self.n_features = n_features

        # Variable selection
        self.variable_selection = VariableSelectionNetwork(
            n_features,
            config.d_model,
            config.dropout
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model)

        # LSTM for sequential encoding
        self.lstm = nn.LSTM(
            config.d_model,
            config.d_model,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Gated residual networks
        self.grn1 = GatedResidualNetwork(config.d_model, config.d_model * 2, config.d_model, config.dropout)
        self.grn2 = GatedResidualNetwork(config.d_model, config.d_model * 2, config.d_model, config.dropout)

        # Output heads for each quantile
        self.output_heads = nn.ModuleList([
            nn.Linear(config.d_model, config.forecast_horizon)
            for _ in config.quantiles
        ])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            predictions: Dict of quantile predictions
        """
        batch_size, seq_len, _ = x.shape

        # Variable selection
        x_selected, var_weights = self.variable_selection(x)

        # Project to d_model
        if x_selected.size(-1) != self.config.d_model:
            projection = nn.Linear(x_selected.size(-1), self.config.d_model).to(x.device)
            x_proj = projection(x_selected)
        else:
            x_proj = x_selected

        # Positional encoding
        x_pos = self.pos_encoder(x_proj)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x_pos)
        lstm_out = self.dropout(lstm_out)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)

        # Residual connection
        x_attn = lstm_out + attn_out

        # Gated residual networks
        x_grn1 = self.grn1(x_attn)
        x_grn2 = self.grn2(x_grn1)

        # Take last timestep for forecasting
        x_final = x_grn2[:, -1, :]

        # Quantile outputs
        predictions = {}
        for i, q in enumerate(self.config.quantiles):
            predictions[f'q{int(q*100)}'] = self.output_heads[i](x_final)

        return predictions, attn_weights, var_weights


class TransformerForecaster:
    """
    Wrapper class for Temporal Fusion Transformer with training and inference.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"TransformerForecaster initialized on {self.device}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the transformer model.

        Args:
            X_train: Training features [n_samples, lookback, n_features]
            y_train: Training targets [n_samples, forecast_horizon]
            X_val: Validation features
            y_val: Validation targets
        """
        # Normalize
        from sklearn.preprocessing import StandardScaler

        n_samples, lookback, n_features = X_train.shape

        # Flatten for scaling
        X_flat = X_train.reshape(-1, n_features)
        self.scaler_X = StandardScaler()
        X_scaled_flat = self.scaler_X.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n_samples, lookback, n_features)

        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y_train)

        # Initialize model
        self.model = TemporalFusionTransformer(self.config, n_features).to(self.device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            # Training
            self.model.train()
            train_loss = self._train_epoch(X_scaled, y_scaled, optimizer)

            # Validation
            if X_val is not None and y_val is not None:
                X_val_scaled = self._scale_X(X_val)
                y_val_scaled = self.scaler_y.transform(y_val)
                val_loss = self._validate(X_val_scaled, y_val_scaled)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}")

        logger.info("Training complete")

    def predict(self, X: np.ndarray, return_uncertainty: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty quantification.

        Args:
            X: Input features [n_samples, lookback, n_features]
            return_uncertainty: Whether to return quantiles

        Returns:
            Dictionary with predictions and uncertainty
        """
        self.model.eval()

        X_scaled = self._scale_X(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            predictions, attn_weights, var_weights = self.model(X_tensor)

        # Convert to numpy and inverse transform
        results = {}
        for key, pred in predictions.items():
            pred_np = pred.cpu().numpy()
            pred_original = self.scaler_y.inverse_transform(pred_np)
            results[key] = pred_original

        if return_uncertainty:
            # Median prediction
            results['mean'] = results['q50']

            # Uncertainty bounds
            results['lower'] = results['q10']
            results['upper'] = results['q90']

            # Predictive std (from quantiles)
            results['std'] = (results['q90'] - results['q10']) / 2.56  # ~90% CI

        results['attention_weights'] = attn_weights.cpu().numpy()
        results['variable_importance'] = var_weights.cpu().numpy()

        return results

    def _train_epoch(self, X, y, optimizer):
        """Train one epoch."""
        total_loss = 0
        n_batches = len(X) // self.config.batch_size

        for i in range(n_batches):
            start_idx = i * self.config.batch_size
            end_idx = start_idx + self.config.batch_size

            X_batch = torch.FloatTensor(X[start_idx:end_idx]).to(self.device)
            y_batch = torch.FloatTensor(y[start_idx:end_idx]).to(self.device)

            optimizer.zero_grad()

            predictions, _, _ = self.model(X_batch)

            # Quantile loss
            loss = self._quantile_loss(predictions, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / n_batches

    def _validate(self, X, y):
        """Validate model."""
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        with torch.no_grad():
            predictions, _, _ = self.model(X_tensor)
            loss = self._quantile_loss(predictions, y_tensor)

        return loss.item()

    def _quantile_loss(self, predictions, targets):
        """Quantile regression loss."""
        total_loss = 0

        for i, q in enumerate(self.config.quantiles):
            pred = predictions[f'q{int(q*100)}']
            error = targets - pred
            loss = torch.max((q - 1) * error, q * error)
            total_loss += loss.mean()

        return total_loss / len(self.config.quantiles)

    def _scale_X(self, X):
        """Scale input features."""
        n_samples, lookback, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled_flat = self.scaler_X.transform(X_flat)
        return X_scaled_flat.reshape(n_samples, lookback, n_features)

    def save(self, path: str):
        """Save model."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']

        # Reconstruct model
        n_features = self.scaler_X.n_features_in_
        self.model = TemporalFusionTransformer(self.config, n_features).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])

        logger.info(f"Model loaded from {path}")
