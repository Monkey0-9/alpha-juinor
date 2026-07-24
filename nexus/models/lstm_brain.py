import torch
import torch.nn as nn

class LSTMBrain(nn.Module):
    """
    Institutional Deep Bidirectional LSTM with Multi-Head Self-Attention.
    Includes Residual Skip Connections, LayerNorm, and Cell State Projection.
    """
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        nhead: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim * self.num_directions)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        # Multi-Head Self-Attention over LSTM sequence outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * self.num_directions,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(hidden_dim * self.num_directions)

        # Peephole / Cell State Projection layer
        self.cell_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # Output projection head
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        norm_x = self.input_norm(x)
        input_residual = self.input_proj(norm_x)

        # Forward pass through Bidirectional LSTM
        lstm_out, (h_n, c_n) = self.lstm(norm_x)

        # Multi-Head Self-Attention over sequence
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection + LayerNorm
        res_seq = self.attn_norm(attn_out + lstm_out + input_residual)

        # Extract last time-step and pool with cell state projection
        last_step = res_seq[:, -1, :]
        features = self.cell_proj(last_step)

        # Bounded Signal Prediction in [-1.0, 1.0]
        signal = self.fc_head(features)
        return signal
