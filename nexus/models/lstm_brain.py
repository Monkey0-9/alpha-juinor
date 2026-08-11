import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        scores = self.attn(lstm_outputs)
        weights = F.softmax(scores, dim=1)
        context = (weights * lstm_outputs).sum(dim=1)
        return context, weights

class LSTMBrain(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=3, output_dim=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            proj_size=0,
        )
        lstm_out_dim = hidden_dim * 2
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.attention = LSTMAttention(lstm_out_dim)
        self.residual_proj = nn.Linear(input_dim, lstm_out_dim) if input_dim != lstm_out_dim else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.LayerNorm(lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, lstm_out_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 4, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        context, _ = self.attention(lstm_out)
        residual = self.residual_proj(x.mean(dim=1))
        combined = context + residual
        return self.fc(combined).squeeze(-1)