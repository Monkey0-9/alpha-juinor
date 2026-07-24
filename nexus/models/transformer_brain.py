import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable 1D Positional Embeddings with LayerNorm and Residual Connection.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = self.pos_embedding[:, :seq_len, :]
        x = self.layer_norm(x + pos)
        return self.dropout(x)

class TransformerBrain(nn.Module):
    """
    Institutional Transformer Encoder for Temporal Signal Processing.
    Includes Residual Skip Connections, LayerNorm, Learnable Positional Encodings,
    and Gradient Checkpointing for memory-efficient deep sequence modeling.
    """
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
        use_checkpointing: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_residual = nn.Linear(d_model, d_model)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        embedded = self.input_layer(x) * math.sqrt(self.d_model)
        pos_enc = self.pos_encoder(embedded)

        if self.use_checkpointing and self.training:
            out = checkpoint.checkpoint(self.transformer_encoder, pos_enc)
        else:
            out = self.transformer_encoder(pos_enc)

        # Residual connection from input embedding to transformer output
        res = self.norm(out + embedded)
        last_step = res[:, -1, :]
        
        # Dense projection with residual connection
        dense_res = last_step + self.fc_residual(last_step)
        signal = self.fc_out(dense_res)
        return signal
