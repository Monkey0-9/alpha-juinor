import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerBrain(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, output_dim=1):
        super(TransformerBrain, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Output shape: (batch_size, seq_len, d_model)
        out = self.transformer_encoder(x)
        
        # Take the output of the last sequence step
        last_out = out[:, -1, :]
        
        res = self.fc(last_out)
        res = self.tanh(res) # signal -1 to 1
        return res
