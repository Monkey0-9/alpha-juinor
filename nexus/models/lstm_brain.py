import torch.nn as nn

class LSTMBrain(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMBrain, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Check if x is missing batch dim (e.g., during ONNX inference test)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.tanh(out) # Constrain output to [-1, 1] for signal strength
        return out
