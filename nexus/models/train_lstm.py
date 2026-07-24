import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from nexus.data.historical_data_loader import HistoricalDataLoader
from nexus.core.strategies import StrategyFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training LSTM on device: {device}")

class PriceLSTM(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)
        out = self.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = torch.tanh(self.fc2(out))
        return out.squeeze()

def prepare_lstm_features(df, seq_len=60):
    strategies = StrategyFactory.all_strategies()
    features = []
    targets = []
    prices = df["close"].values
    returns = df["returns"].values

    for i in range(seq_len, len(df) - 5):
        window = df.iloc[i - seq_len:i]
        feature_seq = []
        for j in range(seq_len):
            row_features = []
            p = prices[i - seq_len + j]
            row_features.append(p)
            row_features.append(returns[i - seq_len + j] if not np.isnan(returns[i - seq_len + j]) else 0.0)
            vol = df["vol_30d"].iloc[i - seq_len + j] if "vol_30d" in df.columns else 0.0
            row_features.append(vol if not np.isnan(vol) else 0.0)
            regime_val = df["regime"].iloc[i - seq_len + j] if "regime" in df.columns else 0
            row_features.append(regime_val)

            for strat in strategies:
                s = strat.score("TRAIN", 0.0, window.iloc[:j+1], "SIDEWAYS")
                row_features.append(float(s))

            n_expected = 4 + len(strategies)
            while len(row_features) < n_expected:
                row_features.append(0.0)
            feature_seq.append(row_features[:n_expected])

        features.append(feature_seq)
        future_ret = np.sum(returns[i:i+5])
        targets.append(np.tanh(future_ret * 20))

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

def train_lstm():
    loader = HistoricalDataLoader()
    df = loader.load_macro_data("^GSPC")
    if df.empty or len(df) < 500:
        logger.error("Insufficient data")
        return

    logger.info(f"Loaded {len(df)} days from {df.index.min().date()} to {df.index.max().date()}")
    X, y = prepare_lstm_features(df)
    logger.info(f"Prepared {len(X)} samples, shape={X.shape}")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train).float()
    y_train_t = torch.tensor(y_train).float()
    X_test_t = torch.tensor(X_test).float()
    y_test_t = torch.tensor(y_test).float()

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    model = PriceLSTM(input_size=X.shape[2], hidden_size=128, num_layers=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t.to(device))
            test_loss = criterion(test_pred, y_test_t.to(device))

        if test_loss < best_loss:
            best_loss = test_loss
            model_path = "nexus/models/lstm_price_model.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Epoch {epoch}: train_loss={total_loss/len(train_loader):.6f}, test_loss={test_loss:.6f} [SAVED]")

    logger.info(f"LSTM training complete. Best test loss: {best_loss:.6f}")

    model.eval()
    with torch.no_grad():
        test_pred_np = model(X_test_t.to(device)).cpu().numpy()
    ic = np.corrcoef(test_pred_np.flatten(), y_test)[0, 1]
    logger.info(f"Out-of-sample IC: {ic:.4f}")

    dummy_input = torch.randn(1, X.shape[1], X.shape[2]).to(device)
    torch.onnx.export(model, dummy_input, "nexus/models/lstm_price_model.onnx",
                     input_names=["sequence"], output_names=["score"],
                     dynamic_axes={"sequence": {0: "batch"}},
                     opset_version=17)
    logger.info("LSTM exported to ONNX")

if __name__ == "__main__":
    train_lstm()