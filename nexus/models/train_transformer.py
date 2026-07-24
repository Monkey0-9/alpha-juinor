import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from nexus.data.historical_data_loader import HistoricalDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training Transformer on device: {device}")

class MultiAssetTransformer(nn.Module):
    def __init__(self, n_assets=10, d_model=128, nhead=8, num_layers=4, seq_len=60):
        super().__init__()
        self.n_assets = n_assets
        self.seq_len = seq_len
        self.input_proj = nn.Linear(5, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.asset_embed = nn.Parameter(torch.randn(1, n_assets, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=512, dropout=0.1,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_assets)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoder
        x = x + self.asset_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.fc_out(x)
        return torch.tanh(out)

def prepare_transformer_features(symbols, seq_len=60):
    loader = HistoricalDataLoader()
    all_returns = []
    all_features = []

    for sym in symbols[:10]:
        df = loader.load_macro_data(sym)
        if df.empty or len(df) < seq_len + 10:
            continue
        returns = df["returns"].values
        close = df["close"].values
        vol = df["vol_30d"].values if "vol_30d" in df.columns else np.zeros_like(returns)

        features = np.column_stack([
            close / np.mean(close[-100:]) - 1,
            returns,
            np.where(np.isnan(returns), 0, returns),
            vol,
            np.where(df["regime"].values if "regime" in df.columns else 0, 1, -1)
        ])
        all_returns.append(returns)
        all_features.append(features)

    n_assets = len(all_features)
    if n_assets < 3:
        logger.warning(f"Only {n_assets} assets with sufficient data")
        return np.array([]), np.array([])

    min_len = min(len(f) for f in all_features)
    X_list, y_list = [], []

    for i in range(seq_len, min_len - 5):
        sample = np.zeros((n_assets, seq_len, 5), dtype=np.float32)
        targets = np.zeros(n_assets, dtype=np.float32)
        valid = True
        for a in range(n_assets):
            f = all_features[a]
            if i > len(f) - 1:
                valid = False
                break
            sample[a] = f[i - seq_len:i, :5]
            targets[a] = np.tanh(np.nansum(all_returns[a][i:i+5]) * 20)
        if valid:
            X_list.append(sample)
            y_list.append(targets)

    if not X_list:
        return np.array([]), np.array([])
    return np.array(X_list), np.array(y_list)

def train_transformer():
    symbols = ["^GSPC", "^DJI", "^IXIC", "^RUT", "SPY", "QQQ", "IWM", "TLT", "GLD", "XLF"]
    X, y = prepare_transformer_features(symbols)
    if X.size == 0:
        logger.error("No training data prepared")
        return

    n_assets = X.shape[1]
    logger.info(f"Prepared {len(X)} samples for {n_assets} assets, shape={X.shape}")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train).float()
    y_train_t = torch.tensor(y_train).float()
    X_test_t = torch.tensor(X_test).float()
    y_test_t = torch.tensor(y_test).float()

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    model = MultiAssetTransformer(n_assets=n_assets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(80):
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
            test_loss = criterion(test_pred, y_test_t.to(device)).item()

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "nexus/models/transformer_multi_asset.pt")
            logger.info(f"Epoch {epoch}: test_loss={test_loss:.6f} [SAVED]")

    logger.info(f"Transformer training complete. Best loss: {best_loss:.6f}")

    model.eval()
    with torch.no_grad():
        test_pred_np = model(X_test_t.to(device)).cpu().numpy()
    for a in range(n_assets):
        ic = np.corrcoef(test_pred_np[:, a], y_test[:, a])[0, 1]
        logger.info(f"Asset {a} IC: {ic:.4f}")

    dummy_input = torch.randn(1, n_assets, X.shape[2], 5).to(device)
    torch.onnx.export(model, dummy_input, "nexus/models/transformer_multi_asset.onnx",
                     input_names=["multi_asset_sequence"], output_names=["asset_scores"],
                     dynamic_axes={"multi_asset_sequence": {0: "batch"}},
                     opset_version=17)
    logger.info("Transformer exported to ONNX")

if __name__ == "__main__":
    train_transformer()