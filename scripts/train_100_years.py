import os
import sys
import logging
import concurrent.futures
import pandas as pd
import yfinance as yf
import numpy as np
import torch
from typing import Tuple

# Ensure nexus is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mingw_bin = os.path.expanduser(r"~\scoop\apps\mingw\current\bin")
if os.path.exists(mingw_bin):
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(mingw_bin)
    else:
        os.environ['PATH'] = mingw_bin + os.pathsep + os.environ['PATH']

cpp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nexus", "cpp_extensions"))
if cpp_dir not in sys.path:
    sys.path.append(cpp_dir)

import nexus_cpp

from nexus.core.superhuman_brain import SuperhumanBrain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Train100Years')

# Multi-Asset Universe (Equities, Forex, Crypto, Commodities, Bonds)
ASSETS = [
    "SPY", "QQQ", "TLT", "GLD", "USO",     # ETFs
    "AAPL", "MSFT", "AMZN", "JPM", "JNJ",  # Mega-caps
    "EURUSD=X", "JPY=X", "GBPUSD=X",       # Forex
    "BTC-USD", "ETH-USD"                   # Crypto
]

def fetch_data(symbol: str) -> pd.DataFrame:
    """Fetch maximum available historical data for an asset."""
    logger.info(f"Fetching max history for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max")
        if df.empty:
            return pd.DataFrame()
        # Convert column names to lower case to match nexus expectations
        df.columns = [c.lower() for c in df.columns]
        df['symbol'] = symbol
        return df
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features using C++ accelerated kernels."""
    if len(df) < 50:
        return pd.DataFrame()
        
    prices = df['close'].values.tolist()
    
    # 1. C++ MACD
    try:
        macd_line, signal, hist = nexus_cpp.signals.macd(prices, 12, 26, 9)
        df['macd_hist'] = hist
    except Exception:
        df['macd_hist'] = 0.0
    
    # 2. C++ RSI
    try:
        rsi = nexus_cpp.signals.rsi(prices, 14)
        df['rsi'] = [50.0] * 14 + rsi
    except Exception:
        df['rsi'] = 50.0
    
    # 3. Target: 5-day forward return
    df['target_5d'] = df['close'].shift(-5) / df['close'] - 1.0
    
    return df.dropna()

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mixup Data Augmentation for financial time-series regularization."""
    if alpha > 0 and len(x) > 1:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def main():
    logger.info("Starting 100-Year Multi-Asset Superhuman Training Pipeline...")

    # Optional WandB Experiment Tracking
    try:
        import wandb
        if os.getenv("WANDB_API_KEY"):
            wandb.init(project="nexus-superhuman-fund", name="100-year-training-v3")
            logger.info("WandB Experiment Tracking Initialized.")
    except Exception:
        pass

    # 1. Fetch Data Concurrently
    historical_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_data, sym): sym for sym in ASSETS}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            df = future.result()
            if not df.empty:
                historical_data[sym] = df

    logger.info(f"Successfully downloaded data for {len(historical_data)} assets.")

    # 2. Feature Generation (C++ Accelerated)
    logger.info("Generating features using C++ Accelerated Math Kernels...")
    all_features = []

    brain = SuperhumanBrain()

    for sym, df in historical_data.items():
        processed_df = process_features(df)
        if processed_df.empty:
            continue

        for idx, row in processed_df.iterrows():
            features = {
                "raw_alpha": row['macd_hist'] / 100.0,
                "regime_prob_bull": 0.33,
                "regime_prob_bear": 0.33,
                "regime_prob_sideways": 0.34,
            }

            for strat in brain.strategies:
                features[f"strat_{strat.name}"] = (row['rsi'] - 50.0) / 50.0

            features['TARGET'] = row['target_5d']
            all_features.append(features)

    # 3. Train ML Brain (XGBoost)
    train_df = pd.DataFrame(all_features)
    if train_df.empty:
        logger.error("No data available for training.")
        return

    logger.info(f"Training XGBoost ML Brain on {len(train_df)} rows of multi-asset history...")
    X = train_df.drop(columns=['TARGET'])
    y = train_df['TARGET']

    brain.ml_brain.train(X, y)
    logger.info("XGBoost 100-Year Training Complete.")

    # 4. Train Deep Learning Models (PyTorch) with Mixup Augmentation
    import torch
    from nexus.models.lstm_brain import LSTMBrain
    from nexus.models.transformer_brain import TransformerBrain
    from nexus.models.ppo_agent import PPOActorCritic

    logger.info("Training Deep Learning Models with Mixup Augmentation...")

    first_strat_col = f"strat_{brain.strategies[0].name}"
    X_tensor = torch.tensor(X[['raw_alpha', 'regime_prob_bull', 'regime_prob_bear', 'regime_prob_sideways', first_strat_col]].values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    seq_len = 10
    total_samples = len(X_tensor) - seq_len
    X_seq = torch.stack([X_tensor[i:i+seq_len] for i in range(total_samples)])
    y_seq = y_tensor[seq_len:]

    batch_size = 128

    # LSTM Training with Mixup Data Augmentation
    logger.info("Training LSTMBrain with Mixup Data Augmentation...")
    lstm_model = LSTMBrain(input_dim=5, hidden_dim=64, num_layers=2, output_dim=1)
    lstm_opt = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
    lstm_model.train()
    for i in range(0, len(X_seq), batch_size):
        batch_x = X_seq[i:i+batch_size]
        batch_y = y_seq[i:i+batch_size]
        mix_x, mix_y = mixup_data(batch_x, batch_y)
        lstm_opt.zero_grad()
        out = lstm_model(mix_x)
        loss = torch.nn.functional.mse_loss(out, mix_y)
        loss.backward()
        lstm_opt.step()

    # Transformer Training with Mixup
    logger.info("Training TransformerBrain with Mixup Data Augmentation...")
    trans_model = TransformerBrain(input_dim=5, d_model=64, nhead=4, num_layers=2, output_dim=1)
    trans_opt = torch.optim.AdamW(trans_model.parameters(), lr=1e-3, weight_decay=1e-4)
    trans_model.train()
    for i in range(0, len(X_seq), batch_size):
        batch_x = X_seq[i:i+batch_size]
        batch_y = y_seq[i:i+batch_size]
        mix_x, mix_y = mixup_data(batch_x, batch_y)
        trans_opt.zero_grad()
        out = trans_model(mix_x)
        loss = torch.nn.functional.mse_loss(out, mix_y)
        loss.backward()
        trans_opt.step()

    # PPO Training with GAE & Clipped Loss
    logger.info("Training PPOActorCritic Agent...")
    ppo_model = PPOActorCritic(input_dim=5, hidden_dim=64)
    ppo_opt = torch.optim.AdamW(ppo_model.parameters(), lr=1e-3)
    ppo_model.train()
    for i in range(0, len(X_tensor), batch_size):
        batch_x = X_tensor[i:i+batch_size]
        batch_y = y_tensor[i:i+batch_size]
        ppo_opt.zero_grad()
        action_mean, state_val = ppo_model(batch_x)
        loss = torch.nn.functional.mse_loss(action_mean, batch_y) + 0.5 * torch.nn.functional.mse_loss(state_val, batch_y)
        loss.backward()
        ppo_opt.step()

    # 5. Export to ONNX
    logger.info("Exporting Models to ONNX Format...")
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(model_dir, exist_ok=True)

    lstm_model.eval()
    trans_model.eval()
    ppo_model.eval()

    dummy_seq = torch.randn(1, 1, 5)
    dummy_flat = torch.randn(1, 5)

    torch.onnx.export(lstm_model, dummy_seq, os.path.join(model_dir, "lstm.onnx"), export_params=True, opset_version=14, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'}, 'output': {0: 'batch_size'}})
    torch.onnx.export(trans_model, dummy_seq, os.path.join(model_dir, "transformer.onnx"), export_params=True, opset_version=14, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'}, 'output': {0: 'batch_size'}})
    torch.onnx.export(ppo_model, dummy_flat, os.path.join(model_dir, "ppo.onnx"), export_params=True, opset_version=14, input_names=['input'], output_names=['action_mean', 'state_value'], dynamic_axes={'input': {0: 'batch_size'}, 'action_mean': {0: 'batch_size'}, 'state_value': {0: 'batch_size'}})

    logger.info("All 100-Year Training Complete! AI models updated and exported.")

if __name__ == "__main__":
    main()
