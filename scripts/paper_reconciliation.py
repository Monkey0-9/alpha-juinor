
import sys
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add project root
sys.path.append(os.getcwd())

try:
    from brokers.alpaca_broker import AlpacaExecutionHandler
    from backtest.registry import BacktestRegistry
except ImportError:
    print("Error importing project modules. Run from root.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reconciliation")

def reconcile_latest_run():
    load_dotenv()
    
    # 1. Get Model Trades (Simulation)
    registry = BacktestRegistry()
    runs = registry.list_runs()
    if not runs:
        logger.error("No model runs found in registry.")
        return
        
    latest_run = runs[0]
    run_id = latest_run['run_id']
    run_dir = Path(latest_run['path'])
    trades_file = run_dir / "trades.csv"
    
    if not trades_file.exists():
        logger.warning(f"No trades.csv in latest run {run_id}")
        model_trades = pd.DataFrame()
    else:
        model_trades = pd.read_csv(trades_file)
        
    logger.info(f"Loaded {len(model_trades)} model trades from {run_id}")

    # 2. Get Broker Fills (Alpaca)
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    if not api_key:
        logger.error("Alpaca credentials missing.")
        return

    handler = AlpacaExecutionHandler(api_key, secret_key, base_url)
    
    try:
        broker_fills_data = handler.get_activities("FILL")
        broker_fills = pd.DataFrame(broker_fills_data)
        if not broker_fills.empty:
            broker_fills['qty'] = broker_fills['qty'].astype(float)
            # side: buy/sell
            broker_fills['net_qty'] = broker_fills.apply(lambda x: x['qty'] if x['side'] == 'buy' else -x['qty'], axis=1)
    except Exception as e:
        logger.error(f"Failed to fetch Alpaca fills: {e}")
        broker_fills = pd.DataFrame()

    logger.info(f"Loaded {len(broker_fills)} broker fills from Alpaca")

    # 3. Reconciliation Logic
    print("\n" + "="*40)
    print("RECONCILIATION REPORT".center(40))
    print("="*40)
    
    if not model_trades.empty:
        model_net = model_trades.groupby('ticker')['quantity'].sum()
        print("\n[MODEL] Net Quantities:")
        print(model_net)
    
    if not broker_fills.empty:
        broker_net = broker_fills.groupby('symbol')['net_qty'].sum()
        print("\n[BROKER] Net Quantities:")
        print(broker_net)
        
        # Intersection comparison
        if not model_trades.empty:
            common = set(model_net.index).intersection(set(broker_net.index))
            print("\n[MATCH] Model vs Broker:")
            for sym in common:
                diff = model_net[sym] - broker_net[sym]
                status = "OK" if abs(diff) < 0.01 else f"MISMATCH ({diff:+.2f})"
                print(f" {sym:5}: Model={model_net[sym]:>8.2f} | Broker={broker_net[sym]:>8.2f} | {status}")
    else:
        print("\nNo broker fills found to compare.")
    
    print("\n" + "="*40)
    
    # Save report
    report_path = Path("output/analysis/reconciliation_latest.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # TODO: Detailed matching logic
    # For now, just dumping both side by side or counting mismatch
    
if __name__ == "__main__":
    reconcile_latest_run()
