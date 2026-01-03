
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
    
    # Fetch trades from Alpaca (Limit 50 or by date)
    # Using requests directly if handler doesn't expose get_activities
    # The handler I wrote earlier doesn't have get_trades/activities.
    # I'll add a helper here or extend handler. 
    # Let's use requests directly here for reporting.
    
    headers = handler.headers
    # Get activities (FILL)
    try:
        r = requests.get(f"{base_url}/v2/account/activities/FILL", headers=headers)
        if r.status_code == 200:
            fills_data = r.json()
            broker_fills = pd.DataFrame(fills_data)
        else:
            logger.error(f"Alpaca API error: {r.text}")
            broker_fills = pd.DataFrame()
    except Exception as e:
        # Fallback if requests not imported (handler imports it but we need it here)
        import requests
        r = requests.get(f"{base_url}/v2/account/activities/FILL", headers=headers)
        broker_fills = pd.DataFrame(r.json() if r.status_code == 200 else [])

    logger.info(f"Loaded {len(broker_fills)} broker fills from Alpaca")

    # 3. Logic to Compare (e.g., match by Symbol and Date)
    # This is a simplified reconciliation
    
    print("\n--- RECONCILIATION REPORT ---")
    print(f"Model Trades: {len(model_trades)}")
    print(f"Broker Fills: {len(broker_fills)}")
    
    # Save report
    report_path = Path("output/analysis/reconciliation_latest.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # TODO: Detailed matching logic
    # For now, just dumping both side by side or counting mismatch
    
if __name__ == "__main__":
    reconcile_latest_run()
