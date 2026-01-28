
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


    # Detailed matching logic with time-based reconciliation
    reconciliation_results = []

    if not model_trades.empty and not broker_fills.empty:
        # Parse timestamps
        model_trades['timestamp'] = pd.to_datetime(model_trades.get('timestamp', model_trades.get('created_at', datetime.now())))
        broker_fills['timestamp'] = pd.to_datetime(broker_fills.get('transaction_time', broker_fills.get('created_at', datetime.now())))

        # Match trades by symbol and approximate timing (within 5 minutes)
        match_window = timedelta(minutes=5)

        for _, model_trade in model_trades.iterrows():
            symbol = model_trade.get('ticker', model_trade.get('symbol'))
            model_qty = model_trade['quantity']
            model_time = model_trade['timestamp']
            model_price = model_trade.get('price', 0.0)

            # Find matching broker fills
            broker_matches = broker_fills[
                (broker_fills['symbol'] == symbol) &
                (abs(broker_fills['timestamp'] - model_time) <= match_window)
            ]

            if not broker_matches.empty:
                # Take first match (could be improved with better matching logic)
                broker_fill = broker_matches.iloc[0]
                broker_qty = broker_fill['net_qty']
                broker_price = float(broker_fill.get('price', 0.0))

                # Calculate differences
                qty_diff = model_qty - broker_qty
                price_diff = broker_price - model_price if model_price > 0 else 0.0
                slippage_bps = (price_diff / model_price * 10000) if model_price > 0 else 0.0

                reconciliation_results.append({
                    'symbol': symbol,
                    'model_qty': model_qty,
                    'broker_qty': broker_qty,
                    'qty_diff': qty_diff,
                    'model_price': model_price,
                    'broker_price': broker_price,
                    'price_diff': price_diff,
                    'slippage_bps': slippage_bps,
                    'model_time': model_time,
                    'broker_time': broker_fill['timestamp'],
                    'time_diff_seconds': (broker_fill['timestamp'] - model_time).total_seconds(),
                    'status': 'MATCHED' if abs(qty_diff) < 0.01 else 'QUANTITY_MISMATCH'
                })
            else:
                # No broker match found
                reconciliation_results.append({
                    'symbol': symbol,
                    'model_qty': model_qty,
                    'broker_qty': 0.0,
                    'qty_diff': model_qty,
                    'model_price': model_price,
                    'broker_price': 0.0,
                    'price_diff': 0.0,
                    'slippage_bps': 0.0,
                    'model_time': model_time,
                    'broker_time': None,
                    'time_diff_seconds': None,
                    'status': 'NO_BROKER_FILL'
                })

        # Check for broker fills without model trades
        model_symbols_times = set(
            (row.get('ticker', row.get('symbol')), row['timestamp'])
            for _, row in model_trades.iterrows()
        )

        for _, broker_fill in broker_fills.iterrows():
            broker_symbol = broker_fill['symbol']
            broker_time = broker_fill['timestamp']

            # Check if this broker fill was matched
            has_match = any(
                symbol == broker_symbol and abs((timestamp - broker_time).total_seconds()) <= match_window.total_seconds()
                for symbol, timestamp in model_symbols_times
            )

            if not has_match:
                reconciliation_results.append({
                    'symbol': broker_symbol,
                    'model_qty': 0.0,
                    'broker_qty': broker_fill['net_qty'],
                    'qty_diff': -broker_fill['net_qty'],
                    'model_price': 0.0,
                    'broker_price': float(broker_fill.get('price', 0.0)),
                    'price_diff': 0.0,
                    'slippage_bps': 0.0,
                    'model_time': None,
                    'broker_time': broker_time,
                    'time_diff_seconds': None,
                    'status': 'NO_MODEL_TRADE'
                })

    # Save detailed reconciliation report
    if reconciliation_results:
        rec_df = pd.DataFrame(reconciliation_results)
        rec_df.to_csv(report_path, index=False)
        logger.info(f"Detailed reconciliation saved to {report_path}")

        # Summary statistics
        print("\n" + "="*60)
        print("RECONCILIATION SUMMARY".center(60))
        print("="*60)

        matched = rec_df[rec_df['status'] == 'MATCHED']
        qty_mismatch = rec_df[rec_df['status'] == 'QUANTITY_MISMATCH']
        no_broker = rec_df[rec_df['status'] == 'NO_BROKER_FILL']
        no_model = rec_df[rec_df['status'] == 'NO_MODEL_TRADE']

        print(f"\nMatched trades: {len(matched)}")
        print(f"Quantity mismatches: {len(qty_mismatch)}")
        print(f"Model trades without broker fills: {len(no_broker)}")
        print(f"Broker fills without model trades: {len(no_model)}")

        if len(matched) > 0:
            avg_slippage = matched['slippage_bps'].mean()
            max_slippage = matched['slippage_bps'].abs().max()
            print(f"\nAverage slippage: {avg_slippage:.2f} bps")
            print(f"Max slippage: {max_slippage:.2f} bps")

        if len(qty_mismatch) > 0:
            print("\nQUANTITY MISMATCHES:")
            for _, row in qty_mismatch.iterrows():
                print(f"  {row['symbol']}: Model={row['model_qty']:.2f}, Broker={row['broker_qty']:.2f}, Diff={row['qty_diff']:.2f}")

        print("\n" + "="*60)
    else:
        logger.warning("No reconciliation results to save")

if __name__ == "__main__":
    reconcile_latest_run()
