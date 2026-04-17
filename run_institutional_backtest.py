import asyncio
import os
from datetime import datetime, timedelta, timezone
from src.nexus.core.context import engine_context
from src.nexus.data.engine import DataEngine
from src.nexus.data.providers.yahoo import YahooDataProvider
from src.nexus.research.momentum import MomentumAlpha
from src.nexus.backtest.engine import BacktestEngine

async def main():
    # 1. Initialize Global Engine Core
    engine_context.initialize(config_path="config/development.yaml")
    logger = engine_context.get_logger("system")
    logger.info("Starting institutional end-to-end backtest verification...")

    # 2. Setup Data Layer
    # Use a local directory for Parquet caching to ensure reproducibility
    data_engine = DataEngine(cache_dir="data/parquet")
    data_engine.add_provider(YahooDataProvider())

    # 3. Setup Strategy
    # Using a 252-day (1 year) momentum lookback
    strategy = MomentumAlpha(name="momentum_1y", lookback=252)

    # 4. Setup Simulation Engine
    backtester = BacktestEngine(data_engine, initial_cash=100000.0)

    # 5. Execute Backtest
    # Simulation range: Past 2 years of SPY data
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=730)
    
    symbols = ["SPY"]
    
    try:
        results = await backtester.run(
            symbols=symbols,
            strategy=strategy,
            start=start_date,
            end=end_date,
            interval="1d"
        )

        # 6. Report Institutional Metrics
        metrics = results["metrics"]
        logger.info("Backtest Execution Complete.")
        
        print("\n" + "="*50)
        print(" INSTITUTIONAL PERFORMANCE SUMMARY")
        print("="*50)
        print(f" Strategy:           {results['strategy']}")
        print(f" Total Return:      {metrics['total_return']*100:.2f}%")
        print(f" Annualized Return: {metrics['annualized_return']*100:.2f}%")
        print(f" Annualized Vol:    {metrics['annualized_vol']*100:.2f}%")
        print(f" Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f" Max Drawdown:      {metrics['max_drawdown']*100:.2f}%")
        print(f" Total Trades:      {len(results['trades'])}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
