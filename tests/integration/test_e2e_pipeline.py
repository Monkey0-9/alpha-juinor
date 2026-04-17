import pytest
from datetime import datetime, timedelta, timezone
from src.nexus.core.context import engine_context
from src.nexus.data.engine import DataEngine
from src.nexus.research.momentum import MomentumAlpha
from src.nexus.backtest.engine import BacktestEngine
from src.nexus.data.providers.base import DataProvider
from src.nexus.models.market import MarketBar

class MockProvider(DataProvider):
    def get_name(self) -> str:
        return "mock"
        
    async def get_historical_data(self, symbol: str, start: datetime, end: datetime, interval: str = "1d"):
        bars = []
        # Generate 300 days of mock data trending upwards
        base_price = 100.0
        for i in range(300):
            ts = start + timedelta(days=i)
            # Add small random noise but strong uptrend
            open_p = base_price + (i * 0.5)
            bars.append(MarketBar(
                symbol=symbol,
                timestamp=ts,
                open=open_p,
                high=open_p + 2.0,
                low=open_p - 1.0,
                close=open_p + 1.0,
                volume=1000000
            ))
        return bars

@pytest.mark.asyncio
async def test_full_backtest_pipeline():
    engine_context.initialize(config_path="")
    
    data_engine = DataEngine(cache_dir="data/test_parquet")
    data_engine.add_provider(MockProvider())
    
    strategy = MomentumAlpha(name="momentum_test", lookback=50)
    backtester = BacktestEngine(data_engine, initial_cash=100000.0)
    
    start_date = datetime.now(timezone.utc) - timedelta(days=300)
    end_date = datetime.now(timezone.utc)
    
    results = await backtester.run(
        symbols=["TEST"],
        strategy=strategy,
        start=start_date,
        end=end_date,
        interval="1d"
    )
    
    assert results["strategy"] == "momentum_test"
    assert "metrics" in results
    assert len(results["trades"]) > 0
    # Because our mock data strictly trends up, momentum should catch it and profit
    assert results["metrics"]["total_return"] > 0
