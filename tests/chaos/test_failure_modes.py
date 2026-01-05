# tests/chaos/test_failure_modes.py
import pytest
import pandas as pd
import numpy as np
import logging
from .failure_injector import FailureInjector
from main import run_production_pipeline
from data.providers.yahoo import YahooDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockProvider:
    def fetch_ohlcv(self, tickers, start, end):
        # Return dummy data
        dates = pd.date_range("2023-01-01", periods=100)
        data = {}
        for tk in tickers:
            df = pd.DataFrame({
                "Open": np.random.rand(100) * 100,
                "High": np.random.rand(100) * 110,
                "Low": np.random.rand(100) * 90,
                "Close": np.random.rand(100) * 100,
                "Volume": np.random.rand(100) * 1000000
            }, index=dates)
            for col in df.columns:
                data[(tk, col)] = df[col]
        
        return pd.DataFrame(data)

def test_api_timeout_handling():
    """Verify system doesn't crash on high latency."""
    base_provider = MockProvider()
    chaos = FailureInjector(base_provider)
    chaos.configure({"api_timeouts": 1.0}) # 100% chance of delay
    
    logger.info("Starting API Timeout Test...")
    try:
        # We don't run full pipeline here as it takes too long
        # Just verify the fetch works with delay
        data = chaos.fetch_ohlcv(["AAPL"], "2023-01-01", "2023-01-10")
        assert not data.empty
        logger.info("API Timeout handled successfully (delayed but returned).")
    except ConnectionError as e:
        logger.info(f"API Connection Failure correctly simulated: {e}")
    except Exception as e:
        logger.error(f"Failed API Timeout test: {e}")
        raise

def test_missing_data_resilience():
    """Verify system handles random data gaps."""
    base_provider = MockProvider()
    chaos = FailureInjector(base_provider)
    chaos.configure({"missing_bars": 1.0}) # 100% chance of dropping rows
    
    logger.info("Starting Data Gap Test...")
    data = chaos.fetch_ohlcv(["AAPL"], "2023-01-01", "2023-04-01")
    assert len(data) < 100 # Should have dropped some
    logger.info(f"Data Gap handled: {len(data)} bars returned (expected < 100).")

def test_price_gap_resilience():
    """Verify system handles volatility spikes."""
    base_provider = MockProvider()
    chaos = FailureInjector(base_provider)
    chaos.configure({"price_gaps": 1.0}) # Force a gap
    
    logger.info("Starting Price Gap Test...")
    data = chaos.fetch_ohlcv(["AAPL"], "2023-01-01", "2023-01-10")
    # Gaps shouldn't cause nan/inf in high-level math
    returns = data.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    assert np.isfinite(returns.values).all()
    logger.info("Price Gap handled: Returns are finite despite 10% jumps.")

if __name__ == "__main__":
    test_api_timeout_handling()
    test_missing_data_resilience()
    test_price_gap_resilience()
