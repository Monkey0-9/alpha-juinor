import pytest
import pandas as pd
import numpy as np
from alpha_families.agent_runner import run_agent
from strategies.institutional_strategy import InstitutionalStrategy

class NanAgent:
    def generate_signal(self, data, **kwargs):
        # Explicitly return NaN
        return {'signal': np.nan, 'confidence': np.nan}

def test_agent_runner_nan_safety():
    agent = NanAgent()
    data = pd.DataFrame({'Close': [100, 101, 102]})

    res = run_agent(agent, data)

    assert res['ok'] is True
    assert res['mu'] == 0.0
    assert res['confidence'] == 0.0

def test_strategy_nan_fill():
    # Setup market data
    tickers = ['AAPL', 'GOOGL']
    dates = pd.date_range('2023-01-01', periods=100)
    data = np.random.randn(100, 2) + 100
    market_data = pd.DataFrame(data, index=dates, columns=tickers)

    # Check if MultiIndex is needed
    # InstitutionalStrategy expects level 0 as symbol
    market_data.columns = pd.MultiIndex.from_product([tickers, ['Close']])

    strategy = InstitutionalStrategy()

    # Mock alpha_families to return NaN
    class MockAlpha:
        def generate_signal(self, data, **kwargs):
            return {'signal': np.nan, 'confidence': 0.5}

    strategy.alpha_families = [MockAlpha()]

    # Mock feature_store to return garbage or nothing so it uses raw data
    # Actually InstitutionalStrategy requires feature_store features
    strategy.feature_store.get_latest = lambda tickers: {t: {'features': {}, 'date': pd.Timestamp.utcnow().isoformat()} for t in tickers}

    signals = strategy.generate_signals(market_data)

    # Even if Alpha returns NaN, strategy should fill with 0.5 (Neutral)
    assert not signals.isna().any().any()
    assert (signals == 0.5).all().all()
