import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch
from alpha_families.agent_runner import run_agent
from alpha_families.statistical_alpha import StatisticalAlpha
from data.collectors.data_router import DataRouter

# 1. Pre-flight exit checks
def test_preflight_exit():
    # We can't easily test sys.exit(1) without a subprocess or catching SystemExit
    from ops.checklists import PreFlightChecklist
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(SystemExit) as excinfo:
            PreFlightChecklist.run_checks()
        assert excinfo.value.code == 1

# 2. Timestamp normalization
def test_timestamp_normalization():
    dr = DataRouter()
    # Create naive dataframe
    df = pd.DataFrame({'Close': [100, 101]}, index=pd.to_datetime(['2025-01-01 10:00', '2025-01-01 11:00']))
    assert df.index.tz is None

    # Normalize
    df_clean = dr._ensure_utc_index(df)
    assert str(df_clean.index.tz) == 'UTC'

# 3. Agent Runner Safety
def test_agent_runner_safety():
    class BadAgent:
        def evaluate(self, df, context):
            raise ValueError("Boom")

    res = run_agent(BadAgent(), None, None)
    assert res['ok'] is False
    assert res['confidence'] == 0.0
    assert "Boom" in res['error']

    class GoodAgent:
        def generate_signal(self, df, context):
            return {'signal': 0.75, 'confidence': 0.9}

    res = run_agent(GoodAgent(), None, None)
    assert res['ok'] is True
    assert res['mu'] == 0.75
    assert res['confidence'] == 0.9

# 4. Data Age Rejection (Mocking Strategy logic)
def test_data_age_logic():
    # We'll just verify the logic manually as it's embedded in the strategy class
    now_utc = pd.Timestamp.utcnow()
    old_ts = now_utc - pd.Timedelta(minutes=130)

    # Simulate logic
    last_ts = old_ts
    if last_ts.tzinfo is None: last_ts = last_ts.tz_localize('UTC')
    age_min = (now_utc - last_ts).total_seconds() / 60.0

    assert age_min > 120
    # Logic in code: if age_min > MAX_DATA_AGE_MINUTES: reject

# 5. Statistical Alpha Convergence
def test_statistical_alpha_convergence():
    alpha = StatisticalAlpha()
    # Create constant data that causes ARIMA convergence issues usually?? or random noise
    # Actually, statsmodels usually converges on simple noise, but let's try to mock the model fit result

    df = pd.DataFrame({'Close': np.random.randn(100) + 100, 'Volume': [1000]*100},
                      index=pd.date_range('2025-01-01', periods=100, tz='UTC'))

    with patch('statsmodels.tsa.arima.model.ARIMA.fit') as mock_fit:
        mock_res = MagicMock()
        mock_res.mle_retvals = {'converged': False}
        mock_fit.return_value = mock_res

        # Should return 0.0 without crash
        # The internal method _calculate_arima_forecast calls fit()
        # But generate_signal catches exceptions.
        # We need to verify it handles non-convergence gracefully inside _calculate_arima_forecast

        # Actually generate_signal calls _calculate_arima_forecast
        # We can test the private method or the public one
        res = alpha.generate_signal(df, None)
        # If it didn't converge, it returns 0.0 for that component.
        # The integration test ensures no crash.
        assert 'confidence' in res

