import pandas as pd
import numpy as np
import pytest
from hypothesis import given, strategies as st
from src.nexus.data.validator import DataValidator

@given(
    st.lists(
        st.tuples(
            st.floats(min_value=10, max_value=200), # Open
            st.floats(min_value=10, max_value=200), # High
            st.floats(min_value=10, max_value=200), # Low
            st.floats(min_value=10, max_value=200), # Close
        ),
        min_size=1,
        max_size=10
    )
)
def test_ohlc_invariants(bars):
    """
    Property-based test to ensure validator correctly handles randomized OHLC data.
    The validator MUST ensure High >= Low and all else within bounds.
    """
    df = pd.DataFrame(bars, columns=['Open', 'High', 'Low', 'Close'])
    validated_df = DataValidator.validate_ohlc(df)
    
    if not validated_df.empty:
        # Check invariants on every row
        assert (validated_df['High'] >= validated_df['Low']).all()
        assert (validated_df['Open'] >= validated_df['Low']).all()
        assert (validated_df['Open'] <= validated_df['High']).all()
        assert (validated_df['Close'] >= validated_df['Low']).all()
        assert (validated_df['Close'] <= validated_df['High']).all()

def test_spike_detection():
    """Verify that 50% single-bar spikes are correctly rejected."""
    data = {
        'Open': [100.0, 100.0],
        'High': [100.0, 160.0], # Spike 60%
        'Low': [100.0, 100.0],
        'Close': [100.0, 155.0] # 55% return
    }
    df = pd.DataFrame(data)
    validated_df = DataValidator.validate_ohlc(df, ticker="TEST_SPIKE")
    
    # Should drop the second row
    assert len(validated_df) == 1
    assert float(validated_df.iloc[0]['Close']) == 100.0

def test_numeric_integrity():
    """Verify that NaNs and Infs are purged."""
    data = {
        'Open': [100.0, 'bad_data', 100.0],
        'High': [105.0, 110.0, np.inf],
        'Low': [95.0, 90.0, 95.0],
        'Close': [102.0, 105.0, 102.0]
    }
    df = pd.DataFrame(data)
    validated_df = DataValidator.validate_ohlc(df)
    
    # Only the first row is truly clean
    assert len(validated_df) == 1
