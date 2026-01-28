
import pytest
import pandas as pd
import numpy as np
from utils.timeutils import ensure_business_days, safe_infer_freq

class TestTimeUtils:

    def test_ensure_business_days_fixes_missing_freq(self):
        """Test that frequency is assigned to a daily series"""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({"Close": np.random.randn(10)}, index=dates)

        # Manually clear freq
        df.index.freq = None

        cleaned = ensure_business_days(df)

        assert cleaned.index.freq == 'B'
        assert cleaned.index.tz is not None
        assert str(cleaned.index.tz) == "UTC"

    def test_fills_gaps(self):
        """Test that missing business days are filled"""
        # Create data with a gap (skip Jan 3)
        dates = [
            pd.Timestamp("2023-01-02", tz="UTC"), # Mon
            # Jan 3 Tue missing
            pd.Timestamp("2023-01-04", tz="UTC")  # Wed
        ]
        df = pd.DataFrame({"Close": [100.0, 102.0]}, index=dates)

        cleaned = ensure_business_days(df)

        assert len(cleaned) == 3 # Mon, Tue, Wed
        assert cleaned.index.freq == 'B'
        # Check forward fill
        assert cleaned.loc[pd.Timestamp("2023-01-03", tz="UTC")]["Close"] == 100.0

    def test_timezone_naive_to_utc(self):
        """Test naive timestamps are localized to UTC"""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="B") # Naive
        df = pd.DataFrame({"Close": [1]*5}, index=dates)

        cleaned = ensure_business_days(df)
        assert str(cleaned.index.tz) == "UTC"

    def test_statsmodels_integration(self):
        """Verify no warning with statsmodels (simulated)"""
        # Statsmodels checks 'freq' attribute.
        df = pd.DataFrame({"Close": np.random.randn(20)}, index=pd.date_range("2023-01-01", periods=20, freq="B"))
        cleaned = ensure_business_days(df)
        assert cleaned.index.freqstr == 'B' or cleaned.index.freq == 'B'
