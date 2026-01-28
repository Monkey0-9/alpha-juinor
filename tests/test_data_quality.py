"""
Unit tests for data quality module.
"""

import pytest
import pandas as pd
import numpy as np
from data.quality import compute_data_quality, validate_data_for_trading, validate_data_for_ml


class TestDataQuality:
    """Test data quality scoring functions."""

    @pytest.fixture
    def good_data(self):
        """Create high-quality sample data."""
        dates = pd.date_range('2020-01-01', periods=1500, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 1500)))
        volumes = np.random.lognormal(10, 0.5, 1500)

        return pd.DataFrame({
            'Close': prices,
            'Volume': volumes
        }, index=dates)

    @pytest.fixture
    def poor_data(self):
        """Create low-quality sample data with issues."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.random.uniform(50, 150, 100)
        prices[10:15] = np.nan  # Missing values
        prices[20] = -5  # Negative price
        prices[30] = 0  # Zero price

        return pd.DataFrame({
            'Close': prices
        }, index=dates)

    def test_compute_quality_good_data(self, good_data):
        """Test quality scoring on good data."""
        score, reasons = compute_data_quality(good_data)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score >= 0.8  # Should be high quality
        assert "OK" in reasons or len(reasons) == 0 or all('EXTREME' not in r for r in reasons)

    def test_compute_quality_poor_data(self, poor_data):
        """Test quality scoring on poor data."""
        score, reasons = compute_data_quality(poor_data)

        assert isinstance(score, float)
        assert score < 0.8  # Should be lower quality
        assert len(reasons) > 0
        assert any('MISSING' in r or 'ZERO' in r or 'NEGATIVE' in r for r in reasons)

    def test_compute_quality_empty_data(self):
        """Test quality scoring on empty data."""
        df = pd.DataFrame()
        score, reasons = compute_data_quality(df)

        assert score == 0.0
        assert "NO_DATA" in reasons

    def test_validate_for_trading_sufficient(self, good_data):
        """Test trading validation with sufficient data."""
        is_valid, reason = validate_data_for_trading(good_data, min_rows=1260, min_quality=0.6)

        assert is_valid is True
        assert reason == "OK"

    def test_validate_for_trading_insufficient_rows(self, good_data):
        """Test trading validation with insufficient rows."""
        short_data = good_data.head(100)
        is_valid, reason = validate_data_for_trading(short_data, min_rows=1260, min_quality=0.6)

        assert is_valid is False
        assert "INSUFFICIENT_HISTORY" in reason

    def test_validate_for_trading_low_quality(self, poor_data):
        """Test trading validation with low quality data."""
        is_valid, reason = validate_data_for_trading(poor_data, min_rows=50, min_quality=0.6)

        assert is_valid is False
        assert "LOW_QUALITY" in reason

    def test_validate_for_ml_strict(self, good_data):
        """Test ML validation with stricter requirements."""
        is_ready, reasons = validate_data_for_ml(good_data, min_rows=1260, min_quality=0.7)

        assert isinstance(is_ready, bool)
        assert isinstance(reasons, list)

        if is_ready:
            assert "ML_READY" in reasons

    def test_validate_for_ml_insufficient(self, poor_data):
        """Test ML validation with insufficient data."""
        is_ready, reasons = validate_data_for_ml(poor_data, min_rows=1260, min_quality=0.7)

        assert is_ready is False
        assert len(reasons) > 0
        assert any('INSUFFICIENT' in r or 'QUALITY' in r for r in reasons)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
