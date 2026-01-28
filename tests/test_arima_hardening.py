"""
tests/test_arima_hardening.py
P0-4: Test ARIMA hardening and fallback mechanisms
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from alpha_families.statistical_alpha import StatisticalAlpha, ModelNotFitError, MIN_ARIMA_SAMPLE


def create_price_data(n_rows=100):
    """Create synthetic price data for testing."""
    dates = pd.date_range(end='2026-01-01', periods=n_rows)
    prices = 100 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.01))
    return pd.DataFrame({'Close': prices}, index=dates)


def test_arima_minimum_sample_check():
    """Test that ARIMA requires minimum sample size."""
    alpha = StatisticalAlpha()

    # Create insufficient data
    data = create_price_data(n_rows=20)  # Less than MIN_ARIMA_SAMPLE (30)

    with pytest.raises(ModelNotFitError, match="Insufficient data"):
        alpha.arima_safe_predict(data, symbol="TEST")


def test_arima_ewma_fallback():
    """Test that ARIMA falls back to EWMA on failure."""
    alpha = StatisticalAlpha()

    # Create data but mock ARIMA to fail
    data = create_price_data(n_rows=100)

    with patch('alpha_families.statistical_alpha.ARIMA') as mock_arima:
        mock_arima.side_effect = Exception("ARIMA fit failed")

        result = alpha.arima_safe_predict(data, symbol="TEST")

        assert result["method"] == "EWMA"
        assert "signal" in result
        assert alpha._symbol_failures["TEST"] == 1


def test_symbol_degradation_after_three_failures():
    """Test that symbol is degraded after 3 consecutive failures."""
    alpha = StatisticalAlpha()
    data = create_price_data(n_rows=100)

    # Force 3 failures
    with patch('alpha_families.statistical_alpha.ARIMA') as mock_arima:
        mock_arima.side_effect = Exception("ARIMA fail")

        for i in range(3):
            result = alpha.arima_safe_predict(data, symbol="TEST")
            assert result["method"] == "EWMA"

        # Check degradation
        assert "TEST" in alpha._degraded_symbols
        assert alpha._symbol_failures["TEST"] >= 3


def test_degraded_symbol_uses_ewma_only():
    """Test that degraded symbol only uses EWMA, no ARIMA attempts."""
    alpha = StatisticalAlpha()
    data = create_price_data(n_rows=100)

    # Manually degrade symbol
    alpha._degraded_symbols.add("TEST")
    alpha._symbol_failures["TEST"] = 5

    with patch('alpha_families.statistical_alpha.ARIMA') as mock_arima:
        mock_arima.side_effect = Exception("Should not be called")

        result = alpha.arima_safe_predict(data, symbol="TEST")

        # Should use EWMA without calling ARIMA
        assert result["method"] == "EWMA"
        assert result["reason"] == "DEGRADED_ARIMA"
        # ARIMA should never have been called
        assert not mock_arima.called


def test_failure_counter_resets_on_success():
    """Test that failure counter resets after successful ARIMA fit."""
    alpha = StatisticalAlpha()
    data = create_price_data(n_rows=100)

    # Record 2 failures
    alpha._symbol_failures["TEST"] = 2

    # Mock successful ARIMA fit
    mock_model_fit = Mock()
    mock_model_fit.mle_retvals = {'converged': True}
    mock_model_fit.forecast.return_value = pd.Series([0.01])

    with patch('alpha_families.statistical_alpha.ARIMA') as mock_arima:
        mock_arima.return_value.fit.return_value = mock_model_fit

        result = alpha.arima_safe_predict(data, symbol="TEST")

        assert result["method"] == "ARIMA"
        # Failure counter should be reset
        assert alpha._symbol_failures["TEST"] == 0


def test_arima_fallback_increments_metrics():
    """Test that ARIMA fallbacks increment global metrics."""
    alpha = StatisticalAlpha()
    data = create_price_data(n_rows=100)

    initial_fallbacks = alpha.arima_fallbacks

    with patch('alpha_families.statistical_alpha.ARIMA') as mock_arima:
        mock_arima.side_effect = Exception("ARIMA fail")

        result = alpha.arima_safe_predict(data, symbol="TEST")

        assert alpha.arima_fallbacks == initial_fallbacks + 1


def test_generate_signal_confidence_differs():
    """Test that ARIMA and EWMA have different confidence levels."""
    alpha = StatisticalAlpha()
    data = create_price_data(n_rows=100)

    # Mock successful ARIMA
    mock_model_fit = Mock()
    mock_model_fit.mle_retvals = {'converged': True}
    mock_model_fit.forecast.return_value = pd.Series([0.01])

    with patch('alpha_families.statistical_alpha.ARIMA') as mock_arima:
        mock_arima.return_value.fit.return_value = mock_model_fit

        result = alpha.generate_signal(data, symbol="TEST")
        arima_confidence = result["confidence"]

    # Mock ARIMA failure → EWMA
    with patch('alpha_families.statistical_alpha.ARIMA') as mock_arima:
        mock_arima.side_effect = Exception("fail")

        result = alpha.generate_signal(data, symbol="TEST2")
        ewma_confidence = result["confidence"]

    # ARIMA should have higher confidence than EWMA
    assert arima_confidence > ewma_confidence
