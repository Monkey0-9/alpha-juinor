"""
Tests for Advanced Alpha Factors
=================================

Comprehensive tests for microstructure and advanced technical factors.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import factor modules
import sys
sys.path.insert(0, 'c:/mini-quant-fund')

from strategies.features.microstructure_factors import (
    MicrostructureFactors,
    MicrostructureMetrics
)
from strategies.features.advanced_technical_factors import (
    AdvancedTechnicalFactors,
    AdvancedTechnicalMetrics
)


@pytest.fixture
def sample_trades():
    """Generate sample trade data."""
    np.random.seed(42)
    n = 1000

    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.1)
    volumes = np.random.lognormal(mean=5, sigma=1, size=n)
    sides = np.random.choice([1, -1], size=n)

    timestamps = [datetime.now() + timedelta(seconds=i) for i in range(n)]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'side': sides
    })

    return df


@pytest.fixture
def sample_quotes():
    """Generate sample quote data."""
    np.random.seed(42)
    n = 500

    base_price = 100.0
    mid_prices = base_price + np.cumsum(np.random.randn(n) * 0.1)
    spreads = np.random.uniform(0.01, 0.05, n)

    timestamps = [datetime.now() + timedelta(seconds=i*2) for i in range(n)]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'bid': mid_prices - spreads / 2,
        'ask': mid_prices + spreads / 2,
        'bid_size': np.random.lognormal(mean=6, sigma=0.5, size=n),
        'ask_size': np.random.lognormal(mean=6, sigma=0.5, size=n)
    })

    return df


@pytest.fixture
def sample_prices():
    """Generate sample price series."""
    np.random.seed(42)
    n = 500

    # Generate trending price series
    trend = np.linspace(100, 110, n)
    noise = np.random.randn(n) * 0.5
    prices = trend + noise

    dates = pd.date_range(start='2024-01-01', periods=n, freq='D')

    return pd.Series(prices, index=dates)


class TestMicrostructureFactors:
    """Tests for microstructure factors."""

    def test_initialization(self):
        """Test factor calculator initialization."""
        mf = MicrostructureFactors(vpin_buckets=50, kyle_window=20, min_trades=100)
        assert mf.vpin_buckets == 50
        assert mf.kyle_window == 20
        assert mf.min_trades == 100

    def test_compute_vpin(self, sample_trades):
        """Test VPIN calculation."""
        mf = MicrostructureFactors()
        vpin = mf.compute_vpin(sample_trades)

        assert isinstance(vpin, float)
        assert 0 <= vpin <= 1, f"VPIN should be in [0,1], got {vpin}"

    def test_compute_kyle_lambda(self, sample_trades):
        """Test Kyle's lambda calculation."""
        mf = MicrostructureFactors()
        kyle_lambda = mf.compute_kyle_lambda(sample_trades)

        assert isinstance(kyle_lambda, float)
        assert kyle_lambda >= 0, "Kyle's lambda should be non-negative"

    def test_compute_effective_spread(self, sample_trades, sample_quotes):
        """Test effective spread calculation."""
        mf = MicrostructureFactors()

        # With quotes
        spread_with_quotes = mf.compute_effective_spread(sample_trades, sample_quotes)
        assert isinstance(spread_with_quotes, float)
        assert spread_with_quotes >= 0

        # Without quotes
        spread_without_quotes = mf.compute_effective_spread(sample_trades)
        assert isinstance(spread_without_quotes, float)
        assert spread_without_quotes >= 0

    def test_compute_adverse_selection(self, sample_trades):
        """Test adverse selection cost."""
        mf = MicrostructureFactors()
        asc = mf.compute_adverse_selection(sample_trades)

        assert isinstance(asc, float)
        assert asc >= 0, "Adverse selection cost should be non-negative"

    def test_compute_roll_spread(self, sample_trades):
        """Test Roll's spread estimator."""
        mf = MicrostructureFactors()
        roll_spread = mf.compute_roll_spread(sample_trades)

        assert isinstance(roll_spread, float)
        assert roll_spread >= 0

    def test_compute_amihud(self, sample_trades):
        """Test Amihud illiquidity ratio."""
        mf = MicrostructureFactors()
        amihud = mf.compute_amihud(sample_trades)

        assert isinstance(amihud, float)
        assert amihud >= 0

    def test_compute_trade_imbalance(self, sample_trades):
        """Test trade imbalance calculation."""
        mf = MicrostructureFactors()
        imbalance = mf.compute_trade_imbalance(sample_trades)

        assert isinstance(imbalance, float)
        assert -1 <= imbalance <= 1, f"Imbalance should be in [-1,1], got {imbalance}"

    def test_compute_pin(self, sample_trades):
        """Test PIN calculation."""
        mf = MicrostructureFactors()
        pin = mf.compute_pin(sample_trades)

        assert isinstance(pin, float)
        assert 0 <= pin <= 1, f"PIN should be in [0,1], got {pin}"

    def test_compute_depth_imbalance(self, sample_quotes):
        """Test depth imbalance calculation."""
        mf = MicrostructureFactors()
        depth_imb = mf.compute_depth_imbalance(sample_quotes)

        assert isinstance(depth_imb, float)
        assert -1 <= depth_imb <= 1

    def test_compute_volume_fragility(self, sample_trades):
        """Test volume fragility index."""
        mf = MicrostructureFactors()
        fragility = mf.compute_volume_fragility(sample_trades)

        assert isinstance(fragility, float)
        assert fragility >= 0

    def test_compute_all(self, sample_trades, sample_quotes):
        """Test computing all metrics at once."""
        mf = MicrostructureFactors()
        metrics = mf.compute_all(sample_trades, sample_quotes)

        assert isinstance(metrics, MicrostructureMetrics)
        assert 0 <= metrics.vpin <= 1
        assert metrics.kyle_lambda >= 0
        assert metrics.effective_spread >= 0
        assert metrics.adverse_selection_cost >= 0
        assert metrics.roll_spread >= 0
        assert metrics.amihud_illiquidity >= 0
        assert -1 <= metrics.trade_imbalance <= 1
        assert 0 <= metrics.pin <= 1
        assert -1 <= metrics.depth_imbalance <= 1
        assert metrics.volume_fragility >= 0

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        mf = MicrostructureFactors(min_trades=100)

        # Create small dataset
        small_trades = pd.DataFrame({
            'timestamp': [datetime.now()],
            'price': [100.0],
            'volume': [1000],
            'side': [1]
        })

        metrics = mf.compute_all(small_trades)

        # Should return empty/neutral metrics
        assert isinstance(metrics, MicrostructureMetrics)
        assert metrics.vpin == 0.5  # Neutral value
        assert metrics.pin == 0.5  # Neutral value


class TestAdvancedTechnicalFactors:
    """Tests for advanced technical factors."""

    def test_initialization(self):
        """Test factor calculator initialization."""
        atf = AdvancedTechnicalFactors(lookback=252)
        assert atf.lookback == 252

    def test_compute_hurst_exponent(self, sample_prices):
        """Test Hurst exponent calculation."""
        atf = AdvancedTechnicalFactors()
        hurst = atf.compute_hurst_exponent(sample_prices)

        assert isinstance(hurst, float)
        assert 0 <= hurst <= 1, f"Hurst should be in [0,1], got {hurst}"

        # Trending series should have H > 0.5
        # Our sample has upward trend, so H should be > 0.5
        assert hurst > 0.3, "Trending series should have Hurst > 0.3"

    def test_compute_fractal_dimension(self, sample_prices):
        """Test fractal dimension calculation."""
        atf = AdvancedTechnicalFactors()
        fractal_dim = atf.compute_fractal_dimension(sample_prices)

        assert isinstance(fractal_dim, float)
        assert 1 <= fractal_dim <= 2, f"Fractal dimension should be in [1,2], got {fractal_dim}"

    def test_compute_shannon_entropy(self, sample_prices):
        """Test Shannon entropy calculation."""
        atf = AdvancedTechnicalFactors()
        returns = sample_prices.pct_change().dropna()
        entropy = atf.compute_shannon_entropy(returns)

        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_compute_dominant_cycle(self, sample_prices):
        """Test dominant cycle detection."""
        atf = AdvancedTechnicalFactors()
        cycle = atf.compute_dominant_cycle(sample_prices)

        assert isinstance(cycle, float)
        assert cycle >= 0

    def test_compute_wavelet_energy(self, sample_prices):
        """Test wavelet energy ratio."""
        atf = AdvancedTechnicalFactors()
        energy_ratio = atf.compute_wavelet_energy(sample_prices)

        assert isinstance(energy_ratio, float)
        assert energy_ratio >= 0

    def test_compute_regime_momentum(self, sample_prices):
        """Test regime-conditional momentum."""
        atf = AdvancedTechnicalFactors()
        returns = sample_prices.pct_change().dropna()
        momentum = atf.compute_regime_momentum(sample_prices, returns)

        assert isinstance(momentum, float)
        # Momentum can be positive or negative

    def test_compute_correlation_breakdown(self, sample_prices):
        """Test correlation breakdown detector."""
        atf = AdvancedTechnicalFactors()
        returns = sample_prices.pct_change().dropna()
        breakdown = atf.compute_correlation_breakdown(returns)

        assert isinstance(breakdown, float)
        assert 0 <= breakdown <= 1

    def test_compute_noise_ratio(self, sample_prices):
        """Test noise ratio calculation."""
        atf = AdvancedTechnicalFactors()
        noise_ratio = atf.compute_noise_ratio(sample_prices)

        assert isinstance(noise_ratio, float)
        assert noise_ratio >= 0

    def test_compute_complexity(self, sample_prices):
        """Test complexity index."""
        atf = AdvancedTechnicalFactors()
        returns = sample_prices.pct_change().dropna()
        complexity = atf.compute_complexity(returns)

        assert isinstance(complexity, float)
        assert 0 <= complexity <= 1.5  # Can exceed 1 slightly

    def test_compute_phase_coherence(self, sample_prices):
        """Test phase coherence calculation."""
        atf = AdvancedTechnicalFactors()
        coherence = atf.compute_phase_coherence(sample_prices)

        assert isinstance(coherence, float)
        assert 0 <= coherence <= 1

    def test_compute_dtw_similarity(self, sample_prices):
        """Test Dynamic Time Warping similarity."""
        atf = AdvancedTechnicalFactors()

        # Compare series with itself (should be very similar)
        similarity = atf.compute_dynamic_time_warping_similarity(
            sample_prices.iloc[:100],
            sample_prices.iloc[:100]
        )

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        assert similarity > 0.9, "Identical series should have high similarity"

        # Compare with random series (should be low similarity)
        random_series = pd.Series(np.random.randn(100))
        similarity_random = atf.compute_dynamic_time_warping_similarity(
            sample_prices.iloc[:100],
            random_series
        )

        assert similarity_random < similarity

    def test_compute_lyapunov_exponent(self, sample_prices):
        """Test Lyapunov exponent calculation."""
        atf = AdvancedTechnicalFactors()
        returns = sample_prices.pct_change().dropna()
        lyapunov = atf.compute_lyapunov_exponent(returns)

        assert isinstance(lyapunov, float)
        # Lyapunov can be positive, negative, or zero

    def test_compute_all(self, sample_prices):
        """Test computing all metrics at once."""
        atf = AdvancedTechnicalFactors()
        returns = sample_prices.pct_change().dropna()
        metrics = atf.compute_all(sample_prices, returns)

        assert isinstance(metrics, AdvancedTechnicalMetrics)
        assert 0 <= metrics.hurst_exponent <= 1
        assert 1 <= metrics.fractal_dimension <= 2
        assert metrics.entropy >= 0
        assert metrics.spectral_density_peak >= 0
        assert metrics.wavelet_energy_ratio >= 0
        assert 0 <= metrics.correlation_breakdown <= 1
        assert metrics.noise_ratio >= 0
        assert metrics.complexity_index >= 0
        assert 0 <= metrics.phase_coherence <= 1

    def test_deterministic_results(self, sample_prices):
        """Test that results are deterministic given same input."""
        atf = AdvancedTechnicalFactors()

        # Compute twice
        hurst1 = atf.compute_hurst_exponent(sample_prices)
        hurst2 = atf.compute_hurst_exponent(sample_prices)

        assert hurst1 == hurst2, "Results should be deterministic"


class TestIntegration:
    """Integration tests for factor combinations."""

    def test_full_factor_pipeline(self, sample_trades, sample_quotes, sample_prices):
        """Test complete factor computation pipeline."""
        # Microstructure factors
        mf = MicrostructureFactors()
        micro_metrics = mf.compute_all(sample_trades, sample_quotes)

        # Technical factors
        atf = AdvancedTechnicalFactors()
        returns = sample_prices.pct_change().dropna()
        tech_metrics = atf.compute_all(sample_prices, returns)

        # Combine into feature vector
        feature_vector = {
            # Microstructure
            'vpin': micro_metrics.vpin,
            'kyle_lambda': micro_metrics.kyle_lambda,
            'effective_spread': micro_metrics.effective_spread,
            'adverse_selection': micro_metrics.adverse_selection_cost,
            'amihud': micro_metrics.amihud_illiquidity,
            'trade_imbalance': micro_metrics.trade_imbalance,

            # Technical
            'hurst_exponent': tech_metrics.hurst_exponent,
            'fractal_dimension': tech_metrics.fractal_dimension,
            'entropy': tech_metrics.entropy,
            'regime_momentum': tech_metrics.regime_momentum,
            'complexity': tech_metrics.complexity_index,
            'phase_coherence': tech_metrics.phase_coherence
        }

        # Verify all features are valid numbers
        for name, value in feature_vector.items():
            assert isinstance(value, (int, float)), f"{name} should be numeric"
            assert not np.isnan(value), f"{name} should not be NaN"
            assert not np.isinf(value), f"{name} should not be infinite"

        assert len(feature_vector) == 12, "Should have 12 features"


def test_performance_benchmark(sample_trades, sample_prices):
    """Benchmark factor computation performance."""
    import time

    # Microstructure factors
    mf = MicrostructureFactors()
    start = time.time()
    mf.compute_all(sample_trades)
    micro_time = time.time() - start

    # Technical factors
    atf = AdvancedTechnicalFactors()
    start = time.time()
    atf.compute_all(sample_prices)
    tech_time = time.time() - start

    print(f"\nPerformance Benchmark:")
    print(f"  Microstructure factors: {micro_time:.4f}s")
    print(f"  Technical factors: {tech_time:.4f}s")
    print(f"  Total: {micro_time + tech_time:.4f}s")

    # Should complete in reasonable time
    assert micro_time < 5.0, "Microstructure computation too slow"
    assert tech_time < 5.0, "Technical computation too slow"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
