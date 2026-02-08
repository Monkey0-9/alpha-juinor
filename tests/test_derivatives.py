"""
Test Suite for Derivatives Module
==================================

Comprehensive tests for volatility surface, exotic options, and delta hedging.
"""

import numpy as np
import pytest

from derivatives.exotic_options import (
    AsianOption,
    BarrierOption,
    ExoticOptionsEngine,
    LookbackOption,
    price_asian_option,
    price_barrier_option,
    price_lookback_option,
)
from derivatives.volatility_surface import (
    BlackScholesModel,
    OptionQuote,
    SABRModel,
    VolatilitySurface,
)
from strategies.delta_hedging import (
    DeltaHedgingStrategy,
    GammaScalpingStrategy,
    HedgePosition,
    OptionPosition,
)


class TestBlackScholes:
    """Test Black-Scholes pricing and IV calculation."""

    def test_call_price(self):
        """Test Black-Scholes call pricing."""
        price = BlackScholesModel.price(
            S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.2, option_type="call"
        )
        assert price > 0
        assert price < 100  # Call can't be worth more than stock

    def test_put_price(self):
        """Test Black-Scholes put pricing."""
        price = BlackScholesModel.price(
            S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.2, option_type="put"
        )
        assert price > 0

    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        # Price an option
        target_sigma = 0.25
        market_price = BlackScholesModel.price(
            S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=target_sigma, option_type="call"
        )

        # Recover IV
        iv = BlackScholesModel.implied_volatility(
            market_price, S=100, K=100, T=1.0, r=0.05, q=0.02, option_type="call"
        )

        assert iv is not None
        assert abs(iv - target_sigma) < 0.01  # Should match within 1%


class TestSABR:
    """Test SABR model."""

    def test_atm_vol(self):
        """Test SABR at-the-money volatility."""
        sabr = SABRModel(alpha=0.25, beta=0.5, rho=-0.3, nu=0.4)
        vol = sabr.implied_volatility(F=100, K=100, T=1.0)
        assert vol > 0
        assert vol < 2.0  # Reasonable volatility

    def test_calibration(self):
        """Test SABR calibration to market data."""
        F = 100
        strikes = np.array([90, 95, 100, 105, 110])
        T = 1.0

        # True SABR parameters
        true_sabr = SABRModel(alpha=0.2, beta=0.5, rho=-0.25, nu=0.3)
        market_vols = np.array(
            [true_sabr.implied_volatility(F, K, T) for K in strikes]
        )

        # Calibrate
        calibrated = SABRModel.calibrate(F, strikes, T, market_vols, beta=0.5)

        # Check calibrated vols match market
        calibrated_vols = np.array(
            [calibrated.implied_volatility(F, K, T) for K in strikes]
        )

        error = np.max(np.abs(calibrated_vols - market_vols))
        assert error < 0.01  # Should fit well


class TestExoticOptions:
    """Test exotic options pricing."""

    def test_barrier_option_knockout(self):
        """Test barrier option (knock-out)."""
        option = BarrierOption(
            S=100,
            K=100,
            H=90,
            T=1.0,
            r=0.05,
            q=0.02,
            sigma=0.2,
            option_type="call",
            barrier_type="down-and-out",
        )

        price = price_barrier_option(option, n_simulations=10000)
        assert price >= 0
        assert price <= 20  # Should be less than vanilla

    def test_asian_option(self):
        """Test Asian option pricing."""
        option = AsianOption(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            q=0.02,
            sigma=0.2,
            option_type="call",
            averaging_type="arithmetic",
            n_fixings=12,
        )

        price = price_asian_option(option, n_simulations=10000)
        assert price >= 0
        assert price <= 15  # Asian cheaper than vanilla

    def test_lookback_option(self):
        """Test lookback option pricing."""
        option = LookbackOption(
            S=100,
            K=None,
            T=1.0,
            r=0.05,
            q=0.02,
            sigma=0.2,
            option_type="call",
            lookback_type="floating",
        )

        price = price_lookback_option(option, n_simulations=10000)
        assert price >= 0
        assert price <= 30  # More valuable than vanilla


class TestDeltaHedging:
    """Test delta hedging strategy."""

    def test_calculate_greeks(self):
        """Test portfolio Greeks calculation."""
        strategy = DeltaHedgingStrategy(underlying_symbol="AAPL")

        # Add a long call
        option = OptionPosition(
            symbol="AAPL",
            strike=100,
            expiry_days=30,
            option_type="call",
            quantity=10,
            entry_price=5.0,
        )
        strategy.add_option(option)

        greeks = strategy.calculate_portfolio_greeks(
            underlying_price=100, current_vol=0.25
        )

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert greeks["delta"] > 0  # Long call has positive delta

    def test_hedge_recommendation(self):
        """Test hedge recommendation logic."""
        strategy = DeltaHedgingStrategy(underlying_symbol="AAPL")

        option = OptionPosition(
            symbol="AAPL",
            strike=100,
            expiry_days=30,
            option_type="call",
            quantity=10,
            entry_price=5.0,
        )
        strategy.add_option(option)

        rec = strategy.get_hedge_recommendation(underlying_price=100, current_vol=0.25)

        assert "current_delta" in rec
        assert "adjustment_needed" in rec
        assert rec["adjustment_needed"] < 0  # Should short stock to hedge long call


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
