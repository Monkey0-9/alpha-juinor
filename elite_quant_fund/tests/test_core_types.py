"""
Comprehensive test suite for Elite Quant Fund - Core Types
All tests must pass with zero errors
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from elite_quant_fund.core.types import (
    Result, Side, MarketBar, AlphaSignal, Position, Portfolio,
    calc_cvar, ledoit_wolf_shrinkage, fractional_kelly_size
)


# ============================================================================
# RESULT MONAD TESTS
# ============================================================================

class TestResult:
    """Test Result monad for error handling"""
    
    def test_result_ok(self):
        """Test creating Ok result"""
        result = Result.ok(42)
        assert result.is_ok
        assert not result.is_err
        assert result.unwrap() == 42
    
    def test_result_err(self):
        """Test creating Err result"""
        result = Result.err("error message")
        assert result.is_err
        assert not result.is_ok
        assert result._error == "error message"
    
    def test_result_unwrap_or(self):
        """Test unwrap_or with default"""
        ok_result = Result.ok(42)
        err_result = Result.err("error")
        
        assert ok_result.unwrap_or(0) == 42
        assert err_result.unwrap_or(0) == 0
    
    def test_result_map(self):
        """Test map function"""
        result = Result.ok(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.unwrap() == 10
    
    def test_result_bind(self):
        """Test bind function"""
        result = Result.ok(5)
        bound = result.bind(lambda x: Result.ok(x * 2))
        assert bound.unwrap() == 10


# ============================================================================
# MARKET BAR TESTS
# ============================================================================

class TestMarketBar:
    """Test MarketBar with invariant enforcement"""
    
    def test_valid_bar(self):
        """Test creating valid OHLC bar"""
        bar = MarketBar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=98.0,
            close=102.0,
            volume=1000000
        )
        assert bar.symbol == "AAPL"
        assert bar.high >= bar.close
        assert bar.high >= bar.open
        assert bar.low <= bar.close
        assert bar.low <= bar.open
    
    def test_invalid_high_low(self):
        """Test that invalid high/low raises error"""
        with pytest.raises(ValueError):
            MarketBar(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=100.0,
                high=95.0,  # Invalid: high < open
                low=98.0,
                close=102.0,
                volume=1000000
            )
    
    def test_bar_properties(self):
        """Test bar computed properties"""
        bar = MarketBar(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=98.0,
            close=102.0,
            volume=1000000
        )
        
        assert bar.range == 7.0  # 105 - 98
        assert bar.body == 2.0   # |102 - 100|
        assert abs(bar.returns - 0.02) < 1e-10  # (102 - 100) / 100


# ============================================================================
# ALPHA SIGNAL TESTS
# ============================================================================

class TestAlphaSignal:
    """Test AlphaSignal with strength constraints"""
    
    def test_valid_signal(self):
        """Test creating valid alpha signal"""
        signal = AlphaSignal(
            symbol="AAPL",
            timestamp=datetime.now(),
            signal_type="MOMENTUM",
            strength=0.75,
            horizon=timedelta(hours=1),
            metadata={'confidence': 0.8}
        )
        assert signal.symbol == "AAPL"
        assert signal.strength == 0.75
        assert signal.is_bullish
        assert not signal.is_bearish
        assert signal.confidence == 0.75
    
    def test_signal_strength_bounds(self):
        """Test signal strength must be in [-1, 1]"""
        with pytest.raises(ValueError):
            AlphaSignal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type="MOMENTUM",
                strength=1.5,  # Invalid: > 1
                horizon=timedelta(hours=1)
            )
        
        with pytest.raises(ValueError):
            AlphaSignal(
                symbol="AAPL",
                timestamp=datetime.now(),
                signal_type="MOMENTUM",
                strength=-1.5,  # Invalid: < -1
                horizon=timedelta(hours=1)
            )


# ============================================================================
# POSITION TESTS
# ============================================================================

class TestPosition:
    """Test Position tracking"""
    
    def test_position_creation(self):
        """Test creating position"""
        position = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=100.0,
            entry_time=datetime.now(),
            current_price=105.0,
            unrealized_pnl=500.0,
            realized_pnl=0.0
        )
        
        assert position.symbol == "AAPL"
        assert position.market_value == 10500.0  # 100 * 105
        assert position.pnl_pct == 0.05  # (105 - 100) / 100
    
    def test_position_update(self):
        """Test updating position price"""
        position = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=100.0,
            entry_time=datetime.now(),
            current_price=100.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
        
        updated = position.update_price(110.0)
        assert updated.current_price == 110.0
        assert updated.unrealized_pnl == 1000.0  # 100 * (110 - 100)
        assert updated.pnl_pct == 0.10


# ============================================================================
# PORTFOLIO TESTS
# ============================================================================

class TestPortfolio:
    """Test Portfolio aggregation"""
    
    def test_portfolio_metrics(self):
        """Test portfolio metric calculations"""
        
        positions = {
            "AAPL": Position(
                symbol="AAPL",
                quantity=100,
                entry_price=100.0,
                entry_time=datetime.now(),
                current_price=105.0,
                unrealized_pnl=500.0,
                realized_pnl=0.0
            ),
            "MSFT": Position(
                symbol="MSFT",
                quantity=50,
                entry_price=200.0,
                entry_time=datetime.now(),
                current_price=210.0,
                unrealized_pnl=500.0,
                realized_pnl=0.0
            )
        }
        
        portfolio = Portfolio(
            timestamp=datetime.now(),
            positions=positions,
            cash=1_000_000,
            total_value=1_010_500  # Cash + positions
        )
        
        # AAPL: 100 * 105 = 10,500
        # MSFT: 50 * 210 = 10,500
        # Total exposure = 21,000
        assert portfolio.gross_exposure == 21000.0
        assert portfolio.net_exposure == 21000.0  # Both long
        
        # Leverage = gross / total
        expected_leverage = 21000.0 / 1_010_500
        assert abs(portfolio.leverage - expected_leverage) < 1e-6


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Test mathematical utility functions"""
    
    def test_calc_cvar(self):
        """Test CVaR calculation"""
        # Generate returns with known tail
        returns = np.array([-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        
        cvar_95 = calc_cvar(returns, alpha=0.05)
        
        # CVaR should be negative (tail risk)
        assert cvar_95 < 0
        
        # For this distribution, CVaR should capture the worst return
        assert cvar_95 <= np.min(returns)
    
    def test_fractional_kelly_size(self):
        """Test Kelly criterion sizing"""
        # Perfect coin flip: 50% edge, 1:1 odds
        size = fractional_kelly_size(edge=0.5, odds=1.0, fraction=0.3)
        
        # Kelly = 0.5 / 1 = 0.5
        # Fractional = 0.5 * 0.3 = 0.15
        assert abs(size - 0.15) < 1e-10
        
        # Zero edge should give zero size
        zero_size = fractional_kelly_size(edge=0.0, odds=1.0, fraction=0.3)
        assert zero_size == 0.0
        
        # Negative edge should give zero size
        negative_size = fractional_kelly_size(edge=-0.1, odds=1.0, fraction=0.3)
        assert negative_size == 0.0
    
    def test_ledoit_wolf_shrinkage(self):
        """Test Ledoit-Wolf covariance shrinkage"""
        # Generate correlated returns
        np.random.seed(42)
        n_assets = 5
        n_obs = 100
        
        # Create returns with some correlation
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=np.eye(n_assets) * 0.0001 + 0.00005,  # Slight correlation
            size=n_obs
        )
        
        shrunk_cov = ledoit_wolf_shrinkage(returns)
        
        # Check shape
        assert shrunk_cov.shape == (n_assets, n_assets)
        
        # Check positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(shrunk_cov)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors
        
        # Check diagonal (variances) are positive
        assert np.all(np.diag(shrunk_cov) > 0)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
