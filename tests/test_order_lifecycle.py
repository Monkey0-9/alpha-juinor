"""
Order Lifecycle Tests.

Tests for OMS order lifecycle management.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.oms import (
    OMS, Order, Fill, OrderStatus, OrderSide, OrderType, TimeInForce, RiskCheckResult
)


class TestOrderLifecycle:
    """Test cases for order lifecycle management."""

    @pytest.fixture
    def temp_oms_db(self):
        """Create temporary OMS database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            yield f.name
            try:
                os.unlink(f.name)
            except:
                pass

    @pytest.fixture
    def oms(self, temp_oms_db):
        """Create OMS instance."""
        return OMS(db_path=temp_oms_db, max_order_value=100000, max_position_pct=0.1)

    def test_create_buy_order(self, oms):
        """Test creating a buy order."""
        order, risk_result = oms.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=185.50,
            broker="test"
        )

        assert order is not None
        assert risk_result.passed is True
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING

    def test_order_rejected_for_excessive_value(self, temp_oms_db):
        """Test order rejection for excessive order value."""
        oms = OMS(db_path=temp_oms_db, max_order_value=10000, max_position_pct=0.1)

        order, risk_result = oms.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            limit_price=200,
            broker="test"
        )

        assert order.status == OrderStatus.REJECTED
        assert risk_result.passed is False
        assert "exceeds limit" in risk_result.reason.lower()

    def test_submit_order(self, oms):
        """Test submitting an order."""
        order, _ = oms.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=185.50,
            broker="test"
        )

        success, message = oms.submit_order(order.order_id)

        assert success is True
        assert order.status == OrderStatus.SUBMITTED

    def test_cancel_order(self, oms):
        """Test cancelling an order."""
        order, _ = oms.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=185.50,
            broker="test"
        )

        oms.submit_order(order.order_id)

        success, message = oms.cancel_order(order.order_id, "Test cancellation")

        assert success is True
        assert order.status == OrderStatus.CANCELLED

    def test_process_fill_complete(self, oms):
        """Test processing a complete fill."""
        order, _ = oms.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            broker="test"
        )

        fill = Fill(
            fill_id="fill-001",
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=185.50,
            commission=1.0
        )

        success = oms.process_fill(fill)

        assert success is True
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100

    def test_process_fill_partial(self, oms):
        """Test processing a partial fill."""
        order, _ = oms.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            broker="test"
        )

        fill1 = Fill(
            fill_id="fill-001",
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=60,
            price=185.50
        )

        success1 = oms.process_fill(fill1)

        assert success1 is True
        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 60

        fill2 = Fill(
            fill_id="fill-002",
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=40,
            price=185.60
        )

        success2 = oms.process_fill(fill2)

        assert success2 is True
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100

    def test_order_properties(self, oms):
        """Test order property calculations."""
        order, _ = oms.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            broker="test"
        )

        assert order.remaining_quantity == 100
        assert order.is_active is True
        assert order.is_complete is False

        fill = Fill(
            fill_id="fill-001",
            order_id=order.order_id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=185.50
        )
        oms.process_fill(fill)

        assert order.remaining_quantity == 0
        assert order.is_complete is True

    def test_get_order(self, oms):
        """Test retrieving an order."""
        created_order, _ = oms.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            broker="test"
        )

        retrieved_order = oms.get_order(created_order.order_id)

        assert retrieved_order is not None
        assert retrieved_order.order_id == created_order.order_id

    def test_get_orders_filter(self, oms):
        """Test filtering orders."""
        oms.create_order("AAPL", OrderSide.BUY, 100, OrderType.MARKET, broker="test")
        oms.create_order("AAPL", OrderSide.SELL, 50, OrderType.MARKET, broker="test")
        oms.create_order("GOOG", OrderSide.BUY, 100, OrderType.MARKET, broker="test")

        aapl_orders = oms.get_orders(symbol="AAPL")
        assert len(aapl_orders) == 2

        buy_orders = oms.get_orders(side=OrderSide.BUY)
        assert len(buy_orders) == 2

    def test_order_to_dict(self, oms):
        """Test order serialization."""
        order, _ = oms.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=185.50,
            broker="test"
        )

        order_dict = order.to_dict()

        assert isinstance(order_dict, dict)
        assert order_dict["symbol"] == "AAPL"
        assert order_dict["status"] == "PENDING"

    def test_get_order_stats(self, oms):
        """Test order statistics."""
        oms.create_order("AAPL", OrderSide.BUY, 100, OrderType.MARKET, broker="test")
        oms.create_order("AAPL", OrderSide.SELL, 50, OrderType.MARKET, broker="test")

        stats = oms.get_order_stats()

        assert stats["total_orders"] == 2
        assert stats["active_orders"] == 2


class TestMarketImpact:
    """Test cases for market impact model."""

    def test_impact_estimate(self):
        """Test impact estimation."""
        from execution.market_impact import MarketImpactModel

        model = MarketImpactModel()

        impact = model.estimate_impact(
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=185.50,
            volatility=0.02,
            adv=10000000,
            order_type="MARKET"
        )

        assert impact.impact_bps > 0
        assert impact.market_impact > 0

    def test_optimal_execution(self):
        """Test optimal execution calculation."""
        from execution.market_impact import MarketImpactModel

        model = MarketImpactModel()

        result = model.calculate_optimal_execution(
            symbol="AAPL",
            side="BUY",
            quantity=10000,
            price=185.50,
            volatility=0.02,
            adv=10000000
        )

        assert "optimal_trajectory" in result
        assert "expected_cost" in result
        assert len(result["optimal_trajectory"]) > 0

    def test_liquidity_estimator(self):
        """Test liquidity estimation."""
        from execution.market_impact import LiquidityEstimator

        estimator = LiquidityEstimator()

        spread = estimator.estimate_effective_spread("AAPL", 185.50)

        assert spread > 0
        assert spread < 0.01

        depth = estimator.estimate_market_depth("AAPL", 185.50)

        assert "top_of_book" in depth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
