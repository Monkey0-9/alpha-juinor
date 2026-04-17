import pytest
from datetime import datetime
from src.nexus.models.market import MarketBar
from src.nexus.models.trade import Order, OrderSide, OrderType

def test_market_bar_validation():
    """Verify OHLC invariants in MarketBar."""
    valid_bar = MarketBar(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        open=150.0,
        high=155.0,
        low=149.0,
        close=152.0,
        volume=1000000
    )
    assert valid_bar.symbol == "AAPL"

    # Test High < Low violation
    with pytest.raises(ValueError, match="High.*must be >=.*Close"):
        MarketBar(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            open=150.0,
            high=140.0, # VIOLATION
            low=149.0,
            close=152.0,
            volume=1000000
        )

def test_order_creation():
    """Verify Order entity defaults and validation."""
    order = Order(
        symbol="TSLA",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0
    )
    assert order.status == "PENDING"
    assert len(order.order_id) > 20 # UUID generated
    
    with pytest.raises(ValueError):
        Order(symbol="TSLA", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=-1) # VIOLATION
