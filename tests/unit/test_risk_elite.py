import pytest
from datetime import datetime
from src.nexus.models.trade import Order, OrderSide, OrderType, PortfolioState, Position
from src.nexus.risk.rules import SectorConcentrationRule, LeverageRule

@pytest.fixture
def base_portfolio():
    return PortfolioState(
        cash=100000.0,
        equity=100000.0,
        positions={
            "AAPL": Position(symbol="AAPL", quantity=100, last_price=150.0), # $15,000
            "MSFT": Position(symbol="MSFT", quantity=50, last_price=300.0)   # $15,000
        }
    )

@pytest.fixture
def sector_map():
    return {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "JPM": "Financials",
        "GS": "Financials"
    }

def test_sector_concentration_violation(base_portfolio, sector_map):
    # Current Tech weight = 30% ($30k / $100k)
    # Rule: Max 40%
    rule = SectorConcentrationRule(sector_map=sector_map, max_sector_weight=0.40)
    
    # 1. Valid Order: JPM (Financials)
    order_jpm = Order(symbol="JPM", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10, limit_price=150.0)
    assert rule.validate(order_jpm, base_portfolio) is True
    
    # 2. Invalid Order: NVDA (Technology) - assume NVDA is Tech
    sector_map["NVDA"] = "Technology"
    # Buy $15,000 of NVDA -> Total Tech = $45,000 (45%)
    order_nvda = Order(symbol="NVDA", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100, limit_price=150.0)
    assert rule.validate(order_nvda, base_portfolio) is False
    assert "exceeds limit 40.00%" in rule.get_reason()

def test_leverage_rule(base_portfolio):
    # Current Gross Value = $30,000. Equity = $100,000. Leverage = 0.3x
    # Rule: Max 1.0x
    rule = LeverageRule(max_leverage=1.0)
    
    # Valid Order: Buy $50,000 more -> Total $80,000 (0.8x)
    ok_order = Order(symbol="TSLA", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100, limit_price=500.0)
    assert rule.validate(ok_order, base_portfolio) is True
    
    # Invalid Order: Buy $100,000 more -> Total $130,000 (1.3x)
    bad_order = Order(symbol="TSLA", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=200, limit_price=500.0)
    assert rule.validate(bad_order, base_portfolio) is False
    assert "exceeds limit 1.00x" in rule.get_reason()
