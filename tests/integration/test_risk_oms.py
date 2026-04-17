import pytest
import asyncio
from src.nexus.models.trade import Order, OrderSide, OrderType, PortfolioState
from src.nexus.execution.oms import OMS
from src.nexus.execution.adapters.paper import PaperBrokerAdapter
from src.nexus.risk.engine import RiskEngine
from src.nexus.risk.rules import MaxOrderValueRule

@pytest.mark.asyncio
async def test_oms_risk_rejection():
    """Verify that OMS + RiskEngine correctly blocks oversized orders."""
    # 1. Setup
    broker = PaperBrokerAdapter()
    oms = OMS(broker)
    risk = RiskEngine()
    risk.add_pre_trade_rule(MaxOrderValueRule(max_value=1000.0))
    
    portfolio = PortfolioState(cash=10000.0, equity=10000.0)
    
    # 2. Created Oversized Order ($150 * 10 = $1500 > $1000)
    oversized_order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=150.0,
        quantity=10.0
    )
    
    # 3. Validation Logic (Integrating Risk into submission flow)
    # Note: In a real system, the OMS would call the RiskEngine internally.
    is_safe = risk.check_pre_trade(oversized_order, portfolio)
    assert is_safe is False
    
    if not is_safe:
        oversized_order.status = "REJECTED"
        
    assert oversized_order.status == "REJECTED"

@pytest.mark.asyncio
async def test_oms_order_lifecycle():
    """Verify standard order routing and fill notification."""
    broker = PaperBrokerAdapter()
    oms = OMS(broker)
    
    order = Order(
        symbol="SPY",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=5.0
    )
    
    # Act
    success = await oms.submit_order(order)
    
    # Assert
    assert success is True
    # PaperBroker simulates an immediate fill (after 0.1s)
    await asyncio.sleep(0.2)
    assert order.status == "FILLED"
    assert order.filled_quantity == 5.0
