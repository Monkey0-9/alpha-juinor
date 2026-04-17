import pytest
from src.nexus.backtest.models import InstitutionalCostModel
from src.nexus.models.trade import Order, OrderSide, OrderType

def test_fixed_commission():
    # 1 bps = 0.0001
    model = InstitutionalCostModel(commission_bps=1.0, slippage_coeff=0.0)
    order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
    
    # Value = 100 * 150 = 15,000. Commission = 15,000 * 0.0001 = 1.5
    cost = model.calculate_cost(order, 150.0, volume=1000000)
    assert abs(cost - 1.5) < 1e-6

def test_almgren_chriss_slippage():
    # Slippage = coeff * sqrt(participation)
    model = InstitutionalCostModel(commission_bps=0.0, slippage_coeff=0.1)
    order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
    
    # Price = 100, Volume = 100,000
    # Value = 100,000. Participation = 1000 / 100,000 = 0.01
    # sqrt(0.01) = 0.1
    # slippage_bps = 0.1 * 0.1 = 0.01 (1%)
    # Total slippage = 100,000 * 0.01 = 1,000
    cost = model.calculate_cost(order, 100.0, volume=100000)
    assert abs(cost - 1000.0) < 1e-6

def test_execution_price_adjustment():
    model = InstitutionalCostModel(commission_bps=0.0, slippage_coeff=0.1)
    
    buy_order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
    exec_price_buy = model.get_execution_price(buy_order, 100.0, volume=100000)
    
    # 1,000 slippage on 1,000 units is 1.0 per unit. Since BUY, price goes UP.
    assert abs(exec_price_buy - 101.0) < 1e-6
    
    sell_order = Order(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1000)
    exec_price_sell = model.get_execution_price(sell_order, 100.0, volume=100000)
    
    # Since SELL, price goes DOWN.
    assert abs(exec_price_sell - 99.0) < 1e-6
