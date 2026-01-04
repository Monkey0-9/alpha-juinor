import pytest
from datetime import datetime
from portfolio.ledger import PortfolioLedger, PortfolioEvent, EventType
from backtest.portfolio import Portfolio
from backtest.execution import Trade

def test_ledger_initial_capital():
    ledger = PortfolioLedger(initial_capital=100000)
    assert ledger.cash_book.balance == 100000
    assert len(ledger.position_book.positions) == 0

def test_ledger_order_fill():
    ledger = PortfolioLedger(initial_capital=100000)
    event = PortfolioEvent(
        timestamp=datetime.now(),
        event_type=EventType.ORDER_FILLED,
        ticker="AAPL",
        quantity=10,
        price=150.0,
        commission=10.0
    )
    ledger.record_event(event)
    
    # 10 * 150 + 10 = 1510
    assert ledger.cash_book.balance == 100000 - 1510
    assert ledger.position_book.positions["AAPL"] == 10
    assert ledger.position_book.cost_basis["AAPL"] == 1500

def test_ledger_sell_pnl():
    ledger = PortfolioLedger(initial_capital=100000)
    now = datetime.now()
    
    # Buy 10 @ 150
    ledger.record_event(PortfolioEvent(
        timestamp=now,
        event_type=EventType.ORDER_FILLED,
        ticker="AAPL",
        quantity=10,
        price=150.0,
        commission=0.0
    ))
    
    # Sell 5 @ 160
    ledger.record_event(PortfolioEvent(
        timestamp=now,
        event_type=EventType.ORDER_FILLED,
        ticker="AAPL",
        quantity=-5,
        price=160.0,
        commission=0.0
    ))
    
    # PnL = (160 - 150) * 5 = 50
    assert ledger.pnl_book.realized_pnl == 50.0
    assert ledger.position_book.positions["AAPL"] == 5
    assert ledger.position_book.cost_basis["AAPL"] == 750.0 # 5 * 150

def test_portfolio_integration():
    portfolio = Portfolio(initial_capital=100000)
    trade = Trade(
        trade_id="t1",
        order_id="o1",
        ticker="TSLA",
        quantity=10,
        fill_price=200.0,
        expected_price=200.0,
        market_impact=0.0,
        slippage=0.0,
        commission=5.0,
        cost=2005.0,
        timestamp=datetime.now()
    )
    
    portfolio.on_trade(trade)
    assert portfolio.cash == 100000 - (2000 + 5)
    assert portfolio.positions["TSLA"] == 10
    
    # Snapshot
    portfolio.update_market_value({"TSLA": 210.0}, datetime.now())
    df = portfolio.get_equity_curve_df()
    assert len(df) == 1
    # TSLA MV = 2100, Cash = 97995 -> Equity = 100095
    assert df.iloc[0]["equity"] == 100095.0
