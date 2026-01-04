import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from .execution import Trade
from portfolio.ledger import PortfolioLedger, PortfolioEvent, EventType

class Portfolio:
    """
    Tracks holdings, cash, and total equity using a double-entry ledger.
    """
    def __init__(self, initial_capital: float = 1_000_000):
        self.ledger = PortfolioLedger(initial_capital)
        
    @property
    def cash(self):
        return self.ledger.cash_book.balance
        
    @property
    def positions(self):
        return self.ledger.position_book.positions

    @property
    def total_equity(self):
        """Returns the last recorded equity value from the ledger."""
        if self.ledger.equity_curve:
            return self.ledger.equity_curve[-1]["equity"]
        return self.ledger.cash_book.balance

    def on_trade(self, trade: Trade):
        # In a real system, timestamp should come from the trade or market
        # For backtest, we might need to pass it or use a default
        timestamp = getattr(trade, 'timestamp', datetime.now())
        
        event = PortfolioEvent(
            timestamp=timestamp,
            event_type=EventType.ORDER_FILLED,
            ticker=trade.ticker,
            quantity=trade.quantity,
            price=trade.fill_price,
            commission=trade.commission
        )
        self.ledger.record_event(event)
            
    def update_market_value(self, current_prices: Dict[str, float], timestamp):
        """
        Updates the ledger with a new snapshot. 
        CRITICAL: current_prices must contain valid prices for all open positions.
        """
        self.ledger.create_snapshot(timestamp, current_prices)
        
    def get_equity_curve_df(self) -> pd.DataFrame:
        return self.ledger.get_equity_curve_df()
