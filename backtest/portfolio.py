import pandas as pd
from typing import Dict, List
from .execution import Trade

class Portfolio:
    """
    Tracks holdings, cash, and total equity.
    """
    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {} # ticker -> quantity
        self.holdings_history: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
    def on_trade(self, trade: Trade):
        cost = trade.size * trade.price
        total_cost = cost + trade.commission
        
        self.cash -= total_cost
        
        current_pos = self.positions.get(trade.ticker, 0.0)
        self.positions[trade.ticker] = current_pos + trade.size
        
        # Clean up empty positions (avoid floating point dust)
        if abs(self.positions[trade.ticker]) < 1e-6:
            del self.positions[trade.ticker]
            
    def update_market_value(self, current_prices: Dict[str, float], timestamp):
        market_value = 0.0
        for ticker, qty in self.positions.items():
            if ticker in current_prices:
                market_value += qty * current_prices[ticker]
            else:
                # Need to handle missing price, maybe use last known?
                # For now assuming full data availability or 0
                pass 
                
        total_equity = self.cash + market_value
        
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": total_equity,
            "cash": self.cash,
            "market_value": market_value
        })
        
    def get_equity_curve_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.equity_curve)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df
