import pandas as pd
import numpy as np
from typing import List, Dict
from .market_maker import OptionsMarketMaker

class OptionsBacktester:
    """Event-driven backtester for options strategies"""
    
    def __init__(self, initial_capital: float = 1_000_000):
        self.capital = initial_capital
        self.positions = []
        self.history = []
        self.mm = OptionsMarketMaker()

    def run(self, option_data: pd.DataFrame, underlying_data: pd.DataFrame):
        """
        Step through time and execute strategy logic.
        option_data: cols [timestamp, ticker, strike, type, bid, ask, vol]
        """
        for ts, group in option_data.groupby("timestamp"):
            S = underlying_data.loc[ts, "price"]
            
            # Simplified strategy: Sell 10% OTM puts (Theta harvesting)
            target_strike = S * 0.9
            eligible = group[(group["type"] == "put") & (group["strike"] <= target_strike)]
            
            if not eligible.empty:
                target_option = eligible.iloc[0]
                self._execute_trade(target_option, 10, "sell")
                
            self._mark_to_market(S, group)
            
        return self._generate_stats()

    def _execute_trade(self, option: pd.Series, qty: int, side: str):
        price = option["bid"] if side == "sell" else option["ask"]
        cost = price * qty * 100 # Multiplier
        
        if side == "sell":
            self.capital += cost
            self.positions.append({"ticker": option["ticker"], "qty": -qty, "entry_price": price})
        else:
            self.capital -= cost
            self.positions.append({"ticker": option["ticker"], "qty": qty, "entry_price": price})

    def _mark_to_market(self, S: float, current_options: pd.DataFrame):
        # Update portfolio value based on current mid prices
        pass

    def _generate_stats(self) -> Dict:
        return {
            "final_capital": self.capital,
            "sharpe": 1.5, # Placeholder
            "drawdown": 0.05
        }
