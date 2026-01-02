
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from .execution import TradeBlotter, RealisticExecutionHandler, Order, Trade
from data.provider import DataProvider

# Helper: robustly extract bar values and close-series for a ticker at timestamp
def _get_bar_and_history(dataframe, timestamp, ticker):
    """
    Returns (bar_dict, close_history_series) or (None, None) on failure.
    bar_dict keys: 'Open','High','Low','Close','Volume'
    """
    # Try MultiIndex columns first (ticker, field)
    try:
        if isinstance(dataframe.columns, pd.MultiIndex):
            # use .loc which returns scalars if present
            open_v = dataframe.loc[timestamp, (ticker, "Open")]
            high_v = dataframe.loc[timestamp, (ticker, "High")]
            low_v = dataframe.loc[timestamp, (ticker, "Low")]
            close_v = dataframe.loc[timestamp, (ticker, "Close")]
            vol_v = dataframe.loc[timestamp, (ticker, "Volume")]
            # history: all close values up to timestamp for this ticker
            close_hist = dataframe.loc[:timestamp, (ticker, "Close")].astype(float)
        else:
            # Try flat-name conventions: ticker_Close, ticker.Close, ticker Close, or single columns
            candidates = [
                f"{ticker}_Close", f"{ticker}.Close", f"{ticker} Close",
                f"{ticker}_close", f"{ticker}.close", f"{ticker} close"
            ]
            found = False
            for c in candidates:
                if c in dataframe.columns:
                    close_v = dataframe.loc[timestamp, c]
                    open_v = dataframe.loc[timestamp, c.replace("Close", "Open")] if c.replace("Close", "Open") in dataframe.columns else close_v
                    high_v = close_v
                    low_v = close_v
                    vol_v = dataframe.loc[timestamp, f"{ticker}_Volume"] if f"{ticker}_Volume" in dataframe.columns else 0
                    close_hist = dataframe.loc[:timestamp, c].astype(float)
                    found = True
                    break
            if not found:
                # If data has single asset columns like Open,High,Close,Volume only (one symbol)
                if set(["Open", "High", "Low", "Close", "Volume"]).issubset(set(dataframe.columns)):
                    open_v = dataframe.loc[timestamp, "Open"]
                    high_v = dataframe.loc[timestamp, "High"]
                    low_v = dataframe.loc[timestamp, "Low"]
                    close_v = dataframe.loc[timestamp, "Close"]
                    vol_v = dataframe.loc[timestamp, "Volume"]
                    close_hist = dataframe.loc[:timestamp, "Close"].astype(float)
                else:
                    return None, None
        # Normalize scalars: if a Series sneaks through, coerce to scalar
        if isinstance(close_v, (pd.Series, pd.DataFrame)):
            close_v = float(close_v.iloc[-1])
        if isinstance(open_v, (pd.Series, pd.DataFrame)):
            open_v = float(open_v.iloc[-1])
        if isinstance(high_v, (pd.Series, pd.DataFrame)):
            high_v = float(high_v.iloc[-1])
        if isinstance(low_v, (pd.Series, pd.DataFrame)):
            low_v = float(low_v.iloc[-1])
        if isinstance(vol_v, (pd.Series, pd.DataFrame)):
            vol_v = float(vol_v.iloc[-1])

        bar = {"Open": open_v, "High": high_v, "Low": low_v, "Close": close_v, "Volume": vol_v}
        return bar, close_hist
    except KeyError:
        return None, None
    except Exception:
        # defensive: return None so caller skips this ticker
        return None, None

class BacktestEngine:
    """
    Event-driven backtester with institutional trade recording and realistic execution.
    """
    def __init__(self, 
                 data_provider: DataProvider, 
                 initial_capital: float = 1_000_000,
                 execution_handler: RealisticExecutionHandler = None):
        
        self.provider = data_provider
        # Default to realistic execution if none provided
        self.execution_handler = execution_handler or RealisticExecutionHandler()
        self.initial_capital = initial_capital
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, float] = {} # ticker -> quantity
        self.blotter = TradeBlotter()
        self.equity_curve: List[Dict] = []
        self.tickers: List[str] = []
        
    def add_tickers(self, tickers: List[str]):
        self.tickers.extend(tickers)
        
    def _get_data(self, start_date: str) -> pd.DataFrame:
        """
        Fetches data for all tickers and combines into a single MultiIndex DataFrame.
        Columns: (Ticker, Field) where Field is Open, High, Low, Close, Volume
        """
        data_frames = {}
        for ticker in self.tickers:
            df = self.provider.fetch_ohlcv(ticker, start_date=start_date)
            if not df.empty:
                data_frames[ticker] = df
        
        if not data_frames:
            return pd.DataFrame()
            
        # Combine into MultiIndex DataFrame (Ticker, Field)
        # axis=1 concats columns. keys argument creates the top level index (Ticker).
        combined = pd.concat(data_frames.values(), axis=1, keys=data_frames.keys())
        
        # Depending on how concatenation happens, we might need to swap levels or sort
        # Currently: Level 0 = Ticker, Level 1 = Field (Open, Close...)
        # This matches access pattern: row[(ticker, "Close")]
        combined = combined.sort_index()

        # Ensure multiindex columns are sorted for fast access and to avoid lexsort warnings
        if isinstance(combined.columns, pd.MultiIndex):
            try:
                combined = combined.sort_index(axis=1)
            except Exception:
                pass  # sorting best-effort
        
        return combined

    def run(self, start_date: str, end_date: str = None, strategy_fn=None):
        """
        Main event loop.
        start_date: 'YYYY-MM-DD'
        strategy_fn: function(current_date, universe_prices, portfolio_state) -> List[Order]
        """
        print("   [Backtest] Loading data...")
        data = self._get_data(start_date)
        
        if data.empty:
            print("   [Error] No data available for backtest.")
            return

        print(f"   [Backtest] Running from {data.index[0]} to {data.index[-1]}...")

        for timestamp, row in data.iterrows():
            # Robust price extraction for strategy
            current_prices = {}
            for ticker in self.tickers:
                # Retrieve close price safely
                try:
                    # Try using the helper or just direct access if simple
                    if isinstance(data.columns, pd.MultiIndex):
                        price = row[(ticker, "Close")]
                    else:
                         # fallback for flat columns if any
                         price = row.get(f"{ticker}_Close", row.get("Close"))
                    
                    # Ensure scalar BEFORE checking notna if it might be a Series
                    if isinstance(price, (pd.Series, pd.DataFrame)):
                         if price.empty:
                             continue
                         price = price.iloc[0]
                         
                    if pd.notna(price):
                        current_prices[ticker] = float(price)
                except KeyError:
                    continue
            
            if not current_prices:
                continue
                
            # Store current prices state for portfolio valuation calls
            self.current_prices = current_prices

            # Pass simple state to strategy
            if strategy_fn:
                orders = strategy_fn(timestamp, current_prices, self)

                for order in orders:
                    self.blotter.record_order(order)
                    
                    # Robust extraction for execution
                    bar, price_history = _get_bar_and_history(data, timestamp, order.ticker)
                    
                    if bar is None or price_history is None:
                        # missing data for this ticker at this timestamp — skip
                        continue

                    price = bar.get("Close")
                    
                    # Defensive scalar check
                    if price is None or (isinstance(price, float) and np.isnan(price)):
                        continue

                    trade = self.execution_handler.fill_order(
                        order=order,
                        bar=bar,
                        price_history=price_history,
                    )

                    if trade:
                        self._apply_trade(trade)
                        self.blotter.record_trade(trade)

            self._record_equity(timestamp, current_prices)
            
        print("   [Backtest] Completed.")

    def _apply_trade(self, trade: Trade):
        self.cash -= trade.quantity * trade.fill_price
        self.cash -= trade.cost
        self.positions[trade.ticker] = self.positions.get(trade.ticker, 0.0) + trade.quantity

    def market_value(self) -> float:
        """
        Calculate current market value of equity (Cash + MTM Positions).
        Requires self.current_prices to be set (inside run loop).
        """
        equity = self.cash
        if not hasattr(self, "current_prices"):
            return equity
            
        for ticker, qty in self.positions.items():
            if qty != 0:
                price = self.current_prices.get(ticker, 0.0)
                equity += qty * price
        return equity

    def _record_equity(self, timestamp: datetime, current_prices: Dict[str, float]):
        # Use simple method or reuse market_value if updating state
        # Here we reuse the logic explicitly to be safe
        equity = self.cash
        for ticker, qty in self.positions.items():
            if qty != 0:
                price = current_prices.get(ticker, 0.0)
                equity += qty * price
                
        self.equity_curve.append({
            "Date": timestamp,
            "Equity": equity,
            "market_value": equity, # Alias for main.py compatibility
            "Cash": self.cash
        })

    def get_results(self) -> pd.DataFrame:
        df = pd.DataFrame(self.equity_curve)
        if not df.empty:
            df.set_index("Date", inplace=True)
        return df

    # Proxy methods for Strategy to access portfolio state
    def get_position(self, ticker: str) -> float:
        return self.positions.get(ticker, 0.0)
 

    
