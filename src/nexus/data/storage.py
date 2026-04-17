import pandas as pd
import os
from typing import List, Optional
from ..models.market import MarketBar
from ..core.context import engine_context

class MarketDataStore:
    """
    Handles persistence of market data to local Parquet files.
    Ensures data versioning and reproducible datasets.
    """
    def __init__(self, base_path: str = "data/parquet"):
        self.base_path = base_path
        self.logger = engine_context.get_logger("data_store")
        os.makedirs(base_path, exist_ok=True)

    def _get_file_path(self, symbol: str, interval: str) -> str:
        return os.path.join(self.base_path, f"{symbol.upper()}_{interval}.parquet")

    def save_bars(self, bars: List[MarketBar]):
        """Saves a list of validated MarketBars to Parquet."""
        if not bars:
            return

        symbol = bars[0].symbol
        interval = "1d" # Default for now
        file_path = self._get_file_path(symbol, interval)
        
        # Convert to DataFrame
        new_df = pd.DataFrame([bar.model_dump() for bar in bars])
        new_df.set_index('timestamp', inplace=True)
        
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            # Combine, deduplicate, and sort
            df = pd.concat([existing_df, new_df])
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
        else:
            df = new_df.sort_index()
            
        df.to_parquet(file_path)
        self.logger.info(f"Persisted {len(df)} bars for {symbol} to {file_path}")

    def load_bars(self, symbol: str, interval: str = "1d") -> List[MarketBar]:
        """Loads bars from local Parquet storage."""
        file_path = self._get_file_path(symbol, interval)
        if not os.path.exists(file_path):
            return []
            
        df = pd.read_parquet(file_path)
        bars = []
        for ts, row in df.iterrows():
            bars.append(MarketBar(
                symbol=symbol,
                timestamp=ts,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            ))
        return bars
