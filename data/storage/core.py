import pandas as pd
import os
from pathlib import Path

class DataStore:
    """
    Manages local storage of market data using Parquet files.
    Ensures fast I/O and institutional reproducibility.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, ticker: str, df: pd.DataFrame):
        """Save DataFrame to Parquet."""
        if df.empty:
            return
            
        file_path = self.data_dir / f"{ticker}.parquet"
        df.to_parquet(file_path)
        print(f"   [Save] Persisted {ticker} to {file_path}")
        
    def load(self, ticker: str) -> pd.DataFrame:
        """Load DataFrame from Parquet."""
        file_path = self.data_dir / f"{ticker}.parquet"
        
        if not file_path.exists():
            return pd.DataFrame()
            
        return pd.read_parquet(file_path)
        
    def exists(self, ticker: str) -> bool:
        return (self.data_dir / f"{ticker}.parquet").exists()
