import pandas as pd
import os
from pathlib import Path

class DataStore:
    """
    Manages local storage of market data using Parquet files.
    Ensures fast I/O and reproducibility.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, ticker: str, df: pd.DataFrame):
        """Save DataFrame to CSV."""
        if df.empty:
            return
            
        file_path = self.data_dir / f"{ticker}.csv"
        df.to_csv(file_path)
        print(f"   [Save] Saved {ticker} to {file_path}")
        
    def load(self, ticker: str) -> pd.DataFrame:
        """Load DataFrame from CSV."""
        file_path = self.data_dir / f"{ticker}.csv"
        
        if not file_path.exists():
            print(f"   [Error] {file_path} not found.")
            return pd.DataFrame()
            
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
        
    def exists(self, ticker: str) -> bool:
        return (self.data_dir / f"{ticker}.csv").exists()
