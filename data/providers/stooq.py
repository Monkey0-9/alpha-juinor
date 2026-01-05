
import pandas as pd
import logging
import requests
from io import StringIO

logger = logging.getLogger(__name__)

class StooqDataProvider:
    """
    Stooq Data Provider.
    Downloads CSV directly. Good backup for Yahoo.
    """
    
    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        try:
            # Stooq format: joined by .US for US stocks
            if "." not in ticker: 
                symbol = f"{ticker}.US"
            else:
                symbol = ticker
                
            url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
            
            # Stooq returns a CSV download stream
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return pd.DataFrame()
            
            content = response.content.decode('utf-8')
            if "No data" in content:
                return pd.DataFrame()
                
            df = pd.read_csv(StringIO(content))
            
            # Stooq columns: Date, Open, High, Low, Close, Volume
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            # Filter by date
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
            return df[["Open", "High", "Low", "Close", "Volume"]]
            
        except Exception as e:
            logger.warning(f"Stooq Fetch Failed for {ticker}: {e}")
            return pd.DataFrame()
