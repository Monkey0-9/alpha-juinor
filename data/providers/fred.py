
import os
import requests
import pandas as pd
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class FredDataProvider:
    """
    Federal Reserve Economic Data (FRED) Collector.
    Provides: VIX, Yield Curve (10Y-2Y), Inflation (CPI), Unemployment.
    Source: St. Louis Fed API (Free).
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            logger.warning("FRED API Key missing. Macro data (Regime Detection) will be unavailable.")
            
    def fetch_series(self, series_id: str, start_date: str = "2020-01-01") -> pd.Series:
        """
        Fetch a specific economic series.
        """
        if not self.api_key:
            return pd.Series()
            
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "observations" not in data:
                return pd.Series()
                
            df = pd.DataFrame(data["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Clean and set index
            df = df.dropna().set_index("date").sort_index()
            return df["value"]
            
        except Exception as e:
            logger.error(f"FRED Fetch Failed [{series_id}]: {e}")
            return pd.Series()

    def get_macro_regime_indicators(self) -> pd.DataFrame:
        """
        Fetches core regime indicators:
        - VIXCLS: Volatility (Risk)
        - T10Y2Y: Yield Curve Slope (Recession)
        - CPIAUCSL: CPI Inflation (Trend)
        """
        vix = self.fetch_series("VIXCLS")
        yield_curve = self.fetch_series("T10Y2Y")
        
        # Merge into single timeline
        df = pd.concat([vix, yield_curve], axis=1, keys=["VIX", "YieldCurve"])
        df = df.ffill().dropna()
        return df
