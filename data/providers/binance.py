
import requests
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Optional, List

logger = logging.getLogger(__name__)

class BinanceDataProvider:
    """
    Binance Public API Collector (Spot).
    Unlimited free data for crypto (limit 1000 bars/call, but performant).
    No API Key required for public data.
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def get_latest_price(self, symbol: str) -> float:
        try:
            # Binance uses 'BTCUSDT' format, standard is 'BTC-USD'
            formatted_sym = symbol.replace("-", "").replace("/", "").upper()
            if formatted_sym.endswith("USD"): formatted_sym += "T" # Usually USDT
            
            url = f"{self.BASE_URL}/ticker/price?symbol={formatted_sym}"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            return float(data['price'])
        except Exception as e:
            logger.warning(f"Binance Price Fetch Failed {symbol}: {e}")
            return 0.0

    def fetch_ohlcv(self, symbol: str, start_date: str, end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch OHLCV history with pagination support.
        """
        # Symbol normalization: "BTC-USD" -> "BTCUSDT"
        formatted_sym = symbol.replace("-", "").upper()
        if formatted_sym.endswith("USD"): formatted_sym += "T"
        
        # Convert date to timestamp ms
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000) if end_date else int(time.time() * 1000)
        
        all_candles = []
        current_start = start_ts
        
        # Loop for pagination (1000 limit)
        while True:
            params = {
                "symbol": formatted_sym,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 1000
            }
            try:
                resp = requests.get(f"{self.BASE_URL}/klines", params=params, timeout=10)
                if resp.status_code != 200:
                    logger.warning(f"Binance API Error: {resp.text}")
                    break
                    
                data = resp.json()
                if not data:
                    break
                    
                all_candles.extend(data)
                
                # Update start time for next batch
                last_open_time = data[-1][0]
                current_start = last_open_time + 1
                
                if current_start >= end_ts:
                    break
                    
                time.sleep(0.1) # Be nice
                
            except Exception as e:
                logger.error(f"Binance History Failed: {e}")
                break
        
        if not all_candles:
            return pd.DataFrame()
            
        # Parse
        # Binance Kline: [OpenTime, Open, High, Low, Close, Volume, ...]
        df = pd.DataFrame(all_candles, columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume", 
            "CloseTime", "QuoteAssetVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"
        ])
        
        df["date"] = pd.to_datetime(df["OpenTime"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
            
        df = df.set_index("date")
        return df[["Open", "High", "Low", "Close", "Volume"]]
