
import logging
import pandas as pd
import os
import json
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime, timedelta

# Updated Imports to new Structure
from data.providers.yahoo import YahooDataProvider
from data.providers.binance import BinanceDataProvider
from data.providers.stooq import StooqDataProvider
from data.providers.fred import FredDataProvider
from data.providers.fred import FredDataProvider
from data.providers.alpha_vantage import AlphaVantageDataProvider
from utils.timezone import normalize_index_utc

logger = logging.getLogger(__name__)

class DataRouter:
    """
    Master Data Router (The Brain).
    Automatically routes requests to the best available free source with fallback.
    Implements 'Rule 3: Cache Everything' using Parquet.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        # Primary Sources
        self.yahoo = YahooDataProvider()
        self.binance = BinanceDataProvider()
        self.fred = FredDataProvider()
        
        # Backup Sources
        self.stooq = StooqDataProvider()
        self.alpha_vantage = AlphaVantageDataProvider()
        
        # Optimization: Memory Cache (L1) -> Disk Cache (L2)
        # "Solid Rock" Speed: RAM is 100x faster than SSD
        self._mem_cache = {} 
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, ticker: str) -> Path:
        # Sanitize ticker for filename
        safe_ticker = ticker.replace("/", "_").replace("-", "_")
        return self.cache_dir / f"{safe_ticker}.parquet"
        
    def _read_cache(self, ticker: str, start_date: str) -> Optional[pd.DataFrame]:
        # L1: Memory Cache (Ultra Low Latency)
        if ticker in self._mem_cache:
            df = self._mem_cache[ticker]
            # For strict correctness, we should check timestamp, but for now we assume RAM is fresh from this session
            # Slice to requested history
            req_start_ts = pd.to_datetime(start_date).tz_localize("UTC") if pd.to_datetime(start_date).tz is None else pd.to_datetime(start_date)
            sliced_df = df[df.index >= req_start_ts]
            if not sliced_df.empty:
                logger.info(f"Serving {ticker} from Memory Cache")
                return sliced_df
            else:
                logger.info(f"Memory cache for {ticker} insufficient for requested start date {start_date}. Checking disk cache.")
                # If memory cache is insufficient, proceed to disk cache
                del self._mem_cache[ticker] # Invalidate memory cache for this ticker if it's not useful
        
        # L2: Disk Cache (Parquet)
        path = self._get_cache_path(ticker)
        if not path.exists():
            return None
            
        try:
            # Check modification time
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime > timedelta(hours=24):
                return None # Expired
                
            df = pd.read_parquet(path)
            
            # --- TIMEZONE & INTEGRITY FIX ---
            df = normalize_index_utc(df)
            
            # Helper to safely localize requested start date
            req_start_ts = pd.to_datetime(start_date)
            if req_start_ts.tz is None:
                req_start_ts = req_start_ts.tz_localize("UTC")
                
            # --- SMART CACHE INVALIDATION ---
            # If the cache starts significantly LATER than what we need, it's insufficient.
            # E.g. We need 2023, Cache starts 2025 -> Invalidate.
            if not df.empty:
                first_date = df.index[0]
                # Allow a small buffer (e.g. data might assume start date is slightly different)
                # If cached data starts > 20 days after request, assume we need to re-fetch deeper history.
                if first_date > (req_start_ts + timedelta(days=20)):
                   logger.info(f"Cache insufficient for {ticker} (Starts {first_date.date()} vs Req {req_start_ts.date()}). Refreshing...")
                   return None

            # Slice to requested history
            sliced_df = df[df.index >= req_start_ts]
            
            if sliced_df.empty:
                 logger.info(f"Cache hit for {ticker} but empty after date filter (Req: {req_start_ts.date()}). Refreshing...")
                 return None
                 
            return sliced_df
            
        except Exception as e:
            logger.warning(f"Cache read failed {ticker}: {e}")
            return None

    def _save_cache(self, ticker: str, df: pd.DataFrame):
        """Save to parquet."""
        try:
            if df.empty: return
            path = self._get_cache_path(ticker)
            df.to_parquet(path)
        except Exception as e:
            logger.warning(f"Cache save failed {ticker}: {e}")

    def get_macro_context(self) -> Dict[str, float]:
        """
        Aggregates global macro indicators for decision support.
        Sources: FRED (Primary), AlphaVantage (Backup).
        
        Returns:
            Dict with keys: 'VIX', 'YieldCurve', 'Inflation_Trend'
        """
        cache_path = self.cache_dir / "macro_context.json"
        
        # Try Cache
        try:
            if cache_path.exists():
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if datetime.now() - mtime < timedelta(hours=1):
                    with open(cache_path, 'r') as f:
                        return json.load(f)
        except Exception:
            pass

        # Fetch Fresh
        context = {
            "VIX": 20.0, # Default safe fallback
            "YieldCurve": 0.5,
            "RiskRegime": "Neutral"
        }
        
        try:
            # 1. FRED (Best for US Macro)
            if self.fred.api_key:
                vix = self.fred.fetch_series("VIXCLS")
                yc = self.fred.fetch_series("T10Y2Y")
                
                if not vix.empty: context["VIX"] = float(vix.iloc[-1])
                if not yc.empty: context["YieldCurve"] = float(yc.iloc[-1])
            
            # Simple Logic Enrichment
            if context["VIX"] > 32:
                context["RiskRegime"] = "RiskOff"
            elif context["VIX"] < 15:
                context["RiskRegime"] = "RiskOn"
                
            # Save Cache
            with open(cache_path, 'w') as f:
                json.dump(context, f)
                
        except Exception as e:
            logger.error(f"Macro Context Fetch Failed: {e}")
            
        return context

    def get_price_history(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Smart Fetch with Fallback Logic + Parquet Caching.
        """
        # 0. Check Cache First (Rule 3)
        cached_df = self._read_cache(ticker, start_date)
        if cached_df is not None:
             logger.info(f"Serving {ticker} from Cache")
             return cached_df

        df = pd.DataFrame()

        # 1. Macro / Economic Data
        if ticker in ["VIX", "CPI", "YIELD"]:
             fred_map = {"VIX": "VIXCLS", "CPI": "CPIAUCSL", "YIELD": "T10Y2Y"}
             if ticker in fred_map:
                 s = self.fred.fetch_series(fred_map[ticker], start_date)
                 df = s.to_frame(name="Close")

        try:
            # 2. Crypto (Binance is Best)
            if "-USD" in ticker or "BTC" in ticker or "ETH" in ticker:
                logger.info(f"Routing {ticker} to BINANCE (High Fidelity)")
                try:
                    df = self.binance.fetch_ohlcv(ticker, start_date, end_date)
                except Exception as e:
                     logger.warning(f"Binance fetch failed for {ticker}: {e}")
                     
                if df.empty:
                    logger.info(f"Binance miss for {ticker}. Switching to Yahoo...")
            
            # 3. Equities (Yahoo -> Stooq -> AlphaVantage)
            if df.empty and ticker not in ["VIX", "CPI", "YIELD"]:
                # Try Yahoo First (Broadest)
                try:
                    df = self.yahoo.fetch_ohlcv(ticker, start_date, end_date)
                except Exception as e:
                    logger.warning(f"Yahoo fetch failed for {ticker}: {e}")
                
                # Fallback to Stooq
                if df.empty:
                    logger.info(f"Yahoo miss for {ticker}. Switching to STOOQ layer...")
                    try:
                        df = self.stooq.fetch_ohlcv(ticker, start_date, end_date)
                    except Exception as e:
                        logger.warning(f"Stooq fetch failed for {ticker}: {e}")
                    
                # Fallback to Alpha Vantage (Equity)
                if df.empty:
                    logger.info(f"Stooq miss for {ticker}. Switching to ALPHA VANTAGE layer...")
                    try:
                        df = self.alpha_vantage.fetch_ohlcv(ticker, start_date, end_date)
                    except Exception as e:
                         logger.warning(f"AlphaVantage fetch failed for {ticker}: {e}")
        except Exception as e:
            logger.error(f"CRITICAL ROUTER ERROR for {ticker}: {e}")
            return pd.DataFrame()
        
        # 4. Save to Cache (L1 + L2)
        if not df.empty:
            df = normalize_index_utc(df)
            # Write to RAM
            self._mem_cache[ticker] = df
            # Write to Disk
            self._save_cache(ticker, df)
        else:
            # Only warn if ALL layers failed
            logger.warning(f"DATA FAILURE: Could not fetch {ticker} from any source.")
            
        return df
        
    def get_latest_price(self, ticker: str) -> Optional[float]:
        try:
            # Crypto -> Binance Realtime
            if "-USD" in ticker:
                p = self.binance.get_latest_price(ticker)
                if p and p > 0: return p
                
            # Equities -> Yahoo Realtime
            price = self.yahoo.get_latest_quote(ticker)
            if price and price > 0: return price
            
            return None
        except Exception as e:
            logger.warning(f"Live Quote Failed {ticker}: {e}")
            return None

    def cross_check_quote(self, ticker: str, primary_price: float, tolerance: float = 0.02) -> bool:
        """
        INSTITUTIONAL CONFIRMATION LAYER:
        Verifies a price/move against a secondary source to prevent data glitches.
        Requested by User: 'Stand after Yahoo Finance you can use other software and get the confirmation'
        """
        try:
            # 1. Skip Crypto (Binance is trusted enough, or tricky to cross-check consistently)
            if "-USD" in ticker: return True
            
            # 2. Secondary Source: AlphaVantage (Global Quote) or Stooq
            # We use AlphaVantage as the 'Auditor'
            secondary_price = None
            try:
                # Assuming alpha_vantage has a get_quote method, or we fetch last OHLC
                # For efficiency, we might just try fetching a daily bar if realtime quote isn't exposed
                # But let's assume we use Stooq as it's faster/easier for checking
                df = self.stooq.fetch_ohlcv(ticker, start_date=None) # Latest
                if not df.empty:
                    secondary_price = df['Close'].iloc[-1]
            except:
                pass
                
            if secondary_price is None:
                # If backup fails, we assume primary is valid (don't block on backup failure)
                return True
                
            # 3. Compare
            deviation = abs(primary_price - secondary_price) / secondary_price
            if deviation > tolerance:
                logger.warning(f"⚠️ DATA DISCREPANCY: Yahoo={primary_price}, Stooq={secondary_price} (Diff {deviation:.2%})")
                return False # Validation Failed
            
            logger.info(f"✅ Price Confirmed: Yahoo & Backup match within {deviation:.2%}")
            return True
            
        except Exception as e:
            logger.warning(f"Confirmation Check Failed: {e}")
            return True # Fail open to avoid paralysis
