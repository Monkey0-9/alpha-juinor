import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from ..models.market import MarketBar
from .providers.base import DataProvider
from .storage import MarketDataStore
from .validator import DataValidator
from ..core.context import engine_context

class DataEngine:
    """
    Orchestrator for market data ingestion, validation, and retrieval.
    Handles caching, deduplication, and schema enforcement using MarketDataStore.
    """
    def __init__(self, cache_dir: str = "data/parquet"):
        self.cache_dir = cache_dir
        self.logger = engine_context.get_logger("data_engine")
        self.providers: Dict[str, DataProvider] = {}
        self.store = MarketDataStore(base_path=cache_dir)
        
    def add_provider(self, provider: DataProvider):
        self.providers[provider.get_name()] = provider
        self.logger.info(f"Registered data provider: {provider.get_name()}")

    async def get_data(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime, 
        provider_name: str = "yahoo",
        interval: str = "1d",
        use_cache: bool = True
    ) -> List[MarketBar]:
        """
        Retrieves data from cache or provider, ensuring validation and deduplication.
        """
        # 1. Check Cache
        if use_cache:
            cached_bars = self.store.load_bars(symbol, interval)
            # Filter for requested range
            matching_bars = [b for b in cached_bars if start <= b.timestamp <= end]
            if matching_bars:
                # If we have enough data (simplified check for now)
                # In real institutional systems, we'd check for gaps
                self.logger.info(f"Retrieved {len(matching_bars)} bars for {symbol} from cache")
                return matching_bars

        # 2. Fetch from Provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not registered.")
            
        provider = self.providers[provider_name]
        self.logger.info(f"Fetching {symbol} from {provider_name} ({start} to {end})")
        
        bars = await provider.get_historical_data(symbol, start, end, interval)
        
        if not bars:
            return []
            
        # 3. Enhanced Validation & Normalization
        # Convert MarketBars to DF for efficient validation then back
        df = pd.DataFrame([b.model_dump() for b in bars])
        # Standardize column naming for DataValidator
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df = DataValidator.validate_ohlc(df, ticker=symbol)
        
        # Enforce UTC normalization
        validated_bars = []
        for _, row in df.iterrows():
            ts = row['timestamp']
            if ts.tzinfo is None:
                ts = ts.replace(hour=23, minute=59, second=59).tz_localize('UTC')
            else:
                ts = ts.tz_convert('UTC')
                
            validated_bars.append(MarketBar(
                symbol=symbol,
                timestamp=ts,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume'])
            ))

        # 4. Save to Store (Cache)
        self.store.save_bars(validated_bars)
        
        self.logger.info(f"Retrieved and cached {len(validated_bars)} validated bars for {symbol}")
        
        return validated_bars
