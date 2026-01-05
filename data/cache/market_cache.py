"""
Market Data Caching System

High-performance cache for market data with TTL-based invalidation.
Eliminates duplicate downloads and reduces latency.
"""
import os
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MarketDataCache:
    """
    Thread-safe market data cache with TTL expiration.
    
    Features:
    - Local file-based storage
    - Configurable TTL (default 24 hours)
    - Automatic cache invalidation
    - Deduplication within runs
    - Cache statistics tracking
    """
    
    def __init__(self, cache_dir: str = "data/cache/market_data", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        
        # Runtime deduplication tracking
        self._runtime_cache = {}
        
        # Statistics
        self.stats = {"hits": 0, "misses": 0, "runtime_hits": 0}
    
    def _get_cache_key(self, ticker: str, start_date: str, end_date: str, data_type: str = "ohlcv") -> str:
        """Generate deterministic cache key."""
        key_str = f"{ticker}_{start_date}_{end_date}_{data_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get metadata file path."""
        return self.cache_dir / f"{cache_key}.meta.json"
    
    def _is_expired(self, metadata: dict) -> bool:
        """Check if cache entry is expired."""
        cached_time = datetime.fromisoformat(metadata["cached_at"])
        return datetime.now() - cached_time > self.ttl
    
    def get(self, ticker: str, start_date: str, end_date: str, data_type: str = "ohlcv") -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache if valid.
        
        Returns:
            DataFrame if cache hit, None if miss
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date, data_type)
        
        # Check runtime cache first (in-memory deduplication)
        if cache_key in self._runtime_cache:
            self.stats["runtime_hits"] += 1
            logger.debug(f"Runtime cache hit: {ticker} ({start_date} to {end_date})")
            return self._runtime_cache[cache_key].copy()
        
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)
        
        # Check disk cache
        if not cache_path.exists() or not meta_path.exists():
            self.stats["misses"] += 1
            return None
        
        try:
            # Load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Check expiration
            if self._is_expired(metadata):
                logger.debug(f"Cache expired: {ticker}")
                self.stats["misses"] += 1
                # Clean up expired entry
                cache_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None
            
            # Load data
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
            
            self.stats["hits"] += 1
            logger.debug(f"Cache hit: {ticker} ({start_date} to {end_date})")
            
            # Store in runtime cache
            self._runtime_cache[cache_key] = df.copy()
            
            return df.copy()
        
        except Exception as e:
            logger.warning(f"Cache read error for {ticker}: {e}")
            self.stats["misses"] += 1
            return None
    
    def set(self, ticker: str, start_date: str, end_date: str, data: pd.DataFrame, data_type: str = "ohlcv"):
        """
        Store data in cache.
        """
        if data.empty:
            return
        
        cache_key = self._get_cache_key(ticker, start_date, end_date, data_type)
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)
        
        try:
            # Store data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Store metadata
            metadata = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "data_type": data_type,
                "cached_at": datetime.now().isoformat(),
                "rows": len(data),
                "ttl_hours": self.ttl.total_seconds() / 3600
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Store in runtime cache
            self._runtime_cache[cache_key] = data.copy()
            
            logger.debug(f"Cached: {ticker} ({len(data)} rows)")
        
        except Exception as e:
            logger.error(f"Cache write error for {ticker}: {e}")
    
    def clear(self, older_than_hours: Optional[int] = None):
        """
        Clear cache entries.
        
        Args:
            older_than_hours: If specified, only clear entries older than this
        """
        cleared = 0
        for meta_file in self.cache_dir.glob("*.meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                should_clear = False
                if older_than_hours is None:
                    should_clear = True
                else:
                    cached_time = datetime.fromisoformat(metadata["cached_at"])
                    if datetime.now() - cached_time > timedelta(hours=older_than_hours):
                        should_clear = True
                
                if should_clear:
                    cache_key = meta_file.stem.replace(".meta", "")
                    cache_path = self._get_cache_path(cache_key)
                    cache_path.unlink(missing_ok=True)
                    meta_file.unlink(missing_ok=True)
                    cleared += 1
            
            except Exception as e:
                logger.warning(f"Error clearing cache entry: {e}")
        
        # Clear runtime cache
        self._runtime_cache.clear()
        
        logger.info(f"Cleared {cleared} cache entries")
        return cleared
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size_mb": sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl")) / (1024 * 1024)
        }


# Global cache instance
_global_cache = None

def get_cache(cache_dir: str = "data/cache/market_data", ttl_hours: int = 24) -> MarketDataCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MarketDataCache(cache_dir, ttl_hours)
    return _global_cache
