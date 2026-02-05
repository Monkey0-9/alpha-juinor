"""
Smart Data Cache - Prevents Rate Limit Hits.

Strategy:
- Cache daily data for 24 hours (no re-fetch needed)
- Cache intraday data for configurable period
- One fetch per symbol per day = ZERO rate limit issues
- High performance: all reads from memory/disk cache
"""

import os
import json
import time
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SmartDataCache:
    """
    Aggressive caching to minimize API calls.

    - Daily data cached for 24 hours
    - Intraday data cached for 1 hour
    - Memory + Disk hybrid for speed + persistence
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/market_data",
        daily_ttl_hours: int = 24,
        intraday_ttl_minutes: int = 60
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.daily_ttl = timedelta(hours=daily_ttl_hours)
        self.intraday_ttl = timedelta(minutes=intraday_ttl_minutes)

        # In-memory cache for ultra-fast access
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

        # Track fetches to prevent duplicates
        self.fetch_log: Dict[str, datetime] = {}

        logger.info(
            f"SmartDataCache initialized: daily_ttl={daily_ttl_hours}h, "
            f"intraday_ttl={intraday_ttl_minutes}m"
        )

    def _cache_key(
        self, symbol: str, data_type: str, start_date: str
    ) -> str:
        """Generate unique cache key."""
        raw = f"{symbol}_{data_type}_{start_date}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _disk_path(self, cache_key: str) -> Path:
        """Get disk cache path."""
        return self.cache_dir / f"{cache_key}.parquet"

    def get(
        self,
        symbol: str,
        data_type: str = "daily",
        start_date: str = ""
    ) -> Optional[pd.DataFrame]:
        """
        Get cached data if available and not expired.

        Returns None if cache miss or expired.
        """
        key = self._cache_key(symbol, data_type, start_date)
        ttl = self.daily_ttl if data_type == "daily" else self.intraday_ttl

        # 1. Check memory cache first (fastest)
        with self.lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if datetime.utcnow() - entry["timestamp"] < ttl:
                    logger.debug(f"Cache HIT (memory): {symbol}/{data_type}")
                    return entry["data"]

        # 2. Check disk cache
        disk_path = self._disk_path(key)
        if disk_path.exists():
            try:
                # Check file age
                file_age = datetime.utcnow() - datetime.fromtimestamp(
                    disk_path.stat().st_mtime
                )
                if file_age < ttl:
                    df = pd.read_parquet(disk_path)
                    # Warm memory cache
                    with self.lock:
                        self.memory_cache[key] = {
                            "data": df,
                            "timestamp": datetime.utcnow()
                        }
                    logger.debug(f"Cache HIT (disk): {symbol}/{data_type}")
                    return df
            except Exception as e:
                logger.warning(f"Disk cache read failed: {e}")

        logger.debug(f"Cache MISS: {symbol}/{data_type}")
        return None

    def set(
        self,
        symbol: str,
        data: pd.DataFrame,
        data_type: str = "daily",
        start_date: str = ""
    ):
        """Store data in cache (memory + disk)."""
        if data is None or data.empty:
            return

        key = self._cache_key(symbol, data_type, start_date)

        # Memory cache
        with self.lock:
            self.memory_cache[key] = {
                "data": data.copy(),
                "timestamp": datetime.utcnow()
            }

        # Disk cache (async would be better, but sync is safer)
        try:
            disk_path = self._disk_path(key)
            data.to_parquet(disk_path)
            logger.debug(f"Cache SET: {symbol}/{data_type}")
        except Exception as e:
            logger.warning(f"Disk cache write failed: {e}")

    def should_fetch(self, symbol: str, data_type: str = "daily") -> bool:
        """
        Check if we should fetch data from API.

        Returns False if:
        - Valid cache exists
        - Already fetched recently (prevents duplicate calls)
        """
        # Check cache first
        cached = self.get(symbol, data_type)
        if cached is not None and not cached.empty:
            return False

        # Check if already fetched today
        fetch_key = f"{symbol}_{data_type}_{datetime.utcnow().date()}"
        if fetch_key in self.fetch_log:
            # Already fetched today
            return False

        return True

    def record_fetch(self, symbol: str, data_type: str = "daily"):
        """Record that we fetched this symbol."""
        fetch_key = f"{symbol}_{data_type}_{datetime.utcnow().date()}"
        self.fetch_log[fetch_key] = datetime.utcnow()

    def get_multi(
        self, symbols: list, data_type: str = "daily"
    ) -> Dict[str, pd.DataFrame]:
        """Get multiple symbols from cache."""
        result = {}
        for sym in symbols:
            data = self.get(sym, data_type)
            if data is not None:
                result[sym] = data
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.parquet"))
        memory_size = len(self.memory_cache)

        return {
            "memory_entries": memory_size,
            "disk_files": len(disk_files),
            "disk_size_mb": sum(f.stat().st_size for f in disk_files) / 1024 / 1024,
            "fetches_today": len([
                k for k in self.fetch_log.keys()
                if str(datetime.utcnow().date()) in k
            ])
        }

    def clear_expired(self):
        """Clear expired cache entries."""
        now = datetime.utcnow()

        # Memory
        with self.lock:
            expired = [
                k for k, v in self.memory_cache.items()
                if now - v["timestamp"] > self.daily_ttl
            ]
            for k in expired:
                del self.memory_cache[k]

        # Disk
        for f in self.cache_dir.glob("*.parquet"):
            try:
                file_age = now - datetime.fromtimestamp(f.stat().st_mtime)
                if file_age > self.daily_ttl:
                    f.unlink()
            except Exception:
                pass

        logger.info(f"Cache cleanup: removed {len(expired)} memory entries")


# Global singleton
_data_cache: Optional[SmartDataCache] = None


def get_data_cache() -> SmartDataCache:
    """Get or create global data cache."""
    global _data_cache
    if _data_cache is None:
        _data_cache = SmartDataCache()
    return _data_cache
