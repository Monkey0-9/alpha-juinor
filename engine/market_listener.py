
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from utils.time import get_now_utc
import asyncio

logger = logging.getLogger("MarketListener")

class MarketListener:
    """
    Real-Time Market Surveillance Component.
    Monitors market conditions via adaptive polling and triggers events.
    Designed to be lightweight and respectful of API limits.
    """
    def __init__(self, data_router, tickers: List[str]):
        self.router = data_router
        self.tickers = tickers

        # State
        self.last_prices: Dict[str, float] = {}
        self.last_poll_time: Dict[str, float] = {}
        self.volatility_window: Dict[str, List[float]] = {t: [] for t in tickers}

        # Configuration (Adaptive Intervals)
        self.crypto_interval = 10.0  # Poll crypto every 10s (Binance is robust)
        self.equity_interval = 60.0  # Poll equities every 60s (Yahoo rate limits)

        # Thresholds
        self.crash_threshold = 0.03  # 3% drop since last check
        self.vol_spike_threshold = 0.02 # 2% move in short window
        self.anomaly_threshold = 0.05   # 5% move triggers cross-check
        self.flash_crash_threshold = 0.10 # 10% move is a crash
        self.tick_counter = 0 # For visual heartbeat

        # Performance: True Async concurrency (no thread pool needed)
        # self.executor = ThreadPoolExecutor(...) -> Removed in favor of asyncio.create_task

    async def tick_async(self) -> List[str]:
        """
        Main heartbeat method (Optimized for Speed via asyncio).
        Parallelizes polling and analysis using native async tasks.
        """
        now = time.time()

        # 1. Identity tasks due for polling
        tickers_to_poll = []
        for ticker in self.tickers:
            is_crypto = "-USD" in ticker
            interval = self.crypto_interval if is_crypto else self.equity_interval
            last_t = self.last_poll_time.get(ticker, 0)
            if now - last_t >= interval:
                tickers_to_poll.append(ticker)

        if not tickers_to_poll:
            return []

        # 2. visual heartbeat
        self.tick_counter += 1
        if self.tick_counter % 5 == 0:
            logger.info(f"[SURVEILLANCE] Scanning {len(tickers_to_poll)}/{len(self.tickers)} assets in parallel...")

        # 3. Parallel Execution via asyncio.gather
        tasks = [self._poll_and_analyze_async(ticker) for ticker in tickers_to_poll]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        events = []
        for ticker, result in zip(tickers_to_poll, results):
            if isinstance(result, Exception):
                logger.debug(f"Listener Parallel Poll Failed {ticker}: {result}")
            elif result:
                events.append(result)

        return events

    async def _poll_and_analyze_async(self, ticker: str) -> Optional[str]:
        """Worker function for async scanner."""
        try:
            price = await self.router.get_latest_price_async(ticker)
            self.last_poll_time[ticker] = time.time()

            if price is None:
                return None

            event = self._analyze_tick(ticker, price)
            self.last_prices[ticker] = price
            return event
        except Exception as e:
            raise e

    def tick(self) -> List[str]:
        """Sync wrapper for compatibility, but recommend tick_async."""
        return asyncio.run(self.tick_async())

    def _analyze_tick(self, ticker: str, current_price: float) -> Optional[str]:
        """Detects anomalies between last price and current price."""
        if ticker not in self.last_prices:
            return None

        last_price = self.last_prices[ticker]
        if last_price <= 0: return None

        pct_change = (current_price - last_price) / last_price

        # 1. Anomaly Detection Logic (Institutional Requirement 3.1)
        # Any move > anomaly_threshold requires cross-check. Moves > flash_crash_threshold are treated as FLASH_CRASH.
        abs_move = abs(pct_change)

        if abs_move > self.anomaly_threshold:
            # Mandate secondary confirmation for significant moves
            cross_check_result = self.router.cross_check_quote(ticker, current_price)
            if cross_check_result == True:
                if pct_change < -self.flash_crash_threshold:
                    return f"FLASH_CRASH: {ticker} dropped {pct_change:.2%} (VERIFIED)"
                return f"VOLATILITY_SPIKE: {ticker} moved {pct_change:.2%} (VERIFIED)"
            elif cross_check_result == "INCONCLUSIVE":
                logger.warning(f"ANOMALY INCONCLUSIVE: {ticker} moved {pct_change:.2%} - insufficient cross-check data.")
                return f"VOLATILITY_SPIKE: {ticker} moved {pct_change:.2%} (INCONCLUSIVE)"
            else:
                # If unverified, we treat it as a glitch if it's extreme (>50%)
                if abs_move > 0.50:
                    logger.error(f"ANOMALY REJECTED: {ticker} moved {pct_change:.2%} but FAILED cross-check. Filtering as glitch.")
                    return None
                else:
                    logger.warning(f"Unverified move detected for {ticker} ({pct_change:.2%}). Cross-check failed.")
                    return None

        # 2. Normal Volatility/Breakout
        if abs_move > self.vol_spike_threshold:
            return f"VOLATILITY_SPIKE: {ticker} moved {pct_change:.2%}"

        return None
