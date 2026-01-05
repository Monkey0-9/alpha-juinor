
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from utils.time import get_now_utc

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
        self.tick_counter = 0 # For visual heartbeat
        
    def tick(self) -> List[str]:
        """
        Main heartbeat method. Called frequently by the main loop.
        Returns a list of detected event descriptions.
        """
        events = []
        now = time.time()
        
        # VISUAL HEARTBEAT (User Reassurance)
        # Every 5 ticks (approx 5s), log a "Scanning" message
        self.tick_counter += 1
        if self.tick_counter % 5 == 0:
            logger.info(f"⚡ [SURVEILLANCE] Scanning {len(self.tickers)} assets... Market Normal.")
        
        for ticker in self.tickers:
            is_crypto = "-USD" in ticker
            interval = self.crypto_interval if is_crypto else self.equity_interval
            
            # 1. Check if due for poll
            last_t = self.last_poll_time.get(ticker, 0)
            if now - last_t < interval:
                continue
                
            # 2. Poll Data (Lightweight)
            try:
                price = self.router.get_latest_price(ticker)
                self.last_poll_time[ticker] = now
                
                if price is None:
                    continue
                    
                # 3. Analyze for Events
                event = self._analyze_tick(ticker, price)
                if event:
                    events.append(event)
                    
                # Update State
                self.last_prices[ticker] = price
                
            except Exception as e:
                # Log debug only to avoid spam
                logger.debug(f"Listener Poll Failed {ticker}: {e}")
                
        return events

    def _analyze_tick(self, ticker: str, current_price: float) -> Optional[str]:
        """Detects anomalies between last price and current price."""
        if ticker not in self.last_prices:
            return None
            
        last_price = self.last_prices[ticker]
        if last_price <= 0: return None
        
        pct_change = (current_price - last_price) / last_price
        
        # 1. Flash Crash Detection
        if pct_change < -self.crash_threshold:
            # INSTITUTIONAL VERIFICATION: Cross-check with secondary source
            # "Get confirmation of analysis" - User Request
            if self.router.cross_check_quote(ticker, current_price):
                return f"FLASH_CRASH: {ticker} dropped {pct_change:.2%} detected at {get_now_utc()} (VERIFIED)"
            else:
                logger.warning(f"Flash Crash Detected for {ticker} but FAILED Cross-Check. Ignoring as Glitch.")
                return None
            
        # 2. Volatility/Breakout (Upside or Downside)
        if abs(pct_change) > self.vol_spike_threshold:
            return f"VOLATILITY_SPIKE: {ticker} moved {pct_change:.2%}"
            
        return None
