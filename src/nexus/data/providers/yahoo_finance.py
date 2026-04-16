"""
Yahoo Finance Data Provider.

Fetches free market data for any stock.
"""

import logging
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class YahooFinanceProvider:
    """Free market data from Yahoo Finance."""

    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_time: Dict[str, float] = {}
        self.cache_duration = 300  # 5 minutes

    def get_historical_data(
        self,
        symbol: str,
        days: int = 60,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data."""
        cache_key = f"{symbol}_{days}_{interval}"

        # Check cache
        if cache_key in self.cache:
            if time.time() - self.cache_time.get(cache_key, 0) < self.cache_duration:
                return self.cache[cache_key]

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            end = datetime.now()
            start = end - timedelta(days=days)

            df = ticker.history(start=start, end=end, interval=interval)

            if df.empty:
                return None

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]

            # Cache result
            self.cache[cache_key] = df
            self.cache_time[cache_key] = time.time()

            return df

        except Exception as e:
            logger.debug(f"Yahoo fetch failed for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Try different price fields
            price = info.get("regularMarketPrice")
            if price is None:
                price = info.get("currentPrice")
            if price is None:
                price = info.get("previousClose")

            return float(price) if price else None

        except Exception as e:
            logger.debug(f"Price fetch failed for {symbol}: {e}")
            return None

    def get_multiple_prices(
        self,
        symbols: List[str]
    ) -> Dict[str, float]:
        """Get current prices for multiple symbols."""
        prices = {}

        try:
            import yfinance as yf

            # Batch download
            data = yf.download(
                symbols,
                period="1d",
                interval="1d",
                progress=False
            )

            if "Close" in data.columns:
                for symbol in symbols:
                    try:
                        price = data["Close"][symbol].iloc[-1]
                        if not pd.isna(price):
                            prices[symbol] = float(price)
                    except:
                        pass
            elif len(symbols) == 1 and not data.empty:
                prices[symbols[0]] = float(data["Close"].iloc[-1])

        except Exception as e:
            logger.debug(f"Batch price fetch failed: {e}")

        return prices

    def get_fundamentals(self, symbol: str) -> Dict:
        """Get fundamental data for a symbol."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "avg_volume": info.get("averageVolume"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "roa": info.get("returnOnAssets"),
                "roe": info.get("returnOnEquity")
            }

        except Exception as e:
            logger.debug(f"Fundamentals fetch failed for {symbol}: {e}")
            return {}


# Global singleton
_provider: Optional[YahooFinanceProvider] = None


def get_yahoo_provider() -> YahooFinanceProvider:
    """Get or create global Yahoo provider."""
    global _provider
    if _provider is None:
        _provider = YahooFinanceProvider()
    return _provider
