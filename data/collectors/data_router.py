# Premium Institutional Data Providers
from data.providers.bloomberg import BloombergDataProvider
from data.providers.reuters import ReutersDataProvider
# Alternative Data Providers
from data.providers.sentiment_provider import SentimentDataProvider
from data.providers.news_provider import NewsDataProvider
from data.providers.yahoo import YahooDataProvider
from data.providers.binance import BinanceDataProvider
from data.providers.coingecko import CoinGeckoDataProvider
from data.providers.polygon import PolygonDataProvider
from data.providers.finnhub import FinnhubDataProvider
from data.providers.twelvedata import TwelveDataDataProvider
from utils.timezone import normalize_index_utc
import pandas as pd
import logging
import os
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class DataRouter:
    """
    Intelligent data router that selects the best data provider based on ticker type,
    availability, and institutional requirements.
    """

    def __init__(self, max_workers: int = 10):
        # Initialize providers
        self.providers = {
            'yahoo': YahooDataProvider(),
            'binance': BinanceDataProvider(),
            'coingecko': CoinGeckoDataProvider(),
            'bloomberg': BloombergDataProvider(),
            'reuters': ReutersDataProvider(),
            'sentiment': SentimentDataProvider(),
            'news': NewsDataProvider(),
            'polygon': PolygonDataProvider(),
            'finnhub': FinnhubDataProvider(),
            'twelvedata': TwelveDataDataProvider()
        }

        # High-Speed Infrastructure
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Provider priority (Institutional Order: Polygon -> Finnhub -> TwelveData -> Yahoo)
        self.provider_priority = {
            'equity': ['polygon', 'finnhub', 'twelvedata', 'yahoo', 'bloomberg', 'reuters'],
            'crypto': ['binance', 'coingecko', 'bloomberg'],
            'forex': ['reuters', 'bloomberg', 'yahoo'],
            'commodity': ['bloomberg', 'reuters', 'yahoo']
        }

    def _classify_ticker(self, ticker: str) -> str:
        """Classify ticker by asset class."""
        ticker = ticker.upper()

        # Crypto detection
        if ticker in ['BTC', 'ETH', 'ADA', 'SOL', 'DOT'] or '-USD' in ticker:
            return 'crypto'

        # Forex detection
        if len(ticker) == 6 and ticker.endswith(('USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD')):
            return 'forex'

        # Commodity detection
        commodities = ['GC', 'SI', 'CL', 'NG', 'HG', 'PL']  # Gold, Silver, Oil, etc.
        if ticker in commodities or ticker.startswith(('CL', 'NG')):
            return 'commodity'

        # Default to equity
        return 'equity'

    def _get_best_provider(self, ticker: str, capability: str = None):
        """Get the best available provider for a ticker, checking capabilities."""
        asset_class = self._classify_ticker(ticker)
        priority_list = self.provider_priority.get(asset_class, ['yahoo'])

        for provider_name in priority_list:
            if provider_name in self.providers:
                p = self.providers[provider_name]
                
                # Mandatory Institutional Capability Check
                if capability == 'ohlcv' and not getattr(p, 'supports_ohlcv', True):
                    continue

                # Environment-based kill-switch
                env_key = f"{provider_name.upper()}_ENABLED"
                if os.getenv(env_key, "true").lower() == "false":
                    continue

                # Authentication Guard
                if hasattr(p, '_authenticated') and not p._authenticated:
                    continue

                return p

        return self.providers.get('yahoo')

    def get_price_history(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Trust Primary Provider, Fail-Fast if successful."""
        provider = self._get_best_provider(ticker, capability='ohlcv')
        if not provider: return pd.DataFrame()

        def _validate_schema(df):
            if df is None or df.empty: return False
            required = ['Close'] # Minimum requirement
            return all(col in df.columns for col in required)

        try:
            df = provider.fetch_ohlcv(ticker, start_date, end_date)
            if _validate_schema(df):
                df = normalize_index_utc(df)
                logger.debug(f"Fast-path: {ticker} from {provider.__class__.__name__}")
                return df
        except Exception as e:
            logger.debug(f"Primary fetch failed for {ticker}: {e}")

        # Fallback only if Primary failed (MANDATORY: Asset Class Isolation)
        asset_class = self._classify_ticker(ticker)
        allowed_fallbacks = self.provider_priority.get(asset_class, ['yahoo'])
        
        for fb_name in allowed_fallbacks:
            fb_p = self.providers.get(fb_name)
            if not fb_p or fb_p == provider or not getattr(fb_p, 'supports_ohlcv', False):
                continue
            
            try:
                df = fb_p.fetch_ohlcv(ticker, start_date, end_date)
                if not df.empty and 'Close' in df.columns:
                    df = normalize_index_utc(df)
                    logger.info(f"Fallback successful: {ticker} via {fb_name}")
                    return df
            except Exception as e:
                logger.debug(f"Fallback {fb_name} failed for {ticker}: {e}")
                continue

        return pd.DataFrame()

    def get_panel_parallel(self, tickers: List[str], start_date: str, end_date: str = None) -> pd.DataFrame:
        """Fetch multiple tickers in parallel using multithreading."""
        tasks = [(tk, start_date, end_date) for tk in tickers]
        
        def _fetch_wrap(args):
            tk, sd, ed = args
            return tk, self.get_price_history(tk, sd, ed)

        results = list(self.executor.map(_fetch_wrap, tasks))
        
        data = {}
        for tk, df in results:
            if not df.empty:
                for col in df.columns:
                    data[(tk, col)] = df[col]
        
        if not data: return pd.DataFrame()
        
        panel = pd.DataFrame(data)
        if not isinstance(panel.columns, pd.MultiIndex):
            panel.columns = pd.MultiIndex.from_tuples(panel.columns)
            
        return panel

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get latest price for a ticker (Trust Primary)."""
        provider = self._get_best_provider(ticker)
        if not provider: return None

        try:
            price = provider.get_latest_quote(ticker)
            if price is not None:
                logger.debug(f"Latest price for {ticker}: {price} from {provider.__class__.__name__}")
                return price
        except Exception as e:
            logger.debug(f"Primary price fetch failed for {ticker}: {e}")

        # Fallback (Asset Class Isolation)
        asset_class = self._classify_ticker(ticker)
        allowed_fallbacks = self.provider_priority.get(asset_class, ['yahoo'])
        
        for fb_name in allowed_fallbacks:
            fb_p = self.providers.get(fb_name)
            if not fb_p or fb_p == provider: continue
            try:
                price = fb_p.get_latest_quote(ticker)
                if price is not None:
                    return price
            except Exception:
                continue

        return None

    def get_latest_prices_parallel(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch latest prices for multiple tickers in parallel."""
        results = list(self.executor.map(self.get_latest_price, tickers))
        return {tk: pr for tk, pr in zip(tickers, results) if pr is not None}

    async def get_latest_price_async(self, ticker: str) -> Optional[float]:
        """Async version of get_latest_price."""
        import asyncio
        return await asyncio.to_thread(self.get_latest_price, ticker)

    def cross_check_quote(self, ticker: str, original_price: float, tolerance: float = 0.05) -> bool:
        """
        Cross-checks a price with a secondary provider.
        Returns True if a secondary price is within tolerance.
        """
        provider = self._get_best_provider(ticker)
        
        # Try fallbacks for cross-check (MANDATORY: Asset Class Isolation)
        asset_class = self._classify_ticker(ticker)
        allowed_names = self.provider_priority.get(asset_class, ['yahoo'])

        for name in allowed_names:
            fb_p = self.providers.get(name)
            if not fb_p or fb_p == provider: continue
            if not getattr(fb_p, 'supports_latest_quote', False): continue
            
            try:
                check_price = fb_p.get_latest_quote(ticker)
                if check_price is not None:
                    diff = abs(check_price - original_price) / original_price
                    if diff <= tolerance:
                        logger.info(f"Cross-check PASSED for {ticker} via {name} (diff: {diff:.2%})")
                        return True
                    else:
                        logger.warning(f"Cross-check FAILED for {ticker} via {name} (diff: {diff:.2%})")
            except Exception as e:
                logger.debug(f"Cross-check error on {name} for {ticker}: {e}")
                continue
        return False

    def get_macro_context(self) -> dict:
        """Get macroeconomic context data."""
        macro_data = {}

        # Try Bloomberg for institutional macro data
        if 'bloomberg' in self.providers:
            try:
                # This would fetch FRED data, interest rates, etc.
                macro_data.update(self.providers['bloomberg'].get_institutional_data('SPX', 'macro'))
            except Exception as e:
                logger.warning(f"Failed to get macro data from Bloomberg: {e}")

        # Fallback to other providers
        # For now, return empty dict - would need to implement macro data fetching
        return macro_data
