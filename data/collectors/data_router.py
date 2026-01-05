# Premium Institutional Data Providers
from data.providers.bloomberg import BloombergDataProvider
from data.providers.reuters import ReutersDataProvider
# Alternative Data Providers
from data.providers.sentiment_provider import SentimentDataProvider
from data.providers.news_provider import NewsDataProvider
from utils.timezone import normalize_index_utc
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataRouter:
    """
    Intelligent data router that selects the best data provider based on ticker type,
    availability, and institutional requirements.
    """

    def __init__(self):
        # Initialize providers
        self.providers = {
            'bloomberg': BloombergDataProvider(),
            'reuters': ReutersDataProvider(),
            'sentiment': SentimentDataProvider(),
            'news': NewsDataProvider()
        }

        # Provider priority for different asset classes
        self.provider_priority = {
            'equity': ['bloomberg', 'reuters', 'yahoo'],
            'crypto': ['coingecko', 'binance', 'bloomberg'],
            'forex': ['reuters', 'bloomberg'],
            'commodity': ['bloomberg', 'reuters']
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

    def _get_best_provider(self, ticker: str):
        """Get the best available provider for a ticker."""
        asset_class = self._classify_ticker(ticker)
        priority_list = self.provider_priority.get(asset_class, ['yahoo'])

        for provider_name in priority_list:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                # Check if provider is available (has API keys, etc.)
                if hasattr(provider, '_authenticated') and provider._authenticated:
                    return provider
                elif not hasattr(provider, '_authenticated'):
                    # For providers without authentication check
                    return provider

        # Fallback to yahoo if available
        if 'yahoo' in self.providers:
            return self.providers['yahoo']

        # Last resort - return first available provider
        return next(iter(self.providers.values()))

    def get_price_history(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Get price history for a ticker using the best available provider."""
        provider = self._get_best_provider(ticker)

        try:
            df = provider.fetch_ohlcv(ticker, start_date, end_date)
            if not df.empty:
                df = normalize_index_utc(df)
                logger.info(f"Successfully fetched {len(df)} records for {ticker} from {provider.__class__.__name__}")
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker} from {provider.__class__.__name__}: {e}")

        # Fallback providers
        for fallback_provider in self.providers.values():
            if fallback_provider != provider:
                try:
                    df = fallback_provider.fetch_ohlcv(ticker, start_date, end_date)
                    if not df.empty:
                        df = normalize_index_utc(df)
                        logger.info(f"Fallback: Successfully fetched {len(df)} records for {ticker} from {fallback_provider.__class__.__name__}")
                        return df
                except Exception as e:
                    logger.warning(f"Fallback failed for {ticker} from {fallback_provider.__class__.__name__}: {e}")

        logger.error(f"No data available for {ticker} from any provider")
        return pd.DataFrame()

    def get_latest_price(self, ticker: str) -> float:
        """Get latest price for a ticker."""
        provider = self._get_best_provider(ticker)

        try:
            price = provider.get_latest_quote(ticker)
            if price is not None:
                logger.info(f"Latest price for {ticker}: {price} from {provider.__class__.__name__}")
                return price
        except Exception as e:
            logger.warning(f"Failed to get latest price for {ticker} from {provider.__class__.__name__}: {e}")

        # Fallback
        for fallback_provider in self.providers.values():
            if fallback_provider != provider:
                try:
                    price = fallback_provider.get_latest_quote(ticker)
                    if price is not None:
                        logger.info(f"Fallback latest price for {ticker}: {price} from {fallback_provider.__class__.__name__}")
                        return price
                except Exception as e:
                    logger.warning(f"Fallback failed for latest price {ticker} from {fallback_provider.__class__.__name__}: {e}")

        logger.error(f"No latest price available for {ticker}")
        return None

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
