from .base import DataProvider
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional
import time
import logging

logger = logging.getLogger(__name__)

class CoinGeckoDataProvider(DataProvider):
    """
    CoinGecko Data Provider for crypto fundamentals.
    Free API with rate limits (50 calls/minute, 10k/month).
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Simple ticker to CoinGecko ID mapping for common cryptos
    TICKER_MAP = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'ADA': 'cardano',
        'SOL': 'solana',
        'DOT': 'polkadot',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'AAVE': 'aave',
        'SUSHI': 'sushi',
        'COMP': 'compound-governance-token',
        'MKR': 'maker',
        'YFI': 'yearn-finance',
        'BAL': 'balancer',
        'CRV': 'curve-dao-token',
        'REN': 'ren',
        'KNC': 'kyber-network-crystal',
        'ZRX': '0x',
        'BAT': 'basic-attention-token',
        'OMG': 'omisego',
        'LRC': 'loopring',
        'REP': 'augur',
        'GNT': 'golem',
        'STORJ': 'storj',
        'ANT': 'aragon',
        'MLN': 'melon',
        'FUN': 'funfair',
        'WAVES': 'waves',
        'LSK': 'lisk',
        'STRAT': 'stratis',
        'ARK': 'ark',
        'XEM': 'nem',
        'QTUM': 'qtum',
        'BTG': 'bitcoin-gold',
        'ZEC': 'zcash',
        'DASH': 'dash',
        'XMR': 'monero',
        'ETC': 'ethereum-classic',
        'XRP': 'ripple',
        'LTC': 'litecoin',
        'BCH': 'bitcoin-cash',
        'BSV': 'bitcoin-cash-sv',
        'EOS': 'eos',
        'TRX': 'tron',
        'NEO': 'neo',
        'IOTA': 'iota',
        'XLM': 'stellar',
        'ADA': 'cardano',
        'BNB': 'binancecoin',
        'DOGE': 'dogecoin',
        'SHIB': 'shiba-inu',
        'MATIC': 'matic-network',
        'AVAX': 'avalanche-2',
        'FTM': 'fantom',
        'ONE': 'harmony',
        'ICP': 'internet-computer',
        'FIL': 'filecoin',
        'VET': 'vechain',
        'THETA': 'theta-token',
        'TRB': 'tellor',
        'REN': 'ren',
        'KAVA': 'kava',
        'RUNE': 'thorchain',
        'CAKE': 'pancakeswap-token',
        'SUSHI': 'sushi',
        'COMP': 'compound-governance-token',
        'MKR': 'maker',
        'YFI': 'yearn-finance',
        'BAL': 'balancer',
        'CRV': 'curve-dao-token',
        'REN': 'ren',
        'KNC': 'kyber-network-crystal',
        'ZRX': '0x',
        'BAT': 'basic-attention-token',
        'OMG': 'omisego',
        'LRC': 'loopring',
        'REP': 'augur',
        'GNT': 'golem',
        'STORJ': 'storj',
        'ANT': 'aragon',
        'MLN': 'melon',
        'FUN': 'funfair',
        'WAVES': 'waves',
        'LSK': 'lisk',
        'STRAT': 'stratis',
        'ARK': 'ark',
        'XEM': 'nem',
        'QTUM': 'qtum',
        'BTG': 'bitcoin-gold',
        'ZEC': 'zcash',
        'DASH': 'dash',
        'XMR': 'monero',
        'ETC': 'ethereum-classic',
        'XRP': 'ripple',
        'LTC': 'litecoin',
        'BCH': 'bitcoin-cash',
        'BSV': 'bitcoin-cash-sv',
        'EOS': 'eos',
        'TRX': 'tron',
        'NEO': 'neo',
        'IOTA': 'iota',
        'XLM': 'stellar',
        'ADA': 'cardano',
        'BNB': 'binancecoin',
        'DOGE': 'dogecoin',
        'SHIB': 'shiba-inu',
        'MATIC': 'matic-network',
        'AVAX': 'avalanche-2',
        'FTM': 'fantom',
        'ONE': 'harmony',
        'ICP': 'internet-computer',
        'FIL': 'filecoin',
        'VET': 'vechain',
        'THETA': 'theta-token',
        'TRB': 'tellor',
        'REN': 'ren',
        'KAVA': 'kava',
        'RUNE': 'thorchain',
        'CAKE': 'pancakeswap-token',
    }

    def __init__(self, enable_cache: bool = True, cache_ttl_hours: int = 24):
        self.enable_cache = enable_cache
        from data.cache.market_cache import get_cache
        self.cache = get_cache(ttl_hours=cache_ttl_hours) if enable_cache else None

    def _get_coingecko_id(self, ticker: str) -> Optional[str]:
        """Map ticker to CoinGecko ID."""
        ticker_upper = ticker.upper()
        return self.TICKER_MAP.get(ticker_upper)

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        # Try cache first
        if self.enable_cache:
            cached_data = self.cache.get(ticker, start_date, end_date)
            if cached_data is not None:
                print(f"   [Cache Hit] {ticker} from cache")
                return cached_data

        coingecko_id = self._get_coingecko_id(ticker)
        if not coingecko_id:
            print(f"   [Warning] No CoinGecko ID found for {ticker}")
            return pd.DataFrame()

        print(f"   [Download] Fetching {ticker} from CoinGecko...")

        # CoinGecko OHLC API: /coins/{id}/ohlc?vs_currency=usd&days=365
        # But days is limited, so we need to calculate days from start_date
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days = (end_dt - start_dt).days

        if days > 365:
            # CoinGecko free tier limits to 365 days, so fetch in chunks if needed
            # For simplicity, limit to 365 days
            days = 365

        try:
            url = f"{self.BASE_URL}/coins/{coingecko_id}/ohlc"
            params = {
                'vs_currency': 'usd',
                'days': days
            }
            resp = self.session.get(url, params=params, timeout=5)
            resp.raise_for_status()

            data = resp.json()
            if not data:
                return pd.DataFrame()

            # Parse OHLC data: [[timestamp, open, high, low, close], ...]
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # CoinGecko OHLC doesn't include volume, so add NaN
            df['volume'] = pd.NA

            # Filter by date range
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            # Standardize columns
            df = df[['open', 'high', 'low', 'close', 'volume']].rename(columns=str.capitalize)

            # Use centralized utility
            from utils.timezone import normalize_index_utc
            df = normalize_index_utc(df)

            # INSTITUTIONAL: Proactive Validation
            from data.validator import DataValidator
            df = DataValidator.validate_ohlc(df, ticker=ticker)

            # Cache the result
            if self.enable_cache and not df.empty:
                self.cache.set(ticker, start_date, end_date, df)

            return df

        except requests.exceptions.RequestException as e:
            logger.warning(f"CoinGecko network error for {ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"CoinGecko fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch and combine multiple tickers into a MultiIndex panel."""
        data = {}
        for ticker in tickers:
            df = self.fetch_ohlcv(ticker, start_date, end_date)
            if not df.empty:
                for col in df.columns:
                    data[(ticker, col)] = df[col]

        if not data:
            return pd.DataFrame()

        panel = pd.DataFrame(data)
        panel.columns = pd.MultiIndex.from_tuples(panel.columns)
        return panel
