import logging
import pandas as pd
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os
from data.providers.base import DataProvider
from utils.timezone import normalize_index_utc

logger = logging.getLogger(__name__)

class ReutersDataProvider(DataProvider):
    """
    Institutional-grade Reuters data provider.
    Requires Reuters API access or Refinitiv Eikon subscription.
    """

    def __init__(self, api_key: Optional[str] = None, app_id: Optional[str] = None):
        self.api_key = api_key or os.getenv('REUTERS_API_KEY')
        self.app_id = app_id or os.getenv('REUTERS_APP_ID')
        self.session = None
        self._authenticated = False

        # Reuters/Refinitiv API endpoints
        self.base_url = "https://api.refinitiv.com"  # Refinitiv API
        self.auth_endpoint = f"{self.base_url}/auth/oauth2/v1/token"
        self.data_endpoint = f"{self.base_url}/data"

        # Initialize connection
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize Reuters/Refinitiv API connection."""
        try:
            if not self.api_key or not self.app_id:
                logger.info("Reuters API credentials not provided. Provider will be unavailable.")
                return

            self.session = requests.Session()

            # Authenticate with Reuters API
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.app_id,
                'client_secret': self.api_key,
                'scope': 'trapi'
            }

            response = self.session.post(self.auth_endpoint, data=auth_data)

            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data.get('access_token')
                self.session.headers.update({
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                })
                self._authenticated = True
                logger.info("Reuters API connection established successfully")
            else:
                logger.warning(f"Reuters API authentication failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to initialize Reuters connection: {e}")
            self._authenticated = False

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from Reuters/Refinitiv.
        Reuters uses RIC (Reuters Instrument Code) format.
        """
        if not self._authenticated:
            logger.warning("Reuters not authenticated, returning empty DataFrame")
            return pd.DataFrame()

        try:
            # Convert ticker to Reuters RIC format if needed
            ric = self._convert_to_ric(ticker)

            # Set default end_date if not provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Reuters API request for historical data
            request_data = {
                "universe": [ric],
                "fields": ["TR.OPENPRICE", "TR.HIGHPRICE", "TR.LOWPRICE", "TR.CLOSEPRICE", "TR.VOLUME"],
                "parameters": {
                    "SDate": start_date,
                    "EDate": end_date,
                    "Frq": "D"  # Daily frequency
                }
            }

            response = self.session.post(f"{self.data_endpoint}/historical-pricing/v1/views/interday-summaries",
                                       json=request_data)

            if response.status_code == 200:
                data = response.json()
                return self._parse_reuters_response(data)
            else:
                logger.warning(f"Reuters API request failed: {response.status_code} - {response.text}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching Reuters data for {ticker}: {e}")
            return pd.DataFrame()

    def _convert_to_ric(self, ticker: str) -> str:
        """Convert standard ticker to Reuters RIC format."""
        # Reuters RIC format examples:
        # AAPL -> AAPL.O (NYSE)
        # MSFT -> MSFT.O (NASDAQ)
        # SPY -> SPY.P (ARCA)
        # BTC -> BTC= (Crypto)

        if ticker.endswith('-USD') or ticker in ['BTC', 'ETH']:
            # Crypto tickers
            crypto_map = {
                'BTC': 'BTC=',
                'ETH': 'ETH=',
                'BTC-USD': 'BTC=',
                'ETH-USD': 'ETH='
            }
            return crypto_map.get(ticker, ticker)

        # For equities, add exchange suffix
        if '.' not in ticker:
            # Assume US equities
            return f"{ticker}.O"  # .O for NYSE/NASDAQ

        return ticker

    def _parse_reuters_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse Reuters API response into standardized OHLCV DataFrame."""
        try:
            if not data.get('data') or not data['data']:
                return pd.DataFrame()

            records = []
            for item in data['data'][0].get('dataPoints', []):
                record = {
                    'timestamp': pd.to_datetime(item['date']),
                    'Open': item.get('TR.OPENPRICE'),
                    'High': item.get('TR.HIGHPRICE'),
                    'Low': item.get('TR.LOWPRICE'),
                    'Close': item.get('TR.CLOSEPRICE'),
                    'Volume': item.get('TR.VOLUME', 0)
                }
                records.append(record)

            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = normalize_index_utc(df)

            # Ensure numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df.dropna()

        except Exception as e:
            logger.error(f"Error parsing Reuters response: {e}")
            return pd.DataFrame()

    def get_latest_quote(self, ticker: str) -> Optional[float]:
        """Get latest price quote from Reuters."""
        if not self._authenticated:
            return None

        try:
            ric = self._convert_to_ric(ticker)

            request_data = {
                "universe": [ric],
                "fields": ["CF_CLOSE"]
            }

            response = self.session.post(f"{self.data_endpoint}/quotes/v1/snapshots",
                                       json=request_data)

            if response.status_code == 200:
                data = response.json()
                if data.get('data') and data['data']:
                    return data['data'][0].get('CF_CLOSE')
            else:
                logger.warning(f"Reuters quote request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error getting Reuters quote for {ticker}: {e}")
            return None

    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch panel data for multiple tickers."""
        panel_data = {}

        for ticker in tickers:
            df = self.fetch_ohlcv(ticker, start_date, end_date)
            if not df.empty:
                # Create MultiIndex columns (Ticker, Field)
                for col in df.columns:
                    panel_data[(ticker, col)] = df[col]

        if not panel_data:
            return pd.DataFrame()

        panel = pd.DataFrame(panel_data)
        panel.columns = pd.MultiIndex.from_tuples(panel.columns)
        return panel

    def get_institutional_data(self, ticker: str, data_type: str = 'fundamental') -> Dict[str, Any]:
        """
        Fetch institutional-grade data from Reuters.
        Includes fundamentals, estimates, news, and analytics.
        """
        if not self._authenticated:
            return {}

        try:
            ric = self._convert_to_ric(ticker)

            endpoints = {
                'fundamental': f"{self.data_endpoint}/fundamentals/v1",
                'estimates': f"{self.data_endpoint}/estimates/v1",
                'news': f"{self.data_endpoint}/news/v1",
                'analytics': f"{self.data_endpoint}/analytics/v1"
            }

            endpoint = endpoints.get(data_type)
            if not endpoint:
                logger.warning(f"Unknown data type: {data_type}")
                return {}

            request_data = {"universe": [ric]}

            if data_type == 'news':
                request_data["parameters"] = {"count": 10}

            response = self.session.post(endpoint, json=request_data)

            if response.status_code == 200:
                return response.json().get('data', {})
            else:
                logger.warning(f"Reuters {data_type} request failed: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error fetching Reuters {data_type} data for {ticker}: {e}")
            return {}

    def get_news_sentiment(self, ticker: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get news sentiment analysis for a ticker.
        Reuters provides institutional-grade news analytics.
        """
        if not self._authenticated:
            return {}

        try:
            ric = self._convert_to_ric(ticker)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            request_data = {
                "universe": [ric],
                "fields": ["TR.NI", "TR.NISentiment", "TR.NIVolume"],
                "parameters": {
                    "SDate": start_date.strftime('%Y-%m-%d'),
                    "EDate": end_date.strftime('%Y-%m-%d')
                }
            }

            response = self.session.post(f"{self.data_endpoint}/news-analytics/v1/sentiment",
                                       json=request_data)

            if response.status_code == 200:
                data = response.json()
                return {
                    'sentiment_score': data.get('data', [{}])[0].get('TR.NISentiment', 0),
                    'news_volume': data.get('data', [{}])[0].get('TR.NIVolume', 0),
                    'news_items': data.get('data', [{}])[0].get('TR.NI', [])
                }
            else:
                logger.warning(f"Reuters news sentiment request failed: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error getting Reuters news sentiment for {ticker}: {e}")
            return {}
