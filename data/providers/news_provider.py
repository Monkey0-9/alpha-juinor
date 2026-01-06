import logging
import pandas as pd
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os
from data.providers.base import DataProvider
from utils.timezone import normalize_index_utc

logger = logging.getLogger(__name__)

class NewsDataProvider(DataProvider):
    """
    Alternative data provider for financial news and market-moving events.
    Provides real-time news feeds and historical news data for alpha generation.
    """

    # News capabilities
    supports_news = True

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NEWS_DATA_IO_KEY') or os.getenv('NEWS_API_KEY')
        self.session = None

        # News data APIs
        self.news_data_io = "https://newsdata.io/api/1/news"
        self.alpha_vantage_news = "https://www.alphavantage.co/query"

        # Initialize connection
        self._initialize_connection()

        # Disable if not authenticated
        if not self._authenticated:
            self.disabled = True

    def _initialize_connection(self):
        """Initialize news data API connections."""
        try:
            if not self.api_key:
                logger.warning("News API key not provided. Provider will be unavailable.")
                return

            self.session = requests.Session()
            # NewsData.io uses apikey parameter, not Bearer token in regular headers
            # But we keep session for connection pooling
            
            self._authenticated = True
            logger.info("News data provider initialized for NewsData.io")

        except Exception as e:
            logger.error(f"Failed to initialize news provider: {e}")
            self._authenticated = False

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Not applicable for news data - use get_news_data instead.
        """
        logger.warning("NewsDataProvider does not support OHLCV data. Use get_news_data() instead.")
        return pd.DataFrame()

    def get_news_data(self, ticker: str, start_date: str, end_date: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch news articles and market-moving events for a ticker.
        Returns list of news items with sentiment and impact scores.
        """
        if not self._authenticated:
            logger.warning("News provider not authenticated")
            return []

        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Try primary news API first
            news_items = self._fetch_financial_news(ticker, start_date, end_date, limit)

            # Fallback to Alpha Vantage if primary fails
            if not news_items:
                news_items = self._fetch_alpha_vantage_news(ticker, limit)

            # Enrich with sentiment analysis
            for item in news_items:
                item['sentiment_score'] = self._analyze_sentiment(item.get('title', '') + ' ' + item.get('summary', ''))
                item['market_impact'] = self._assess_market_impact(item)

            return news_items

        except Exception as e:
            logger.error(f"Error fetching news data for {ticker}: {e}")
            return []

    def _fetch_financial_news(self, ticker: str, start_date: str, end_date: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch news from NewsData.io API."""
        try:
            params = {
                'apikey': self.api_key,
                'q': ticker,
                'language': 'en',
                'category': 'business,technology'
            }

            response = self.session.get(self.news_data_io, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                # Normalize to standard format
                news_items = []
                for item in results:
                    news_items.append({
                        'title': item.get('title', ''),
                        'summary': item.get('description', ''),
                        'source': item.get('source_id', ''),
                        'url': item.get('link', ''),
                        'published_at': item.get('pubDate', ''),
                        'sentiment_score': 0.0, # Will be analyzed later
                        'market_impact': 0.0
                    })
                return news_items
            else:
                logger.warning(f"NewsData.io API failed: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error fetching NewsData.io news for {ticker}: {e}")
            return []

    def _fetch_alpha_vantage_news(self, ticker: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback to Alpha Vantage news API."""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.api_key,
                'limit': limit
            }

            response = requests.get(self.alpha_vantage_news, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                feed = data.get('feed', [])

                # Convert Alpha Vantage format to standard format
                news_items = []
                for item in feed:
                    news_items.append({
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'source': item.get('source', ''),
                        'url': item.get('url', ''),
                        'published_at': item.get('time_published', ''),
                        'sentiment_score': item.get('overall_sentiment_score', 0),
                        'relevance_score': item.get('relevance_score', 0)
                    })

                return news_items
            else:
                logger.warning(f"Alpha Vantage news API failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news for {ticker}: {e}")
            return []

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis - in production, use NLP models."""
        if not text:
            return 0.0

        # Basic keyword-based sentiment analysis
        positive_words = ['bullish', 'upgrade', 'beat', 'surprise', 'strong', 'growth', 'profit', 'rise', 'gain']
        negative_words = ['bearish', 'downgrade', 'miss', 'disappoint', 'weak', 'decline', 'loss', 'fall', 'drop']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0

        # Return normalized sentiment score (-1 to 1)
        return (positive_count - negative_count) / max(total_words, 1)

    def _assess_market_impact(self, news_item: Dict[str, Any]) -> float:
        """Assess potential market impact of news item."""
        impact_score = 0.0

        # Source credibility weight
        credible_sources = ['reuters', 'bloomberg', 'wsj', 'cnbc', 'ft']
        source = news_item.get('source', '').lower()
        if any(cred_source in source for cred_source in credible_sources):
            impact_score += 0.3

        # Title keywords indicating high impact
        title = news_item.get('title', '').lower()
        high_impact_keywords = ['earnings', 'guidance', 'merger', 'acquisition', 'lawsuit', 'fda', 'sec']
        if any(keyword in title for keyword in high_impact_keywords):
            impact_score += 0.4

        # Sentiment extremity
        sentiment = abs(news_item.get('sentiment_score', 0))
        impact_score += sentiment * 0.3

        return min(1.0, impact_score)

    def get_market_moving_news(self, tickers: List[str], hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Get high-impact news that could move markets.
        Filters for news with high relevance and impact scores.
        """
        all_news = []

        for ticker in tickers:
            news = self.get_news_data(ticker,
                                    start_date=(datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d %H:%M:%S'),
                                    limit=20)

            # Filter for high-impact news
            high_impact_news = [
                item for item in news
                if item.get('market_impact', 0) > 0.6 and abs(item.get('sentiment_score', 0)) > 0.3
            ]

            all_news.extend(high_impact_news)

        # Sort by impact score and recency
        all_news.sort(key=lambda x: (x.get('market_impact', 0), x.get('published_at', '')), reverse=True)

        return all_news[:20]  # Return top 20

    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Not applicable for news data."""
        logger.warning("NewsDataProvider does not support panel data.")
        return pd.DataFrame()

    def get_news_signal(self, ticker: str, lookback_days: int = 7) -> float:
        """
        Generate a news-based trading signal.
        Analyzes recent news sentiment and impact.
        """
        news_data = self.get_news_data(ticker,
                                     start_date=(datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d'),
                                     limit=100)

        if not news_data:
            return 0.0

        # Calculate weighted sentiment
        total_weight = 0
        weighted_sentiment = 0

        for item in news_data:
            impact = item.get('market_impact', 0.1)
            sentiment = item.get('sentiment_score', 0)

            # Weight by impact and recency (newer news has higher weight)
            published_at = item.get('published_at', '')
            if published_at:
                try:
                    published_time = pd.to_datetime(published_at)
                    hours_old = (datetime.now() - published_time).total_seconds() / 3600
                    recency_weight = max(0.1, 1.0 / (1.0 + hours_old / 24))  # Decay over days
                except:
                    recency_weight = 0.5
            else:
                recency_weight = 0.5

            weight = impact * recency_weight
            weighted_sentiment += sentiment * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Normalize to [-1, 1] range
        signal = weighted_sentiment / total_weight
        return max(-1.0, min(1.0, signal))
