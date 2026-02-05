import logging
import pandas as pd
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os
from data.providers.base import DataProvider
from utils.timezone import normalize_index_utc

logger = logging.getLogger(__name__)

class SentimentDataProvider(DataProvider):
    """
    Alternative data provider for market sentiment analysis.
    Aggregates sentiment from social media, news, and analyst reports.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('SENTIMENT_API_KEY')
        self.session = None
        self._authenticated = False

        # Alternative data APIs (these would be actual service endpoints)
        self.sentiment_api = "https://api.alternative-data.com/sentiment"
        self.social_api = "https://api.social-sentiment.com"
        self.news_api = "https://api.news-sentiment.com"

        # Initialize connection
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize sentiment data API connections."""
        try:
            if not self.api_key:
                logger.info("Sentiment API key not provided. Provider will be unavailable.")
                return

            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })

            # Test connection
            response = self.session.get(f"{self.sentiment_api}/status")
            if response.status_code == 200:
                self._authenticated = True
                logger.info("Sentiment data provider initialized successfully")
            else:
                logger.warning(f"Sentiment API connection failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to initialize sentiment provider: {e}")
            self._authenticated = False

    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Not applicable for sentiment data - use get_sentiment_data instead.
        """
        logger.warning("SentimentDataProvider does not support OHLCV data. Use get_sentiment_data() instead.")
        return pd.DataFrame()

    def get_sentiment_data(self, ticker: str, start_date: str, end_date: str = None) -> Dict[str, Any]:
        """
        Fetch comprehensive sentiment data for a ticker.
        Returns aggregated sentiment metrics from multiple sources.
        """
        if not self._authenticated:
            logger.warning("Sentiment provider not authenticated")
            return {}

        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Fetch from multiple sentiment sources
            sentiment_data = {
                'ticker': ticker,
                'date_range': {'start': start_date, 'end': end_date},
                'social_sentiment': self._get_social_sentiment(ticker, start_date, end_date),
                'news_sentiment': self._get_news_sentiment(ticker, start_date, end_date),
                'analyst_sentiment': self._get_analyst_sentiment(ticker),
                'aggregated_score': 0.0,
                'confidence': 0.0
            }

            # Aggregate sentiment scores
            scores = []
            weights = []

            if sentiment_data['social_sentiment']:
                scores.append(sentiment_data['social_sentiment'].get('score', 0))
                weights.append(0.4)  # 40% weight for social

            if sentiment_data['news_sentiment']:
                scores.append(sentiment_data['news_sentiment'].get('score', 0))
                weights.append(0.4)  # 40% weight for news

            if sentiment_data['analyst_sentiment']:
                scores.append(sentiment_data['analyst_sentiment'].get('score', 0))
                weights.append(0.2)  # 20% weight for analysts

            if scores:
                sentiment_data['aggregated_score'] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                sentiment_data['confidence'] = len(scores) / 3.0  # Confidence based on data completeness

            return sentiment_data

        except Exception as e:
            logger.error(f"Error fetching sentiment data for {ticker}: {e}")
            return {}

    def _get_social_sentiment(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get sentiment from social media platforms."""
        try:
            params = {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'platforms': ['twitter', 'reddit', 'stocktwits']
            }

            response = self.session.get(f"{self.social_api}/sentiment", params=params)

            if response.status_code == 200:
                data = response.json()
                return {
                    'score': data.get('sentiment_score', 0),  # -1 to 1 scale
                    'volume': data.get('mention_volume', 0),
                    'platforms': data.get('platform_breakdown', {}),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"Social sentiment API failed: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error getting social sentiment for {ticker}: {e}")
            return {}

    def _get_news_sentiment(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get sentiment from news articles and financial media."""
        try:
            params = {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'sources': ['reuters', 'bloomberg', 'wsj', 'cnbc']
            }

            response = self.session.get(f"{self.news_api}/sentiment", params=params)

            if response.status_code == 200:
                data = response.json()
                return {
                    'score': data.get('sentiment_score', 0),
                    'article_count': data.get('article_count', 0),
                    'top_sources': data.get('source_breakdown', {}),
                    'key_themes': data.get('themes', []),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"News sentiment API failed: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error getting news sentiment for {ticker}: {e}")
            return {}

    def _get_analyst_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get sentiment from analyst ratings and reports."""
        try:
            params = {'ticker': ticker}

            response = self.session.get(f"{self.sentiment_api}/analyst-sentiment", params=params)

            if response.status_code == 200:
                data = response.json()
                return {
                    'score': data.get('consensus_score', 0),  # Based on buy/hold/sell ratings
                    'rating_distribution': data.get('rating_breakdown', {}),
                    'price_target': data.get('average_price_target'),
                    'analyst_count': data.get('analyst_count', 0),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"Analyst sentiment API failed: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error getting analyst sentiment for {ticker}: {e}")
            return {}

    def get_panel(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Not applicable for sentiment data."""
        logger.warning("SentimentDataProvider does not support panel data.")
        return pd.DataFrame()

    def get_sentiment_signal(self, ticker: str) -> float:
        """
        Get a normalized sentiment signal for trading decisions.
        Returns value between -1 (very bearish) and 1 (very bullish).
        """
        sentiment_data = self.get_sentiment_data(ticker, start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))

        if not sentiment_data or 'aggregated_score' not in sentiment_data:
            return 0.0

        # Normalize and weight the signal
        score = sentiment_data['aggregated_score']
        confidence = sentiment_data.get('confidence', 0.0)

        # Apply confidence weighting
        signal = score * confidence

        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, signal))
