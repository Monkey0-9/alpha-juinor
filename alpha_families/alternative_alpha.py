"""
Alternative Alpha Model - News, social media, and alternative data signals.

Combines sentiment from news articles, social media trends, search data,
and other non-traditional sources to generate alpha signals.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

from .base_alpha import BaseAlpha

logger = logging.getLogger(__name__)

class AlternativeAlpha(BaseAlpha):
    """
    Alternative Data Alpha Model.

    Sources:
    - News sentiment analysis
    - Social media sentiment
    - Search trends (Google Trends proxy)
    - Alternative data feeds
    """

    def __init__(self,
                 news_weight: float = 0.4,
                 social_weight: float = 0.3,
                 search_weight: float = 0.3,
                 sentiment_decay_days: int = 7):
        """
        Initialize Alternative Alpha.

        Args:
            news_weight: Weight for news sentiment
            social_weight: Weight for social media sentiment
            search_weight: Weight for search trends
            sentiment_decay_days: Days over which sentiment decays
        """
        super().__init__()

        self.news_weight = news_weight
        self.social_weight = social_weight
        self.search_weight = search_weight
        self.sentiment_decay_days = sentiment_decay_days

        # Sentiment tracking
        self.sentiment_history: Dict[str, List[Dict[str, Any]]] = {
            'news': [],
            'social': [],
            'search': []
        }

        # News sentiment keywords
        self.positive_keywords = [
            'bullish', 'strong', 'gains', 'rally', 'surge', 'boost', 'upgrade',
            'positive', 'optimistic', 'growth', 'earnings beat', 'record high'
        ]

        self.negative_keywords = [
            'bearish', 'weak', 'decline', 'drop', 'fall', 'sell', 'downgrade',
            'negative', 'pessimistic', 'recession', 'earnings miss', 'crash'
        ]

    def generate_signal(self,
                       market_data: pd.DataFrame,
                       regime_context: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate alternative data alpha signal.

        Args:
            market_data: OHLCV market data
            regime_context: Current market regime

        Returns:
            Signal dictionary with signal, confidence, and metadata
        """
        try:
            current_date = market_data.index[-1] if not market_data.empty else pd.Timestamp.now()

            # Generate sentiment from different sources
            news_sentiment = self._get_news_sentiment(current_date)
            social_sentiment = self._get_social_sentiment(current_date)
            search_sentiment = self._get_search_sentiment(current_date)

            # Combine sentiments
            combined_sentiment = (
                news_sentiment * self.news_weight +
                social_sentiment * self.social_weight +
                search_sentiment * self.search_weight
            )

            # Normalize to [-1, 1] range
            signal = np.clip(combined_sentiment, -1.0, 1.0)

            # Calculate confidence based on data availability
            confidence = self._calculate_confidence(news_sentiment, social_sentiment, search_sentiment)

            # Adjust for regime if provided
            if regime_context:
                signal, confidence = self._adjust_for_regime(signal, confidence, regime_context)

            # Store sentiment data
            self._store_sentiment_data(current_date, news_sentiment, social_sentiment, search_sentiment)

            return {
                'signal': float(signal),
                'confidence': float(confidence),
                'metadata': {
                    'news_sentiment': float(news_sentiment),
                    'social_sentiment': float(social_sentiment),
                    'search_sentiment': float(search_sentiment),
                    'combined_sentiment': float(combined_sentiment),
                    'regime_adjusted': regime_context is not None,
                    'sentiment_sources': 3
                }
            }

        except Exception as e:
            logger.error(f"Alternative alpha signal generation failed: {e}")
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }

    def _get_news_sentiment(self, current_date: pd.Timestamp) -> float:
        """
        Generate news sentiment signal.

        In a real implementation, this would analyze actual news feeds.
        Here we simulate based on date patterns.
        """
        # Simulate news sentiment based on date patterns
        # Weekends tend to have less news, weekdays more
        day_of_week = current_date.dayofweek

        # Base sentiment with weekly pattern
        base_sentiment = 0.1 * np.sin(2 * np.pi * current_date.dayofyear / 365)

        # Weekend adjustment (less news = neutral sentiment)
        if day_of_week >= 5:  # Saturday/Sunday
            base_sentiment *= 0.3

        # Monthly earnings season boost
        if current_date.day <= 7:  # First week of month
            base_sentiment += 0.2

        # Random noise to simulate news events
        np.random.seed(int(current_date.timestamp()) % 10000)
        noise = np.random.normal(0, 0.3)

        sentiment = base_sentiment + noise

        # Apply decay to older sentiment
        if hasattr(current_date, 'tz') and current_date.tz:
            now = pd.Timestamp.now(tz=current_date.tz)
        else:
            now = pd.Timestamp.now()
        days_old = (now - current_date).days
        decay_factor = np.exp(-days_old / self.sentiment_decay_days)

        return np.clip(sentiment * decay_factor, -1.0, 1.0)

    def _get_social_sentiment(self, current_date: pd.Timestamp) -> float:
        """
        Generate social media sentiment signal.

        Simulates sentiment from Twitter, Reddit, etc.
        """
        # Social sentiment often follows market hours and news
        hour = current_date.hour

        # Market hours boost (9:30 AM - 4:00 PM ET)
        if 9 <= hour <= 16:
            base_sentiment = 0.2
        else:
            base_sentiment = 0.0

        # Weekend reduction
        if current_date.dayofweek >= 5:
            base_sentiment *= 0.5

        # Trend following market sentiment
        trend_factor = 0.1 * np.sin(2 * np.pi * current_date.dayofyear / 30)  # Monthly cycle

        # Random social media buzz
        np.random.seed(int(current_date.timestamp() * 1.5) % 10000)
        social_noise = np.random.normal(0, 0.4)

        sentiment = base_sentiment + trend_factor + social_noise

        return np.clip(sentiment, -1.0, 1.0)

    def _get_search_sentiment(self, current_date: pd.Timestamp) -> float:
        """
        Generate search trends sentiment signal.

        Simulates Google Trends or similar search interest.
        """
        # Search trends often lead market sentiment
        day_of_year = current_date.dayofyear

        # Seasonal patterns
        seasonal_sentiment = 0.15 * np.sin(2 * np.pi * day_of_year / 365)

        # Economic calendar events (simplified)
        if current_date.day in [1, 15]:  # Paydays, mid-month
            seasonal_sentiment += 0.1

        # Weekend search patterns (different from weekday)
        if current_date.dayofweek >= 5:
            seasonal_sentiment *= 0.7

        # Random search interest spikes
        np.random.seed(int(current_date.timestamp() * 2) % 10000)
        search_noise = np.random.normal(0, 0.25)

        sentiment = seasonal_sentiment + search_noise

        return np.clip(sentiment, -1.0, 1.0)

    def _calculate_confidence(self,
                            news_sentiment: float,
                            social_sentiment: float,
                            search_sentiment: float) -> float:
        """
        Calculate confidence based on sentiment consistency and magnitude.
        """
        sentiments = [news_sentiment, social_sentiment, search_sentiment]

        # Consistency confidence (how well they agree)
        mean_sentiment = np.mean(sentiments)
        std_sentiment = np.std(sentiments)

        if std_sentiment > 0:
            consistency_conf = 1 / (1 + std_sentiment)  # Lower std = higher confidence
        else:
            consistency_conf = 1.0

        # Magnitude confidence (stronger signals = higher confidence)
        magnitude_conf = min(abs(mean_sentiment) * 2, 0.8)  # Up to 80% confidence

        # Combine confidences
        confidence = (consistency_conf + magnitude_conf) / 2

        return min(confidence, 1.0)

    def _adjust_for_regime(self,
                          signal: float,
                          confidence: float,
                          regime_context: Dict[str, Any]) -> tuple[float, float]:
        """
        Adjust signal and confidence based on market regime.
        """
        regime = regime_context.get('regime_tag', 'NORMAL')

        # Alternative data reliability varies by regime
        regime_multipliers = {
            'HIGH_VOL': 0.9,   # Alternative data can be noisy in high vol
            'LOW_VOL': 1.1,    # More reliable in stable markets
            'BULL_QUIET': 1.0,
            'BEAR_CRISIS': 0.8,  # News/social can be overly negative
            'NORMAL': 1.0
        }

        multiplier = regime_multipliers.get(regime, 1.0)
        adjusted_signal = signal * multiplier

        # Confidence adjustments
        if regime == 'LOW_VOL':
            adjusted_confidence = min(confidence * 1.1, 1.0)
        elif regime in ['HIGH_VOL', 'BEAR_CRISIS']:
            adjusted_confidence = confidence * 0.9
        else:
            adjusted_confidence = confidence

        return adjusted_signal, adjusted_confidence

    def _store_sentiment_data(self,
                             date: pd.Timestamp,
                             news_sentiment: float,
                             social_sentiment: float,
                             search_sentiment: float):
        """
        Store sentiment data for analysis and decay calculations.
        """
        sentiment_data = {
            'date': date,
            'news': news_sentiment,
            'social': social_sentiment,
            'search': search_sentiment,
            'combined': (news_sentiment + social_sentiment + search_sentiment) / 3
        }

        # Keep only recent data
        max_history = 30
        for source in self.sentiment_history:
            self.sentiment_history[source].append(sentiment_data)
            if len(self.sentiment_history[source]) > max_history:
                self.sentiment_history[source] = self.sentiment_history[source][-max_history:]

    def analyze_news_text(self, news_text: str) -> float:
        """
        Analyze actual news text for sentiment.

        Args:
            news_text: Raw news article text

        Returns:
            Sentiment score between -1 and 1
        """
        if not news_text:
            return 0.0

        text_lower = news_text.lower()

        # Count positive and negative keywords
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

        total_keywords = positive_count + negative_count

        if total_keywords == 0:
            return 0.0

        # Calculate sentiment
        sentiment = (positive_count - negative_count) / total_keywords

        return np.clip(sentiment, -1.0, 1.0)

    def get_sentiment_history(self, source: str = 'combined', days: int = 7) -> pd.DataFrame:
        """
        Get historical sentiment data.

        Args:
            source: 'news', 'social', 'search', or 'combined'
            days: Number of days of history

        Returns:
            DataFrame with sentiment history
        """
        if source not in self.sentiment_history:
            return pd.DataFrame()

        data = self.sentiment_history[source][-days:] if len(self.sentiment_history[source]) >= days else self.sentiment_history[source]

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        return df

    def get_sentiment_stats(self) -> Dict[str, Any]:
        """
        Get statistics about sentiment data.
        """
        stats = {}

        for source in self.sentiment_history:
            data = self.sentiment_history[source]
            if data:
                sentiments = [d.get(source, 0) for d in data]
                stats[source] = {
                    'count': len(sentiments),
                    'mean': np.mean(sentiments),
                    'std': np.std(sentiments),
                    'min': np.min(sentiments),
                    'max': np.max(sentiments)
                }
            else:
                stats[source] = {'count': 0}

        return stats
