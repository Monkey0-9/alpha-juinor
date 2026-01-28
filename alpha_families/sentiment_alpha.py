"""
Sentiment Alpha Model - NLP-based sentiment analysis for trading signals.

Uses natural language processing to analyze news articles, social media,
earnings calls, and other text data to generate sentiment-based alpha signals.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

from .base_alpha import BaseAlpha

logger = logging.getLogger(__name__)

class SentimentAlpha(BaseAlpha):
    """
    NLP-based Sentiment Alpha Model.

    Analyzes text data from various sources to generate sentiment signals:
    - News articles
    - Social media posts
    - Earnings call transcripts
    - SEC filings
    """

    def __init__(self,
                 news_weight: float = 0.5,
                 social_weight: float = 0.3,
                 earnings_weight: float = 0.2,
                 sentiment_decay_hours: int = 24,
                 min_confidence_threshold: float = 0.6):
        """
        Initialize Sentiment Alpha.

        Args:
            news_weight: Weight for news sentiment
            social_weight: Weight for social media sentiment
            earnings_weight: Weight for earnings sentiment
            sentiment_decay_hours: Hours over which sentiment decays
            min_confidence_threshold: Minimum confidence for signal generation
        """
        super().__init__()

        self.news_weight = news_weight
        self.social_weight = social_weight
        self.earnings_weight = earnings_weight
        self.sentiment_decay_hours = sentiment_decay_hours
        self.min_confidence_threshold = min_confidence_threshold

        # Sentiment tracking
        self.sentiment_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # Sentiment lexicons
        self._initialize_sentiment_lexicons()

    def _initialize_sentiment_lexicons(self):
        """Initialize sentiment analysis lexicons."""
        # Positive sentiment words
        self.positive_words = {
            'bullish', 'strong', 'gains', 'rally', 'surge', 'boost', 'upgrade',
            'positive', 'optimistic', 'growth', 'beat', 'record', 'high',
            'excellent', 'outstanding', 'remarkable', 'impressive', 'robust',
            'thriving', 'prosperous', 'flourishing', 'successful', 'profitable'
        }

        # Negative sentiment words
        self.negative_words = {
            'bearish', 'weak', 'decline', 'drop', 'fall', 'sell', 'downgrade',
            'negative', 'pessimistic', 'recession', 'miss', 'crash', 'plunge',
            'terrible', 'awful', 'dismal', 'disappointing', 'concerning', 'worrisome',
            'troubling', 'disturbing', 'alarming', 'crisis', 'disaster'
        }

        # Intensifiers
        self.intensifiers = {
            'very', 'extremely', 'highly', 'particularly', 'especially',
            'incredibly', 'remarkably', 'exceptionally', 'significantly'
        }

        # Negation words
        self.negations = {
            'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor',
            'cannot', 'cant', 'wont', 'dont', 'doesnt', 'didnt'
        }

    def generate_signal(self,
                       market_data: pd.DataFrame,
                       regime_context: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate sentiment-based alpha signal.

        Args:
            market_data: OHLCV market data
            regime_context: Current market regime

        Returns:
            Signal dictionary with signal, confidence, and metadata
        """
        try:
            current_time = pd.Timestamp.now()

            # In a real implementation, this would fetch actual text data
            # For now, simulate sentiment signals
            news_sentiment = self._get_news_sentiment(current_time)
            social_sentiment = self._get_social_sentiment(current_time)
            earnings_sentiment = self._get_earnings_sentiment(current_time)

            # Combine sentiments with weights
            combined_sentiment = (
                news_sentiment * self.news_weight +
                social_sentiment * self.social_weight +
                earnings_sentiment * self.earnings_weight
            )

            # Normalize to [-1, 1] range
            signal = np.clip(combined_sentiment, -1.0, 1.0)

            # Calculate confidence
            confidence = self._calculate_confidence(
                news_sentiment, social_sentiment, earnings_sentiment
            )

            # Only generate signal if confidence is above threshold
            if confidence < self.min_confidence_threshold:
                signal = 0.0

            # Adjust for market regime
            if regime_context:
                signal, confidence = self._adjust_for_regime(signal, confidence, regime_context)

            # Store sentiment data
            self._store_sentiment_data(
                current_time, news_sentiment, social_sentiment, earnings_sentiment, signal
            )

            return {
                'signal': float(signal),
                'confidence': float(confidence),
                'metadata': {
                    'news_sentiment': float(news_sentiment),
                    'social_sentiment': float(social_sentiment),
                    'earnings_sentiment': float(earnings_sentiment),
                    'combined_sentiment': float(combined_sentiment),
                    'regime_adjusted': regime_context is not None,
                    'sentiment_sources': 3,
                    'confidence_threshold_met': confidence >= self.min_confidence_threshold
                }
            }

        except Exception as e:
            logger.error(f"Sentiment alpha signal generation failed: {e}")
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }

    def _get_news_sentiment(self, current_time: pd.Timestamp) -> float:
        """
        Generate news sentiment signal.

        In production, this would analyze actual news feeds.
        """
        # Simulate news sentiment based on time patterns
        hour = current_time.hour
        day_of_week = current_time.dayofweek

        # Market hours have more news
        if 9 <= hour <= 16 and day_of_week < 5:  # Weekday market hours
            base_sentiment = 0.1
        else:
            base_sentiment = 0.0

        # Economic data releases (simplified)
        if hour in [8, 10, 14]:  # Common economic data times
            base_sentiment += 0.2

        # Random news events
        np.random.seed(int(current_time.timestamp()) % 10000)
        news_noise = np.random.normal(0, 0.3)

        sentiment = base_sentiment + news_noise

        # Apply time decay
        if hasattr(current_time, 'tz') and current_time.tz:
            now = pd.Timestamp.now(tz=current_time.tz)
        else:
            now = pd.Timestamp.now()
        hours_old = (now - current_time).total_seconds() / 3600
        decay_factor = np.exp(-hours_old / self.sentiment_decay_hours)

        return np.clip(sentiment * decay_factor, -1.0, 1.0)

    def _get_social_sentiment(self, current_time: pd.Timestamp) -> float:
        """
        Generate social media sentiment signal.
        """
        # Social sentiment follows market activity
        hour = current_time.hour

        # Peak social activity during market hours
        if 9 <= hour <= 16:
            base_sentiment = 0.15
        elif 16 <= hour <= 20:  # After-hours trading discussion
            base_sentiment = 0.1
        else:
            base_sentiment = 0.0

        # Weekend reduction
        if current_time.dayofweek >= 5:
            base_sentiment *= 0.6

        # Random social sentiment
        np.random.seed(int(current_time.timestamp() * 1.3) % 10000)
        social_noise = np.random.normal(0, 0.4)

        sentiment = base_sentiment + social_noise

        return np.clip(sentiment, -1.0, 1.0)

    def _get_earnings_sentiment(self, current_time: pd.Timestamp) -> float:
        """
        Generate earnings sentiment signal.
        """
        # Earnings season patterns
        month = current_time.month
        day = current_time.day

        # Earnings seasons (Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec)
        if month in [1, 2, 4, 5, 7, 8, 10, 11]:
            base_sentiment = 0.1
        else:
            base_sentiment = 0.0

        # Earnings release days (often mid-week)
        if current_time.dayofweek in [1, 2, 3]:  # Tue-Thu
            base_sentiment += 0.1

        # Random earnings surprises
        np.random.seed(int(current_time.timestamp() * 1.7) % 10000)
        earnings_noise = np.random.normal(0, 0.25)

        sentiment = base_sentiment + earnings_noise

        return np.clip(sentiment, -1.0, 1.0)

    def _calculate_confidence(self,
                            news_sentiment: float,
                            social_sentiment: float,
                            earnings_sentiment: float) -> float:
        """
        Calculate confidence based on sentiment consistency and strength.
        """
        sentiments = [news_sentiment, social_sentiment, earnings_sentiment]

        # Agreement confidence (how well sentiments agree)
        mean_sentiment = np.mean(sentiments)
        std_sentiment = np.std(sentiments)

        if std_sentiment > 0:
            agreement_conf = 1 / (1 + std_sentiment)
        else:
            agreement_conf = 1.0

        # Strength confidence (magnitude of average sentiment)
        strength_conf = min(abs(mean_sentiment) * 2, 0.7)

        # Recency confidence (how fresh the data is)
        recency_conf = 0.8  # Assume data is reasonably fresh

        confidence = (agreement_conf + strength_conf + recency_conf) / 3

        return min(confidence, 1.0)

    def _adjust_for_regime(self,
                          signal: float,
                          confidence: float,
                          regime_context: Dict[str, Any]) -> Tuple[float, float]:
        """
        Adjust signal and confidence based on market regime.
        """
        regime = regime_context.get('regime_tag', 'NORMAL')

        # Sentiment reliability varies by regime
        regime_multipliers = {
            'HIGH_VOL': 0.8,   # Sentiment can be noisy in high vol
            'LOW_VOL': 1.2,    # More reliable in stable markets
            'BULL_QUIET': 1.1, # Positive sentiment more reliable in bull markets
            'BEAR_CRISIS': 0.7, # Negative sentiment can be exaggerated
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
                             timestamp: pd.Timestamp,
                             news_sentiment: float,
                             social_sentiment: float,
                             earnings_sentiment: float,
                             final_signal: float):
        """
        Store sentiment data for analysis.
        """
        data_point = {
            'timestamp': timestamp,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'earnings_sentiment': earnings_sentiment,
            'combined_sentiment': (news_sentiment + social_sentiment + earnings_sentiment) / 3,
            'final_signal': final_signal
        }

        self.sentiment_history.append(data_point)

        # Maintain max history size
        if len(self.sentiment_history) > self.max_history_size:
            self.sentiment_history = self.sentiment_history[-self.max_history_size:]

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of input text.

        Args:
            text: Text to analyze

        Returns:
            Dict with sentiment score and confidence
        """
        if not text or not isinstance(text, str):
            return {'sentiment': 0.0, 'confidence': 0.0}

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Count sentiment words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return {'sentiment': 0.0, 'confidence': 0.0}

        # Calculate base sentiment
        sentiment_score = (positive_count - negative_count) / total_sentiment_words

        # Check for intensifiers and negations (simplified)
        intensifier_count = sum(1 for word in words if word in self.intensifiers)
        negation_count = sum(1 for word in words if word in self.negations)

        # Adjust for intensifiers
        if intensifier_count > 0:
            sentiment_score *= 1.5

        # Adjust for negations (simplified - just reduce confidence)
        if negation_count > 0:
            sentiment_score *= 0.7

        # Calculate confidence based on word count and sentiment strength
        total_words = len(words)
        word_coverage = total_sentiment_words / max(total_words, 1)

        confidence = min(word_coverage * abs(sentiment_score) * 2, 1.0)

        return {
            'sentiment': np.clip(sentiment_score, -1.0, 1.0),
            'confidence': confidence
        }

    def get_sentiment_history(self, hours: int = 24) -> pd.DataFrame:
        """
        Get recent sentiment history.

        Args:
            hours: Hours of history to retrieve

        Returns:
            DataFrame with sentiment history
        """
        if not self.sentiment_history:
            return pd.DataFrame()

        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        recent_data = [
            d for d in self.sentiment_history
            if d['timestamp'] >= cutoff_time
        ]

        if not recent_data:
            return pd.DataFrame()

        df = pd.DataFrame(recent_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        return df

    def get_sentiment_stats(self) -> Dict[str, Any]:
        """
        Get statistics about sentiment data.
        """
        if not self.sentiment_history:
            return {'total_samples': 0}

        df = pd.DataFrame(self.sentiment_history)

        stats = {
            'total_samples': len(df),
            'avg_news_sentiment': df['news_sentiment'].mean(),
            'avg_social_sentiment': df['social_sentiment'].mean(),
            'avg_earnings_sentiment': df['earnings_sentiment'].mean(),
            'avg_combined_sentiment': df['combined_sentiment'].mean(),
            'sentiment_volatility': df['combined_sentiment'].std(),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if len(df) > 0 else None,
                'end': df['timestamp'].max().isoformat() if len(df) > 0 else None
            }
        }

        return stats
