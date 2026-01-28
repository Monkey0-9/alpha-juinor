import logging
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import Counter, defaultdict
import json
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

# NLP imports
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from textblob import TextBlob
    import spacy
    NLP_AVAILABLE = True

    # Download required NLTK data
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

    try:
        nltk.data.find('punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

except ImportError:
    NLP_AVAILABLE = False
    logger.warning("NLP libraries not available. NLP features disabled.")

class SentimentLabel(Enum):
    STRONG_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    STRONG_POSITIVE = 2

class TopicCategory(Enum):
    EARNINGS = "earnings"
    MERGERS_ACQUISITIONS = "m&a"
    REGULATORY = "regulatory"
    ECONOMIC = "economic"
    TECHNICAL = "technical"
    MARKET_SENTIMENT = "market_sentiment"
    SECTOR_SPECIFIC = "sector_specific"
    GENERAL = "general"

@dataclass
class NewsArticle:
    """Represents a processed news article."""
    article_id: str
    title: str
    content: str
    source: str
    published_at: datetime
    url: Optional[str] = None
    symbols: List[str] = None
    sentiment_score: float = 0.0
    sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL
    topics: List[TopicCategory] = None
    entities: Dict[str, List[str]] = None
    summary: Optional[str] = None
    keywords: List[str] = None
    relevance_score: float = 0.0

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []
        if self.topics is None:
            self.topics = []
        if self.entities is None:
            self.entities = {}
        if self.keywords is None:
            self.keywords = []

@dataclass
class MarketImpact:
    """Represents the market impact of news."""
    direction: str  # 'positive', 'negative', 'neutral'
    magnitude: float  # 0-1 scale
    confidence: float
    time_horizon: str  # 'short', 'medium', 'long'
    affected_sectors: List[str]
    volatility_impact: float

class InstitutionalNLPEngine:
    """
    INSTITUTIONAL-GRADE NATURAL LANGUAGE PROCESSING ENGINE
    Advanced news analysis, sentiment extraction, topic modeling, and market impact assessment.
    """

    def __init__(self, model_dir: str = "models/nlp", cache_dir: str = "data/cache/nlp"):
        global NLP_AVAILABLE
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not NLP_AVAILABLE:
            logger.error("NLP libraries not available")
            return

        # Initialize NLP components
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except (OSError, MemoryError, Exception) as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            logger.warning("NLP features will be running in degraded mode (No NER).")
            self.nlp = None

            # If we encountered a fatal memory error, we might want to disable NLP entirely
            if isinstance(e, MemoryError) or "ArrayMemoryError" in str(e):
                logger.error("Critical Memory Error in NLP initialization. Disabling NLP Engine.")
                NLP_AVAILABLE = False
                return

        # Topic modeling
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            max_iter=10
        )

        # Custom sentiment lexicon for financial terms
        self.financial_lexicon = self._load_financial_lexicon()

        # Keyword patterns for different topics
        self.topic_patterns = self._initialize_topic_patterns()

        # Cache for processed articles
        self.processed_cache = {}

        logger.info("Institutional NLP Engine initialized")

    def process_news_article(self, article_data: Dict[str, Any]) -> NewsArticle:
        """
        Process a single news article with full NLP analysis.
        """
        if not NLP_AVAILABLE:
            return NewsArticle(
                article_id=article_data.get('id', 'unknown'),
                title=article_data.get('title', ''),
                content=article_data.get('content', ''),
                source=article_data.get('source', ''),
                published_at=datetime.fromisoformat(article_data.get('published_at', datetime.utcnow().isoformat()))
            )

        # Create article object
        article = NewsArticle(
            article_id=article_data.get('id', hashlib.md5(article_data.get('title', '').encode(), usedforsecurity=False).hexdigest()[:16]),
            title=article_data.get('title', ''),
            content=article_data.get('content', ''),
            source=article_data.get('source', ''),
            published_at=datetime.fromisoformat(article_data.get('published_at', datetime.utcnow().isoformat())),
            url=article_data.get('url'),
            symbols=article_data.get('symbols', [])
        )

        # Check cache
        cache_key = f"{article.article_id}_{article.published_at.isoformat()}"
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]

        # Process article
        full_text = f"{article.title} {article.content}"

        # Sentiment analysis
        article.sentiment_score, article.sentiment_label = self._analyze_sentiment(full_text)

        # Topic classification
        article.topics = self._classify_topics(full_text)

        # Named entity recognition
        article.entities = self._extract_entities(full_text)

        # Keyword extraction
        article.keywords = self._extract_keywords(full_text)

        # Generate summary
        article.summary = self._generate_summary(article.content)

        # Calculate relevance score
        article.relevance_score = self._calculate_relevance_score(article)

        # Cache result
        self.processed_cache[cache_key] = article

        return article

    def analyze_market_impact(self, articles: List[NewsArticle], symbol: str,
                            lookback_hours: int = 24) -> MarketImpact:
        """
        Analyze the collective market impact of recent news articles.
        """
        if not articles:
            return MarketImpact(
                direction='neutral',
                magnitude=0.0,
                confidence=0.0,
                time_horizon='short',
                affected_sectors=[],
                volatility_impact=0.0
            )

        # Filter articles by time and relevance
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        relevant_articles = [
            article for article in articles
            if article.published_at > cutoff_time and
            symbol in article.symbols and
            article.relevance_score > 0.3
        ]

        if not relevant_articles:
            return MarketImpact(
                direction='neutral',
                magnitude=0.0,
                confidence=0.5,
                time_horizon='short',
                affected_sectors=[],
                volatility_impact=0.0
            )

        # Aggregate sentiment
        sentiment_scores = [article.sentiment_score for article in relevant_articles]
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_std = np.std(sentiment_scores)

        # Determine direction and magnitude
        if abs(avg_sentiment) < 0.1:
            direction = 'neutral'
            magnitude = 0.0
        elif avg_sentiment > 0:
            direction = 'positive'
            magnitude = min(avg_sentiment, 1.0)
        else:
            direction = 'negative'
            magnitude = min(abs(avg_sentiment), 1.0)

        # Calculate confidence based on consensus and volume
        confidence = min(len(relevant_articles) / 10, 1.0) * (1 - sentiment_std)

        # Determine time horizon
        if any(topic in [TopicCategory.EARNINGS, TopicCategory.MERGERS_ACQUISITIONS]
               for article in relevant_articles for topic in article.topics):
            time_horizon = 'long'
        elif any(topic == TopicCategory.ECONOMIC for article in relevant_articles for topic in article.topics):
            time_horizon = 'medium'
        else:
            time_horizon = 'short'

        # Identify affected sectors
        affected_sectors = self._identify_affected_sectors(relevant_articles)

        # Estimate volatility impact
        volatility_impact = magnitude * (1 + len(relevant_articles) * 0.1)

        return MarketImpact(
            direction=direction,
            magnitude=magnitude,
            confidence=confidence,
            time_horizon=time_horizon,
            affected_sectors=affected_sectors,
            volatility_impact=volatility_impact
        )

    def extract_trading_signals(self, articles: List[NewsArticle], symbol: str) -> Dict[str, Any]:
        """
        Extract actionable trading signals from news analysis.
        """
        market_impact = self.analyze_market_impact(articles, symbol)

        signals = {
            'symbol': symbol,
            'signal_strength': market_impact.magnitude,
            'direction': market_impact.direction,
            'confidence': market_impact.confidence,
            'time_horizon': market_impact.time_horizon,
            'recommended_action': self._get_recommended_action(market_impact),
            'stop_loss_level': None,
            'take_profit_level': None,
            'risk_management': self._generate_risk_management_rules(market_impact),
            'supporting_evidence': self._get_supporting_evidence(articles, symbol)
        }

        # Set price levels based on signal
        if market_impact.direction == 'positive':
            signals['take_profit_level'] = 'current_price * 1.05'  # 5% upside
            signals['stop_loss_level'] = 'current_price * 0.98'    # 2% downside
        elif market_impact.direction == 'negative':
            signals['take_profit_level'] = 'current_price * 0.95'  # 5% downside
            signals['stop_loss_level'] = 'current_price * 1.02'    # 2% upside

        return signals

    def perform_topic_modeling(self, articles: List[NewsArticle], num_topics: int = 10) -> Dict[str, Any]:
        """
        Perform topic modeling on a corpus of news articles.
        """
        if not articles:
            return {}

        # Prepare text corpus
        corpus = [f"{article.title} {article.content}" for article in articles]

        # Vectorize
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

        # Fit LDA model
        lda_output = self.lda_model.fit_transform(tfidf_matrix)

        # Extract topics
        topics = {}
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topics[f"topic_{topic_idx}"] = {
                'words': top_words,
                'weight': float(topic.sum()),
                'articles': []
            }

        # Assign articles to topics
        for i, article in enumerate(articles):
            dominant_topic = lda_output[i].argmax()
            topics[f"topic_{dominant_topic}"]['articles'].append(article.article_id)

        return {
            'topics': topics,
            'topic_distribution': lda_output.tolist(),
            'num_articles': len(articles),
            'coherence_score': self._calculate_topic_coherence(topics)
        }

    def _analyze_sentiment(self, text: str) -> Tuple[float, SentimentLabel]:
        """Advanced sentiment analysis combining multiple methods."""
        if not text:
            return 0.0, SentimentLabel.NEUTRAL

        # VADER sentiment
        vader_scores = self.sia.polarity_scores(text)
        vader_compound = vader_scores['compound']

        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity

        # Financial lexicon adjustment
        financial_score = self._calculate_financial_sentiment(text)

        # Ensemble score
        ensemble_score = (vader_compound * 0.4 + textblob_polarity * 0.3 + financial_score * 0.3)

        # Normalize to [-1, 1]
        ensemble_score = np.clip(ensemble_score, -1, 1)

        # Classify sentiment
        if ensemble_score >= 0.3:
            label = SentimentLabel.STRONG_POSITIVE if ensemble_score >= 0.6 else SentimentLabel.POSITIVE
        elif ensemble_score <= -0.3:
            label = SentimentLabel.STRONG_NEGATIVE if ensemble_score <= -0.6 else SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        return ensemble_score, label

    def _calculate_financial_sentiment(self, text: str) -> float:
        """Calculate sentiment using financial-specific lexicon."""
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

        positive_score = 0
        negative_score = 0

        for word in words:
            if word in self.financial_lexicon['positive']:
                positive_score += self.financial_lexicon['positive'][word]
            elif word in self.financial_lexicon['negative']:
                negative_score += self.financial_lexicon['negative'][word]

        total_words = len(words)
        if total_words == 0:
            return 0

        # Normalize and return net sentiment
        return (positive_score - negative_score) / total_words

    def _classify_topics(self, text: str) -> List[TopicCategory]:
        """Classify article into topic categories."""
        topics = []
        text_lower = text.lower()

        for category, patterns in self.topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    topics.append(category)
                    break

        # Default to general if no specific topics found
        if not topics:
            topics.append(TopicCategory.GENERAL)

        return list(set(topics))  # Remove duplicates

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy."""
        if not self.nlp:
            return {}

        try:
            doc = self.nlp(text)
            entities = defaultdict(list)
            for ent in doc.ents:
                entities[ent.label_].append(ent.text)
            return dict(entities)
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            return {}

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords using TF-IDF and frequency analysis."""
        # Simple keyword extraction (in production, use more sophisticated methods)
        words = word_tokenize(text.lower())
        words = [word for word in words if word not in self.stop_words and len(word) > 3]

        # Get most common words
        word_freq = Counter(words)
        keywords = [word for word, _ in word_freq.most_common(max_keywords)]

        return keywords

    def _generate_summary(self, content: str, max_sentences: int = 3) -> str:
        """Generate extractive summary of article content."""
        if not content:
            return ""

        sentences = sent_tokenize(content)

        # Simple extractive summarization (take first few sentences)
        # In production, use more sophisticated methods like TextRank
        summary_sentences = sentences[:max_sentences]

        return ' '.join(summary_sentences)

    def _calculate_relevance_score(self, article: NewsArticle) -> float:
        """Calculate how relevant this article is for trading decisions."""
        score = 0.0

        # Sentiment strength
        score += abs(article.sentiment_score) * 0.3

        # Topic importance
        high_importance_topics = [TopicCategory.EARNINGS, TopicCategory.MERGERS_ACQUISITIONS,
                                TopicCategory.REGULATORY, TopicCategory.ECONOMIC]
        if any(topic in high_importance_topics for topic in article.topics):
            score += 0.4

        # Source credibility (simplified)
        credible_sources = ['reuters', 'bloomberg', 'wsj', 'ft', 'cnbc']
        if any(source.lower() in article.source.lower() for source in credible_sources):
            score += 0.2

        # Recency (newer articles more relevant)
        hours_old = (datetime.utcnow() - article.published_at).total_seconds() / 3600
        recency_score = max(0, 1 - (hours_old / 24))  # Decay over 24 hours
        score += recency_score * 0.1

        return min(score, 1.0)

    def _get_recommended_action(self, impact: MarketImpact) -> str:
        """Get recommended trading action based on market impact."""
        if impact.direction == 'positive' and impact.magnitude > 0.5:
            return 'BUY_STRONG'
        elif impact.direction == 'positive':
            return 'BUY'
        elif impact.direction == 'negative' and impact.magnitude > 0.5:
            return 'SELL_STRONG'
        elif impact.direction == 'negative':
            return 'SELL'
        else:
            return 'HOLD'

    def _generate_risk_management_rules(self, impact: MarketImpact) -> Dict[str, Any]:
        """Generate risk management rules based on market impact."""
        rules = {
            'position_size_limit': 0.05,  # 5% of portfolio
            'stop_loss_pct': 0.02,        # 2% stop loss
            'take_profit_pct': 0.05,      # 5% take profit
            'max_holding_period': '1_week'
        }

        # Adjust based on impact
        if impact.time_horizon == 'long':
            rules['max_holding_period'] = '1_month'
            rules['take_profit_pct'] = 0.10  # Higher target for longer horizon

        if impact.volatility_impact > 0.7:
            rules['position_size_limit'] = 0.02  # Smaller position for high volatility
            rules['stop_loss_pct'] = 0.03       # Wider stop loss

        return rules

    def _get_supporting_evidence(self, articles: List[NewsArticle], symbol: str) -> List[Dict[str, Any]]:
        """Get supporting evidence for the trading signal."""
        evidence = []

        for article in articles:
            if symbol in article.symbols and article.relevance_score > 0.3:
                evidence.append({
                    'article_id': article.article_id,
                    'title': article.title,
                    'sentiment': article.sentiment_label.value,
                    'topics': [topic.value for topic in article.topics],
                    'source': article.source,
                    'published_at': article.published_at.isoformat(),
                    'relevance_score': article.relevance_score
                })

        # Sort by relevance
        evidence.sort(key=lambda x: x['relevance_score'], reverse=True)

        return evidence[:5]  # Top 5 articles

    def _identify_affected_sectors(self, articles: List[NewsArticle]) -> List[str]:
        """Identify sectors affected by the news."""
        sectors = set()

        sector_keywords = {
            'technology': ['tech', 'software', 'semiconductor', 'ai', 'cloud'],
            'healthcare': ['pharma', 'biotech', 'healthcare', 'fda', 'clinical'],
            'finance': ['bank', 'financial', 'fed', 'treasury', 'interest rate'],
            'energy': ['oil', 'gas', 'energy', 'renewable', 'electric'],
            'consumer': ['retail', 'consumer', 'automotive', 'ecommerce']
        }

        for article in articles:
            text = f"{article.title} {article.content}".lower()
            for sector, keywords in sector_keywords.items():
                if any(keyword in text for keyword in keywords):
                    sectors.add(sector)

        return list(sectors)

    def _calculate_topic_coherence(self, topics: Dict[str, Any]) -> float:
        """Calculate topic coherence score."""
        # Simplified coherence calculation
        total_coherence = 0
        num_topics = len(topics)

        for topic_data in topics.values():
            words = topic_data['words'][:5]  # Top 5 words
            # Simple pairwise coherence (in production, use more sophisticated metrics)
            coherence = len(set(words)) / len(words)  # Uniqueness score
            total_coherence += coherence

        return total_coherence / num_topics if num_topics > 0 else 0

    def _load_financial_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Load financial sentiment lexicon."""
        # Simplified financial lexicon (in production, use comprehensive lexicon)
        return {
            'positive': {
                'upgrade': 0.8, 'beat': 0.7, 'surprise': 0.6, 'strong': 0.5,
                'growth': 0.6, 'profit': 0.7, 'revenue': 0.5, 'earnings': 0.6,
                'bullish': 0.8, 'optimistic': 0.5, 'positive': 0.4
            },
            'negative': {
                'downgrade': -0.8, 'miss': -0.7, 'disappoint': -0.6, 'weak': -0.5,
                'decline': -0.6, 'loss': -0.7, 'debt': -0.5, 'lawsuit': -0.8,
                'bearish': -0.8, 'pessimistic': -0.5, 'negative': -0.4
            }
        }

    def _initialize_topic_patterns(self) -> Dict[TopicCategory, List[str]]:
        """Initialize regex patterns for topic classification."""
        return {
            TopicCategory.EARNINGS: [
                r'earnings|revenue|profit|eps|quarterly|annual.*report',
                r'beat.*estimate|miss.*estimate|guidance'
            ],
            TopicCategory.MERGERS_ACQUISITIONS: [
                r'mergers?|acquisitions?|buyout|takeover|bid',
                r'acquire.*company|merge.*with'
            ],
            TopicCategory.REGULATORY: [
                r'regulator|sec|fda|compliance|lawsuit|legal',
                r'investigation|fine|penalty|regulation'
            ],
            TopicCategory.ECONOMIC: [
                r'economic|fed|interest.*rate|inflation|gdp',
                r'unemployment|recession|growth|economy'
            ],
            TopicCategory.TECHNICAL: [
                r'technical.*analysis|chart|pattern|resistance|support',
                r'moving.*average|rsi|macd|bollinger'
            ],
            TopicCategory.MARKET_SENTIMENT: [
                r'market.*sentiment|mood|confidence|fear|greed',
                r'investor.*confidence|bullish|bearish'
            ],
            TopicCategory.SECTOR_SPECIFIC: [
                r'sector|industry|tech|healthcare|finance|energy',
                r'automotive|retail|consumer|industrial'
            ]
        }
