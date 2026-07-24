import logging
import asyncio
import feedparser
import re
from typing import Dict, Any, List
import datetime

logger = logging.getLogger(__name__)

class SentimentEngine:
    """
    FinBERT Transformer NLP & RSS News Sentiment Engine.
    Utilizes HuggingFace FinBERT (`ProsusAI/finbert`) model pipeline for financial NLP classification,
    with an optimized keyword-lexicon fallback when offline.
    """
    
    BULLISH_TERMS = {
        "surge", "soar", "jump", "climb", "gain", "rally", "upbeat", 
        "beat", "beats", "growth", "buy", "upgrade", "outperform",
        "record", "profit", "bullish", "strong", "higher", "dividend", "revenue"
    }
    
    BEARISH_TERMS = {
        "plunge", "tumble", "fall", "drop", "decline", "slump", 
        "miss", "misses", "loss", "sell", "downgrade", "underperform",
        "warning", "bearish", "weak", "lower", "crash", "lawsuit", "investigation", "bankrupt"
    }

    def __init__(self, use_finbert: bool = True):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._CACHE_TTL = 3600  # 1 hour cache
        self.finbert_pipeline = None

        if use_finbert:
            self._init_finbert()

    def _init_finbert(self):
        """Initialize FinBERT Transformer pipeline asynchronously or on-demand."""
        try:
            from transformers import pipeline
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None
            )
            logger.info("FinBERT Transformer NLP Sentiment Pipeline initialized.")
        except Exception as e:
            logger.warning(f"FinBERT initialization skipped ({e}). Using financial keyword lexicon.")
            self.finbert_pipeline = None

    async def get_sentiment(self, symbol: str) -> float:
        """Fetch news for symbol and calculate sentiment score in [-1.0, 1.0]."""
        now = datetime.datetime.now().timestamp()
        
        if symbol in self._cache:
            entry = self._cache[symbol]
            if now - entry["timestamp"] < self._CACHE_TTL:
                return entry["score"]

        try:
            score = await asyncio.to_thread(self._fetch_and_score, symbol)
            self._cache[symbol] = {"timestamp": now, "score": score}
            return score
        except Exception as e:
            logger.debug(f"Sentiment fetch failed for {symbol}: {e}")
            return 0.0

    def _fetch_and_score(self, symbol: str) -> float:
        """Fetch headlines and run FinBERT or keyword classifier."""
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        
        if not feed.entries:
            return 0.0

        headlines = [entry.title for entry in feed.entries[:10]]
        
        if self.finbert_pipeline is not None:
            try:
                scores = []
                results = self.finbert_pipeline(headlines)
                for res in results:
                    # res is list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
                    res_dict = {item['label'].lower(): item['score'] for item in res}
                    pos = res_dict.get('positive', 0.0)
                    neg = res_dict.get('negative', 0.0)
                    scores.append(pos - neg)
                if scores:
                    return float(np.clip(np.mean(scores), -1.0, 1.0))
            except Exception as e:
                logger.debug(f"FinBERT inference failed: {e}. Falling back to lexicon.")

        # Fallback to keyword lexicon
        total_score = 0.0
        scored_count = 0
        for title in headlines:
            title_lower = title.lower()
            words = set(re.findall(r'\b\w+\b', title_lower))
            bull_matches = len(words.intersection(self.BULLISH_TERMS))
            bear_matches = len(words.intersection(self.BEARISH_TERMS))

            if bull_matches > bear_matches:
                total_score += 1.0
                scored_count += 1
            elif bear_matches > bull_matches:
                total_score -= 1.0
                scored_count += 1

        if scored_count == 0:
            return 0.0

        final_score = total_score / scored_count
        return float(final_score * 0.75)
