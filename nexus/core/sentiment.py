import logging
import asyncio
import feedparser
import re
from typing import Dict, Any, List
import datetime

logger = logging.getLogger(__name__)

class SentimentEngine:
    """
    RSS-based News Sentiment Engine.
    Uses free RSS feeds (Yahoo Finance) to gather headlines
    and applies basic NLP keyword scoring for bullish/bearish bias.
    """
    
    # Simple dictionary for financial sentiment
    BULLISH_TERMS = {
        "surge", "soar", "jump", "climb", "gain", "rally", "upbeat", 
        "beat", "beats", "growth", "buy", "upgrade", "outperform",
        "record", "profit", "bullish", "strong", "higher"
    }
    
    BEARISH_TERMS = {
        "plunge", "tumble", "fall", "drop", "decline", "slump", 
        "miss", "misses", "loss", "sell", "downgrade", "underperform",
        "warning", "bearish", "weak", "lower", "crash", "lawsuit", "investigation"
    }

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._CACHE_TTL = 3600  # 1 hour cache for news

    async def get_sentiment(self, symbol: str) -> float:
        """
        Fetch news for symbol and calculate sentiment score [-1.0, 1.0].
        Returns 0.0 if no news or neutral.
        """
        now = datetime.datetime.now().timestamp()
        
        # Check cache
        if symbol in self._cache:
            entry = self._cache[symbol]
            if now - entry["timestamp"] < self._CACHE_TTL:
                return entry["score"]

        try:
            # Use asyncio to run the blocking feedparser call in a thread
            score = await asyncio.to_thread(self._fetch_and_score, symbol)
            
            self._cache[symbol] = {
                "timestamp": now,
                "score": score
            }
            return score
            
        except Exception as e:
            logger.debug(f"Sentiment fetch failed for {symbol}: {e}")
            return 0.0

    def _fetch_and_score(self, symbol: str) -> float:
        """Blocking call to fetch RSS and score headlines."""
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        
        if not feed.entries:
            return 0.0
            
        total_score = 0.0
        scored_headlines = 0
        
        for entry in feed.entries[:10]: # Look at top 10 headlines
            title = entry.title.lower()
            
            # Simple word extraction
            words = set(re.findall(r'\b\w+\b', title))
            
            bull_matches = len(words.intersection(self.BULLISH_TERMS))
            bear_matches = len(words.intersection(self.BEARISH_TERMS))
            
            if bull_matches > bear_matches:
                total_score += 1.0
                scored_headlines += 1
            elif bear_matches > bull_matches:
                total_score -= 1.0
                scored_headlines += 1
                
        if scored_headlines == 0:
            return 0.0
            
        # Normalize to [-1.0, 1.0]
        final_score = total_score / scored_headlines
        
        # Dampen score slightly to avoid overreaction
        return final_score * 0.75
