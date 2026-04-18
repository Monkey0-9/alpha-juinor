"""
=============================================================================
NEXUS INSTITUTIONAL - LIVE NEWS & MARKET MONITORING SYSTEM
=============================================================================
Real-time monitoring for paper trading: news, market events, sentiment analysis
Designed to operate like: Jane Street, Citadel, Virtu, Jump Trading

Features:
- Real-time news monitoring (Bloomberg, Reuters, financial APIs)
- Market data streaming (prices, volumes, volatility)
- Sentiment analysis (bullish/bearish signals)
- Event-driven trading triggers
- Live portfolio monitoring
- Real-time risk management
- Performance tracking
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import deque

# External libraries (install: pip install feedparser requests textblob alpha-vantage)
try:
    import feedparser
    import requests
    from textblob import TextBlob
except ImportError:
    print("⚠️  Optional dependencies not installed. Install with:")
    print("   pip install feedparser requests textblob alpha-vantage")

logger = logging.getLogger("LiveMonitor")


class SentimentScore(Enum):
    """Market sentiment classification."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class NewsSource(Enum):
    """News sources for market monitoring."""
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    CNBC = "cnbc"
    MARKETWATCH = "marketwatch"
    SEEKING_ALPHA = "seekingalpha"
    CRYPTO_NEWS = "crypto"
    FX_NEWS = "forex"


@dataclass
class NewsArticle:
    """Represents a news article with market impact."""
    title: str
    source: NewsSource
    timestamp: datetime
    url: str
    content: str
    sentiment: SentimentScore = SentimentScore.NEUTRAL
    confidence: float = 0.0  # 0.0-1.0
    relevant_symbols: List[str] = field(default_factory=list)
    impact_level: str = "low"  # low, medium, high, critical
    
    def to_dict(self):
        return {
            "title": self.title,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "sentiment": self.sentiment.name,
            "confidence": self.confidence,
            "symbols": self.relevant_symbols,
            "impact": self.impact_level
        }


@dataclass
class MarketEvent:
    """Represents a market event (earnings, dividends, splits, etc)."""
    symbol: str
    event_type: str  # earnings, dividend, split, acquisition, regulatory, etc
    timestamp: datetime
    details: Dict = field(default_factory=dict)
    expected_volatility: float = 0.0  # Expected price move %
    
    def to_dict(self):
        return {
            "symbol": self.symbol,
            "type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "expected_vol": self.expected_volatility
        }


@dataclass
class TradingAlert:
    """Alert for trading opportunity or risk."""
    alert_type: str  # opportunity, risk, execution, info
    symbol: str
    trigger: str  # news, price, sentiment, volume, correlation, etc
    action: str  # buy, sell, hold, reduce, increase, hedge
    confidence: float  # 0.0-1.0
    suggested_position_size: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            "type": self.alert_type,
            "symbol": self.symbol,
            "trigger": self.trigger,
            "action": self.action,
            "confidence": self.confidence,
            "size": self.suggested_position_size,
            "timestamp": self.timestamp.isoformat()
        }


class NewsMonitor:
    """Real-time news monitoring from financial sources."""
    
    def __init__(self, refresh_interval: int = 60):
        """
        Initialize news monitor.
        
        Args:
            refresh_interval: Seconds between news fetches
        """
        self.refresh_interval = refresh_interval
        self.news_history = deque(maxlen=1000)  # Keep last 1000 articles
        self.symbol_watchlist = {
            'SPY', 'QQQ', 'IWM', 'GLD', 'TLT',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA',  # Tech
            'JPM', 'BAC', 'WFC', 'GS', 'BLK',  # Finance
            'XLF', 'XLE', 'XLV', 'XLI', 'XLK'  # Sectors
        }
        self.running = False
        
    async def fetch_rss_news(self) -> List[NewsArticle]:
        """Fetch news from RSS feeds."""
        articles = []
        
        # Financial RSS feeds
        feeds = {
            NewsSource.CNBC: "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            NewsSource.MARKETWATCH: "https://feeds.marketwatch.com/marketwatch/topstories/",
            NewsSource.REUTERS: "https://feeds.reuters.com/finance/markets",
        }
        
        for source, url in feeds.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:  # Last 5 entries per feed
                    article = NewsArticle(
                        title=entry.get('title', ''),
                        source=source,
                        timestamp=datetime.now(),
                        url=entry.get('link', ''),
                        content=entry.get('summary', '')[:500]
                    )
                    
                    # Analyze sentiment
                    article.sentiment = self._analyze_sentiment(article.content)
                    
                    # Extract relevant symbols
                    article.relevant_symbols = self._extract_symbols(article.title + " " + article.content)
                    
                    # Determine impact level
                    article.impact_level = self._assess_impact(article.title, article.relevant_symbols)
                    
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Error fetching from {source.value}: {e}")
        
        return articles
    
    def _analyze_sentiment(self, text: str) -> SentimentScore:
        """Analyze sentiment of text using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            if polarity < -0.5:
                return SentimentScore.VERY_NEGATIVE
            elif polarity < -0.1:
                return SentimentScore.NEGATIVE
            elif polarity < 0.1:
                return SentimentScore.NEUTRAL
            elif polarity < 0.5:
                return SentimentScore.POSITIVE
            else:
                return SentimentScore.VERY_POSITIVE
        except:
            return SentimentScore.NEUTRAL
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        # Simple extraction (in production, use NLP)
        symbols_in_text = []
        for symbol in self.symbol_watchlist:
            if symbol.upper() in text.upper():
                symbols_in_text.append(symbol)
        return symbols_in_text
    
    def _assess_impact(self, title: str, symbols: List[str]) -> str:
        """Assess impact level of news."""
        keywords_critical = ['bankrupt', 'crash', 'halt', 'investigation', 'fraud', 'recall']
        keywords_high = ['earnings', 'acquisition', 'lawsuit', 'ceo', 'sec']
        keywords_medium = ['rise', 'fall', 'gain', 'loss', 'growth', 'decline']
        
        title_lower = title.lower()
        
        for keyword in keywords_critical:
            if keyword in title_lower:
                return "critical"
        
        for keyword in keywords_high:
            if keyword in title_lower:
                return "high"
        
        for keyword in keywords_medium:
            if keyword in title_lower:
                return "medium"
        
        return "low"


class MarketDataMonitor:
    """Real-time market data monitoring."""
    
    def __init__(self):
        """Initialize market data monitor."""
        self.current_prices = {}
        self.price_history = {}
        self.volumes = {}
        self.volatility_estimates = {}
        self.correlation_matrix = {}
        
    async def fetch_market_data(self, symbols: List[str]) -> Dict:
        """
        Fetch current market data for symbols.
        In production, you'd use: yfinance, alpha_vantage, IB API, etc.
        """
        market_data = {}
        
        for symbol in symbols:
            try:
                # Placeholder - in production use real API
                market_data[symbol] = {
                    'price': self._get_simulated_price(symbol),
                    'volume': self._get_simulated_volume(symbol),
                    'bid_ask_spread': self._get_simulated_spread(symbol),
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {e}")
        
        return market_data
    
    def _get_simulated_price(self, symbol: str) -> float:
        """Simulate market price (replace with real API)."""
        # Placeholder prices
        base_prices = {
            'SPY': 500, 'QQQ': 380, 'IWM': 190, 'GLD': 200, 'TLT': 95,
            'AAPL': 180, 'MSFT': 420, 'GOOGL': 140, 'NVDA': 875, 'TSLA': 250
        }
        base = base_prices.get(symbol, 100)
        # Add random walk
        import random
        return max(1, base + random.gauss(0, base * 0.02))
    
    def _get_simulated_volume(self, symbol: str) -> float:
        """Simulate market volume."""
        import random
        return random.gauss(1000000, 100000)
    
    def _get_simulated_spread(self, symbol: str) -> float:
        """Simulate bid-ask spread in bps."""
        import random
        return random.gauss(5, 2)


class SentimentAggregator:
    """Aggregate market sentiment from multiple sources."""
    
    def __init__(self):
        """Initialize sentiment aggregator."""
        self.sentiment_scores = {}
        self.sentiment_history = deque(maxlen=100)
        
    def aggregate_sentiment(self, articles: List[NewsArticle], market_data: Dict) -> Dict:
        """Aggregate sentiment across all sources."""
        by_symbol = {}
        
        for article in articles:
            for symbol in article.relevant_symbols:
                if symbol not in by_symbol:
                    by_symbol[symbol] = {
                        'sentiment_scores': [],
                        'articles': [],
                        'impact_levels': []
                    }
                
                by_symbol[symbol]['sentiment_scores'].append(article.sentiment.value)
                by_symbol[symbol]['articles'].append(article)
                by_symbol[symbol]['impact_levels'].append(article.impact_level)
        
        # Calculate aggregate sentiment
        aggregated = {}
        for symbol, data in by_symbol.items():
            avg_sentiment = sum(data['sentiment_scores']) / len(data['sentiment_scores']) if data['sentiment_scores'] else 0
            critical_articles = [a for a in data['articles'] if a.impact_level == 'critical']
            
            aggregated[symbol] = {
                'sentiment_score': avg_sentiment,
                'sentiment_direction': 'bullish' if avg_sentiment > 0.3 else 'bearish' if avg_sentiment < -0.3 else 'neutral',
                'article_count': len(data['articles']),
                'critical_articles': len(critical_articles),
                'last_article': data['articles'][-1].timestamp.isoformat() if data['articles'] else None
            }
        
        return aggregated


class EventDrivenExecutor:
    """Generate trading signals based on events."""
    
    def __init__(self):
        """Initialize event-driven executor."""
        self.alerts = deque(maxlen=500)
        
    def generate_trading_signals(
        self,
        articles: List[NewsArticle],
        market_data: Dict,
        sentiment: Dict,
        portfolio: Dict
    ) -> List[TradingAlert]:
        """Generate trading signals from market events."""
        alerts = []
        
        for article in articles:
            for symbol in article.relevant_symbols:
                # Map sentiment to action
                if article.sentiment == SentimentScore.VERY_NEGATIVE and article.confidence > 0.7:
                    alerts.append(TradingAlert(
                        alert_type="risk",
                        symbol=symbol,
                        trigger="negative_news",
                        action="reduce",
                        confidence=article.confidence,
                        suggested_position_size=-0.5  # Reduce position by 50%
                    ))
                
                elif article.sentiment == SentimentScore.VERY_POSITIVE and article.confidence > 0.7:
                    alerts.append(TradingAlert(
                        alert_type="opportunity",
                        symbol=symbol,
                        trigger="positive_news",
                        action="increase",
                        confidence=article.confidence,
                        suggested_position_size=0.25  # Increase by 25%
                    ))
                
                # Check for critical corporate events
                if article.impact_level == "critical":
                    alerts.append(TradingAlert(
                        alert_type="execution",
                        symbol=symbol,
                        trigger="critical_event",
                        action="hedge",
                        confidence=0.9,
                        suggested_position_size=-1.0  # Full hedge
                    ))
        
        return alerts


class LiveTradingMonitor:
    """Main orchestrator for live paper trading monitoring."""
    
    def __init__(self, update_interval: int = 60):
        """
        Initialize live trading monitor.
        
        Args:
            update_interval: Seconds between updates (default 60)
        """
        self.update_interval = update_interval
        self.news_monitor = NewsMonitor(refresh_interval=update_interval)
        self.market_monitor = MarketDataMonitor()
        self.sentiment_aggregator = SentimentAggregator()
        self.event_executor = EventDrivenExecutor()
        
        self.portfolio = {
            'cash': 1000000,
            'positions': {},
            'value': 1000000
        }
        
        self.trades_executed = []
        self.alerts_generated = []
        self.running = False
        self.start_time = datetime.now()
        
    async def run_monitoring_loop(self, duration_seconds: Optional[int] = None):
        """Run continuous monitoring loop."""
        logger.info("="*80)
        logger.info("STARTING LIVE TRADING MONITOR - PAPER TRADING WITH NEWS/EVENT MONITORING")
        logger.info("="*80)
        logger.info(f"Update Interval: {self.update_interval}s")
        logger.info(f"Start Time: {self.start_time}")
        logger.info(f"Portfolio Value: ${self.portfolio['value']:,.2f}")
        logger.info("")
        
        self.running = True
        end_time = datetime.now() + timedelta(seconds=duration_seconds) if duration_seconds else None
        update_count = 0
        
        try:
            while self.running:
                update_count += 1
                current_time = datetime.now()
                
                # Check if duration exceeded
                if end_time and current_time > end_time:
                    logger.info(f"Duration limit reached after {update_count} updates")
                    break
                
                logger.info(f"\n{'='*80}")
                logger.info(f"UPDATE #{update_count} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*80}")
                
                # 1. Fetch news
                logger.info("📰 Fetching news...")
                articles = await self.news_monitor.fetch_rss_news()
                logger.info(f"   Found {len(articles)} articles")
                
                # 2. Fetch market data
                logger.info("📊 Fetching market data...")
                symbols = list(self.news_monitor.symbol_watchlist)
                market_data = await self.market_monitor.fetch_market_data(symbols)
                logger.info(f"   Retrieved data for {len(market_data)} symbols")
                
                # 3. Aggregate sentiment
                logger.info("🎯 Analyzing sentiment...")
                sentiment = self.sentiment_aggregator.aggregate_sentiment(articles, market_data)
                
                # Show top sentiment shifts
                bullish = [s for s, v in sentiment.items() if v['sentiment_direction'] == 'bullish']
                bearish = [s for s, v in sentiment.items() if v['sentiment_direction'] == 'bearish']
                
                if bullish:
                    logger.info(f"   📈 Bullish: {', '.join(bullish[:5])}")
                if bearish:
                    logger.info(f"   📉 Bearish: {', '.join(bearish[:5])}")
                
                # 4. Generate trading signals
                logger.info("⚡ Generating trading signals...")
                alerts = self.event_executor.generate_trading_signals(
                    articles, market_data, sentiment, self.portfolio
                )
                logger.info(f"   Generated {len(alerts)} trading alerts")
                
                # Show alerts
                for alert in alerts[:10]:
                    logger.info(f"   [{alert.alert_type.upper()}] {alert.symbol}: {alert.action} "
                              f"({alert.trigger}, {alert.confidence:.0%} confidence)")
                
                # 5. Update portfolio status
                self._update_portfolio_status(market_data, sentiment)
                
                # Wait for next update
                logger.info(f"\n⏳ Waiting {self.update_interval}s until next update...")
                await asyncio.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            logger.info("\n⋮ Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.shutdown()
    
    def _update_portfolio_status(self, market_data: Dict, sentiment: Dict):
        """Update and display portfolio status."""
        logger.info("\n📈 PORTFOLIO STATUS:")
        logger.info(f"   Cash: ${self.portfolio['cash']:,.2f}")
        logger.info(f"   Positions: {len(self.portfolio['positions'])}")
        logger.info(f"   Total Value: ${self.portfolio['value']:,.2f}")
        
        uptime = datetime.now() - self.start_time
        logger.info(f"   Uptime: {uptime}")
    
    def shutdown(self):
        """Shutdown monitor gracefully."""
        logger.info("")
        logger.info("="*80)
        logger.info("SHUTTING DOWN LIVE TRADING MONITOR")
        logger.info("="*80)
        
        uptime = datetime.now() - self.start_time
        logger.info(f"Total Runtime: {uptime}")
        logger.info(f"Updates Completed: {len(self.alerts_generated)}")
        logger.info(f"Alerts Generated: {len(self.alerts_generated)}")
        logger.info(f"Trades Executed: {len(self.trades_executed)}")
        logger.info(f"Final Portfolio Value: ${self.portfolio['value']:,.2f}")
        
        self.running = False


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Trading Monitor - Paper Trading with News")
    parser.add_argument("--duration", type=int, help="Run for N seconds")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run monitor
    monitor = LiveTradingMonitor(update_interval=args.interval)
    await monitor.run_monitoring_loop(duration_seconds=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
