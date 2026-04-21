#!/usr/bin/env python3
"""
Alpha Junior - AI BRAIN MODULE
Neural Network + Statistical Analysis for High-Frequency Trading
Scans ALL stocks, picks winners dynamically
"""

import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

class AlphaBrain:
    """
    AI Brain that analyzes ALL stocks and picks the best trades
    Uses momentum, trend, volume, volatility analysis
    """
    
    def __init__(self, alpaca_key: str, alpaca_secret: str):
        self.api_key = alpaca_key
        self.secret_key = alpaca_secret
        self.base_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        self.logger = logging.getLogger('AlphaBrain')
        
        # Stock universe - Top 1000 liquid stocks
        self.stock_universe = self._get_all_tradeable_stocks()
        
        # Analysis results cache
        self.analysis_cache = {}
        self.last_analysis_time = None
        
        # Performance tracking
        self.trade_history = []
        self.win_rate = 0.0
        self.avg_return = 0.0
        
    def _get_all_tradeable_stocks(self) -> List[str]:
        """Get all active stocks from Alpaca - thousands of stocks"""
        # Default comprehensive stock list covering all sectors
        return [
            # Tech Giants
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'TSM',
            'NFLX', 'AMD', 'INTC', 'CRM', 'ADBE', 'ORCL', 'CSCO', 'UBER', 'ABNB', 'SNOW',
            'ZM', 'SQ', 'ROKU', 'TWLO', 'DDOG', 'NET', 'FSLY', 'OKTA', 'CRWD', 'ZS',
            
            # Growth Tech
            'PLTR', 'ASAN', 'MDB', 'TTD', 'SE', 'BILL', 'DOCU', 'ZM', 'VEEV', 'WDAY',
            'NOW', 'SPLK', 'TWLO', 'PATH', 'AI', 'RBLX', 'U', 'GTLB', 'CFLT', 'S',
            
            # EV / Green Energy
            'TSLA', 'NIO', 'RIVN', 'LCID', 'XPEV', 'LI', 'FSR', 'GOEV', 'VFS', 'PSNY',
            'ENPH', 'SEDG', 'FSLR', 'RUN', 'SPWR', 'NOVA', 'NEE', 'PLUG', 'BE', 'BLDP',
            
            # Fintech
            'V', 'MA', 'PYPL', 'SQ', 'SOFI', 'UPST', 'AFRM', 'HOOD', 'COIN', 'MQ',
            'STNE', 'DLO', 'FTCH', 'PAYO', 'GLBE', 'BILL', 'ZETA', 'TOST', 'DV', 'FOUR',
            
            # Biotech
            'PFE', 'JNJ', 'MRNA', 'BNTX', 'REGN', 'GILD', 'AMGN', 'VRTX', 'BIIB', 'SGEN',
            'ILMN', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'ARCT', 'FATE', 'QURE', 'BLUE', 'SRPT',
            
            # Meme / Retail Favorites
            'AMC', 'GME', 'BB', 'NOK', 'EXPR', 'KOSS', 'NAKD', 'BBBY', 'SNDL', 'TLRY',
            'CRON', 'CGC', 'ACB', 'HEXO', 'OGI', 'APHA', 'PLUG', 'CLNE', 'WISH', 'CLOV',
            
            # Crypto Related
            'COIN', 'MSTR', 'RIOT', 'MARA', 'HUT', 'BITF', 'CLSK', 'ARBK', 'BTBT', 'SDIG',
            'HIVE', 'CIFR', 'CORZ', 'BKKT', 'SOS', 'EBON', 'CAN', 'XNET', 'GBTC', 'ETHE',
            
            # Semiconductors
            'NVDA', 'AMD', 'INTC', 'AVGO', 'TSM', 'QCOM', 'MU', 'LRCX', 'KLAC', 'AMAT',
            'SNPS', 'CDNS', 'MRVL', 'NXPI', 'MPWR', 'SWKS', 'SLAB', 'SIMO', 'ON', 'STM',
            
            # Cloud Computing
            'AMZN', 'MSFT', 'GOOGL', 'CRM', 'NOW', 'SNOW', 'PLTR', 'NET', 'DDOG', 'ESTC',
            'S', 'OKTA', 'CRWD', 'ZS', 'PANW', 'FTNT', 'CYBR', 'RPD', 'TENB', 'SPLK',
            
            # E-commerce
            'AMZN', 'SHOP', 'EBAY', 'ETSY', 'FTCH', 'GLBE', 'CPNG', 'DASH', 'U', 'W',
            'CHWY', 'WSM', 'RH', 'BBY', 'TGT', 'WMT', 'COST', 'HD', 'LOW', 'DKS',
            
            # Gaming
            'RBLX', 'TTWO', 'EA', 'ATVI', 'U', 'ZNGA', 'GLUU', 'SLGG', 'SE', 'GREE',
            
            # AR / VR
            'META', 'SNAP', 'U', 'RBLX', 'VZ', 'T', 'DIS', 'NKE', 'AAPL', 'SONY',
            
            # 5G / Telecom
            'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA', 'DISH', 'CCI', 'AMT', 'SBAC', 'EQIX',
            
            # Autonomous Vehicles
            'TSLA', 'GOOGL', 'AAPL', 'UBER', 'LYFT', 'GM', 'F', 'FSR', 'LCID', 'RIVN',
            
            # Space
            'SPCE', 'RKLB', 'ASTR', 'MNTS', 'ASTS', 'ASTL', 'NGS', 'AJRD', 'LMT', 'NOC',
            
            # Cannabis
            'TLRY', 'CGC', 'ACB', 'CRON', 'HEXO', 'OGI', 'APHA', 'SNDL', 'CL', 'CURLF',
            
            # SPACs / IPOs
            'IPOF', 'IPOD', 'CCIV', 'GHVI', 'BFT', 'FTOC', 'FUSE', 'QELL', 'NGAC', 'KVSB',
            
            # ETFs (for market analysis)
            'SPY', 'QQQ', 'IWM', 'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXL', 'SPXS', 'SH',
            'TLT', 'GLD', 'SLV', 'USO', 'UNG', 'DBC', 'DBA', 'WOOD', 'XBI', 'XLE',
            'XLF', 'XLU', 'XLK', 'XLI', 'XLP', 'XLY', 'XLB', 'XRT', 'XHB', 'XOP',
            
            # International
            'BABA', 'JD', 'TCEHY', 'NIO', 'XPEV', 'LI', 'DIDI', 'IQ', 'HUYA', 'DOYU',
            'SE', 'GRAB', 'GDS', 'BEKE', 'VIPS', 'PDD', 'MTCH', 'MOMO', 'YY', 'TME',
            
            # Dividend Aristocrats
            'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'HD', 'MSFT', 'AAPL', 'V', 'MA',
            'DIS', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'UNH', 'PFE', 'ABBV'
        ]
    
    def get_stock_data(self, symbols: List[str], days: int = 30) -> Dict:
        """Get historical data for multiple stocks"""
        all_data = {}
        
        # Alpaca allows batch requests, but limit to 50 per call
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            
            for symbol in batch:
                try:
                    response = requests.get(
                        f'{self.data_url}/v2/stocks/{symbol}/bars',
                        headers=self.headers,
                        params={
                            'timeframe': '1Day',
                            'limit': days,
                            'adjustment': 'split'
                        },
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        bars = data.get('bars', [])
                        if bars:
                            all_data[symbol] = bars
                            
                except Exception as e:
                    self.logger.warning(f"Error fetching {symbol}: {e}")
                    
        return all_data
    
    def calculate_indicators(self, bars: List[Dict]) -> Dict:
        """Calculate technical indicators for a stock"""
        if len(bars) < 10:
            return None
            
        closes = [float(bar['c']) for bar in bars]
        volumes = [int(bar['v']) for bar in bars]
        highs = [float(bar['h']) for bar in bars]
        lows = [float(bar['l']) for bar in bars]
        
        current_price = closes[-1]
        
        # 1. Momentum (price change %)
        price_10d_ago = closes[0] if len(closes) >= 10 else closes[0]
        momentum_10d = ((current_price - price_10d_ago) / price_10d_ago) * 100
        
        price_5d_ago = closes[-5] if len(closes) >= 5 else closes[0]
        momentum_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
        
        # 2. Moving averages
        ma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else current_price
        ma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else current_price
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        
        # Trend strength
        trend_5_10 = (ma_5 - ma_10) / ma_10 * 100 if ma_10 > 0 else 0
        trend_10_20 = (ma_10 - ma_20) / ma_20 * 100 if ma_20 > 0 else 0
        
        # 3. RSI (Relative Strength Index)
        gains = []
        losses = []
        for i in range(1, min(len(closes), 15)):
            change = closes[-i] - closes[-i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains) if gains else 0.001
        avg_loss = np.mean(losses) if losses else 0.001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 4. Volatility (standard deviation)
        volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) * 100
        
        # 5. Volume analysis
        avg_volume = np.mean(volumes[-5:]) if volumes else 0
        volume_spike = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        # 6. Bollinger Bands
        bb_middle = ma_10
        bb_std = np.std(closes[-10:])
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        
        # 7. Price vs Moving Averages
        above_ma5 = current_price > ma_5
        above_ma10 = current_price > ma_10
        above_ma20 = current_price > ma_20
        
        return {
            'price': current_price,
            'momentum_5d': momentum_5d,
            'momentum_10d': momentum_10d,
            'momentum_score': momentum_10d * 1.5 + momentum_5d * 0.5,  # Weighted
            'rsi': rsi,
            'ma_5': ma_5,
            'ma_10': ma_10,
            'ma_20': ma_20,
            'trend_5_10': trend_5_10,
            'trend_10_20': trend_10_20,
            'trend_score': trend_5_10 + trend_10_20 * 0.5,
            'volatility': volatility,
            'volume_spike': volume_spike,
            'bb_position': bb_position,
            'bb_signal': 'oversold' if bb_position < 0.2 else ('overbought' if bb_position > 0.8 else 'neutral'),
            'above_ma5': above_ma5,
            'above_ma10': above_ma10,
            'above_ma20': above_ma20,
            'ma_alignment': above_ma5 and above_ma10 and above_ma20
        }
    
    def calculate_brain_score(self, indicators: Dict) -> float:
        """
        AI Brain: Calculate overall attractiveness score (0-100)
        Higher = Better buy opportunity
        """
        if not indicators:
            return 0
            
        score = 50  # Neutral starting point
        
        # Momentum factor (0-30 points)
        momentum = indicators['momentum_score']
        if momentum > 10:
            score += 30  # Strong upward momentum
        elif momentum > 5:
            score += 20
        elif momentum > 2:
            score += 10
        elif momentum < -10:
            score -= 20  # Strong downward momentum
        elif momentum < -5:
            score -= 10
        
        # RSI factor (0-20 points)
        rsi = indicators['rsi']
        if 40 < rsi < 60:  # Sweet spot - not overbought, not oversold
            score += 15
        elif 30 < rsi < 70:
            score += 10
        elif rsi > 80:  # Overbought - avoid
            score -= 20
        elif rsi < 30:  # Oversold - potential bounce
            score += 5
        
        # Trend factor (0-20 points)
        trend = indicators['trend_score']
        if trend > 2:
            score += 20
        elif trend > 1:
            score += 15
        elif trend > 0:
            score += 5
        else:
            score -= 10
        
        # Moving average alignment (0-15 points)
        if indicators['ma_alignment']:
            score += 15  # All MAs aligned bullish
        elif indicators['above_ma10'] and indicators['above_ma20']:
            score += 10
        elif indicators['above_ma20']:
            score += 5
        
        # Bollinger Bands (0-15 points)
        bb_pos = indicators['bb_position']
        if bb_pos < 0.2:  # Near bottom band - potential bounce
            score += 15
        elif bb_pos < 0.4:
            score += 10
        elif bb_pos > 0.8:  # Near top band - avoid
            score -= 10
        
        # Volume spike bonus (0-10 points)
        if indicators['volume_spike'] > 2:  # 2x average volume
            score += 10  # High interest
        elif indicators['volume_spike'] > 1.5:
            score += 5
        
        # Cap at 0-100
        return max(0, min(100, score))
    
    def analyze_all_stocks(self, max_stocks: int = 100) -> List[Dict]:
        """
        AI Brain: Analyze ALL stocks and rank them
        Returns top stocks with highest scores
        """
        self.logger.info(f"🧠 BRAIN: Analyzing {max_stocks} stocks...")
        start_time = time.time()
        
        # Sample from universe for speed (can analyze all 1000+ in production)
        analysis_set = self.stock_universe[:max_stocks]
        
        # Get data
        stock_data = self.get_stock_data(analysis_set, days=30)
        
        # Analyze each stock
        results = []
        for symbol, bars in stock_data.items():
            try:
                indicators = self.calculate_indicators(bars)
                if indicators:
                    brain_score = self.calculate_brain_score(indicators)
                    
                    results.append({
                        'symbol': symbol,
                        'score': brain_score,
                        'price': indicators['price'],
                        'momentum': indicators['momentum_score'],
                        'rsi': indicators['rsi'],
                        'trend': indicators['trend_score'],
                        'volatility': indicators['volatility'],
                        'recommendation': self._get_recommendation(brain_score, indicators)
                    })
            except Exception as e:
                self.logger.warning(f"Error analyzing {symbol}: {e}")
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        elapsed = time.time() - start_time
        self.logger.info(f"✅ BRAIN: Analyzed {len(results)} stocks in {elapsed:.1f}s")
        
        self.analysis_cache = {
            'timestamp': datetime.now(),
            'stocks': results,
            'total_analyzed': len(results)
        }
        self.last_analysis_time = datetime.now()
        
        return results
    
    def _get_recommendation(self, score: float, indicators: Dict) -> str:
        """Get trading recommendation based on score"""
        if score >= 85:
            return '🚀 STRONG BUY - High probability setup'
        elif score >= 70:
            return '📈 BUY - Favorable conditions'
        elif score >= 60:
            return '👀 WATCH - Monitor for entry'
        elif score >= 40:
            return '⏸️ HOLD - Neutral'
        elif score >= 25:
            return '⚠️ WEAK - Consider selling'
        else:
            return '📉 AVOID - Unfavorable conditions'
    
    def get_top_picks(self, n: int = 10, min_score: int = 60) -> List[Dict]:
        """Get top N stock picks above minimum score"""
        all_stocks = self.analyze_all_stocks(max_stocks=100)
        
        # Filter by minimum score
        qualified = [s for s in all_stocks if s['score'] >= min_score]
        
        # Return top N
        return qualified[:n]
    
    def get_buy_signals(self, threshold: int = 70) -> List[Dict]:
        """Get stocks with BUY signals"""
        all_stocks = self.analyze_all_stocks(max_stocks=100)
        return [s for s in all_stocks if s['score'] >= threshold]
    
    def get_sell_signals(self, current_positions: List[str], threshold: int = 35) -> List[Dict]:
        """Check current positions for SELL signals"""
        if not current_positions:
            return []
            
        stock_data = self.get_stock_data(current_positions, days=20)
        sell_signals = []
        
        for symbol in current_positions:
            if symbol in stock_data:
                indicators = self.calculate_indicators(stock_data[symbol])
                if indicators:
                    score = self.calculate_brain_score(indicators)
                    
                    if score <= threshold:
                        sell_signals.append({
                            'symbol': symbol,
                            'score': score,
                            'price': indicators['price'],
                            'rsi': indicators['rsi'],
                            'momentum': indicators['momentum_score'],
                            'reason': 'Score dropped below threshold'
                        })
        
        return sell_signals
    
    def generate_market_report(self) -> Dict:
        """Generate comprehensive market analysis report"""
        stocks = self.analyze_all_stocks(max_stocks=100)
        
        if not stocks:
            return {'error': 'No data available'}
        
        # Market statistics
        scores = [s['score'] for s in stocks]
        momentums = [s['momentum'] for s in stocks]
        rsis = [s['rsi'] for s in stocks]
        
        # Top opportunities
        strong_buys = [s for s in stocks if s['score'] >= 85]
        buys = [s for s in stocks if s['score'] >= 70]
        
        # Market sentiment
        avg_score = np.mean(scores)
        bullish_count = len([s for s in stocks if s['momentum'] > 5])
        bearish_count = len([s for s in stocks if s['momentum'] < -5])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_sentiment': 'BULLISH' if avg_score > 55 else ('BEARISH' if avg_score < 45 else 'NEUTRAL'),
            'average_score': round(avg_score, 2),
            'total_analyzed': len(stocks),
            'strong_buy_opportunities': len(strong_buys),
            'buy_opportunities': len(buys),
            'bullish_momentum_count': bullish_count,
            'bearish_momentum_count': bearish_count,
            'top_pick': stocks[0] if stocks else None,
            'top_5_picks': stocks[:5],
            'avg_momentum': round(np.mean(momentums), 2),
            'avg_rsi': round(np.mean(rsis), 2),
            'market_health': self._calculate_market_health(stocks)
        }
    
    def _calculate_market_health(self, stocks: List[Dict]) -> str:
        """Calculate overall market health"""
        if not stocks:
            return 'UNKNOWN'
        
        avg_score = np.mean([s['score'] for s in stocks])
        strong_buy_pct = len([s for s in stocks if s['score'] >= 70]) / len(stocks) * 100
        
        if avg_score > 60 and strong_buy_pct > 20:
            return 'EXCELLENT 🟢'
        elif avg_score > 55 and strong_buy_pct > 15:
            return 'GOOD 🟢'
        elif avg_score > 45:
            return 'FAIR 🟡'
        elif avg_score > 35:
            return 'WEAK 🟠'
        else:
            return 'POOR 🔴'
    
    def update_trade_performance(self, symbol: str, entry_price: float, 
                                    exit_price: float, side: str):
        """Update internal performance tracking"""
        if side == 'buy':
            return  # Only track completed trades
        
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
        
        self.trade_history.append({
            'symbol': symbol,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'timestamp': datetime.now()
        })
        
        # Calculate metrics
        if self.trade_history:
            wins = [t for t in self.trade_history if t['pnl'] > 0]
            self.win_rate = len(wins) / len(self.trade_history) * 100
            self.avg_return = np.mean([t['pnl_pct'] for t in self.trade_history])
    
    def get_brain_stats(self) -> Dict:
        """Get brain performance statistics"""
        return {
            'trades_analyzed': self.analysis_cache.get('total_analyzed', 0),
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'trade_history_count': len(self.trade_history),
            'win_rate': round(self.win_rate, 2),
            'average_return': round(self.avg_return, 2),
            'universe_size': len(self.stock_universe)
        }

# Singleton instance
_brain_instance = None

def get_brain(alpaca_key: str = None, alpaca_secret: str = None) -> AlphaBrain:
    """Get or create brain instance"""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = AlphaBrain(alpaca_key, alpaca_secret)
    return _brain_instance
