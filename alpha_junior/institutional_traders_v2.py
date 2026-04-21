#!/usr/bin/env python3
"""
Alpha Junior - COMPLETE INSTITUTIONAL TRADING TEAM
14 Specialized AI Traders - Full Coverage of All Strategies
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    SWING_TRADING = "swing_trading"
    SCALPING = "scalping"
    POSITION_TRADING = "position_trading"
    ARBITRAGE = "arbitrage"
    GAP_TRADING = "gap_trading"
    PAIRS_TRADING = "pairs_trading"
    SECTOR_ROTATION = "sector_rotation"
    VOLATILITY = "volatility"
    NEWS_EVENT = "news_event"
    ALGORITHMIC = "algorithmic"

@dataclass
class TradeSignal:
    symbol: str
    side: str
    score: float
    confidence: float
    strategy: StrategyType
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    position_size: int
    time_frame: str
    reasoning: str
    urgency: str  # immediate, high, normal, low

# Base class
class InstitutionalTrader:
    def __init__(self, name: str, strategy: StrategyType, logger: logging.Logger):
        self.name = name
        self.strategy = strategy
        self.logger = logger
        self.performance = {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'avg_return': 0.0, 'total_pnl': 0.0}
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        raise NotImplementedError
    
    def update_performance(self, pnl: float):
        self.performance['trades'] += 1
        if pnl > 0:
            self.performance['wins'] += 1
        else:
            self.performance['losses'] += 1
        self.performance['win_rate'] = (self.performance['wins'] / self.performance['trades'] * 100)
        self.performance['total_pnl'] += pnl
        if self.performance['trades'] > 0:
            self.performance['avg_return'] = self.performance['total_pnl'] / self.performance['trades']

# ============ 1. MOMENTUM TRADER ============
class MomentumTrader(InstitutionalTrader):
    """High-momentum breakout specialist"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Momentum Master", StrategyType.MOMENTUM, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 20:
            return None
        
        closes = [float(bar['c']) for bar in data]
        volumes = [int(bar['v']) for bar in data]
        current_price = closes[-1]
        
        # Multi-timeframe momentum
        momentum_5d = ((closes[-1] - closes[-5]) / closes[-5]) * 100
        momentum_10d = ((closes[-1] - closes[-10]) / closes[-10]) * 100
        momentum_20d = ((closes[-1] - closes[-20]) / closes[-20]) * 100
        
        # Volume surge
        avg_volume = np.mean(volumes[-20:])
        volume_surge = volumes[-1] / avg_volume
        
        # Score calculation
        score = 0
        if momentum_20d > 15 and momentum_10d > 8:
            score += 35
        elif momentum_20d > 10:
            score += 25
        
        if volume_surge > 2.5:
            score += 30
        elif volume_surge > 1.8:
            score += 20
        
        # RSI check
        rsi = self._calculate_rsi(closes)
        if 40 < rsi < 70:
            score += 20
        
        if score >= 70:
            atr = self._calculate_atr(data)
            target = current_price * 1.15
            stop = current_price - (atr * 1.5)
            
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=min(95, score + 10),
                strategy=StrategyType.MOMENTUM, entry_price=current_price,
                target_price=target, stop_loss=stop,
                risk_reward_ratio=(target-current_price)/(current_price-stop) if (current_price-stop) > 0 else 2.0,
                position_size=20, time_frame='1-5 days',
                reasoning=f"Strong momentum {momentum_20d:.1f}% 20d, volume {volume_surge:.1f}x",
                urgency='high' if score > 85 else 'normal'
            )
        return None
    
    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        gains, losses = [], []
        for i in range(1, min(len(closes), period + 1)):
            change = closes[-i] - closes[-i-1]
            gains.append(change if change > 0 else 0)
            losses.append(abs(change) if change < 0 else 0)
        avg_gain = np.mean(gains) if gains else 0.001
        avg_loss = np.mean(losses) if losses else 0.001
        return 100 - (100 / (1 + avg_gain / avg_loss))
    
    def _calculate_atr(self, data: List[Dict]) -> float:
        highs = [float(bar['h']) for bar in data[-14:]]
        lows = [float(bar['l']) for bar in data[-14:]]
        closes = [float(bar['c']) for bar in data[-14:]]
        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) 
               for i in range(1, len(closes))]
        return np.mean(trs) if trs else 1.0

# ============ 2. MEAN REVERSION TRADER ============
class MeanReversionTrader(InstitutionalTrader):
    """Statistical arbitrage - oversold bounce specialist"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Reversion King", StrategyType.MEAN_REVERSION, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 30:
            return None
        
        closes = [float(bar['c']) for bar in data]
        highs = [float(bar['h']) for bar in data]
        lows = [float(bar['l']) for bar in data]
        current_price = closes[-1]
        
        sma_20 = np.mean(closes[-20:])
        std_20 = np.std(closes[-20:])
        bb_lower = sma_20 - (std_20 * 2)
        bb_position = (current_price - bb_lower) / (sma_20 - bb_lower) if (sma_20 - bb_lower) > 0 else 0.5
        
        rsi = self._calculate_rsi(closes)
        z_score = (current_price - sma_20) / std_20 if std_20 > 0 else 0
        
        score = 0
        if rsi < 25 and z_score < -2.0:
            score += 40
        elif rsi < 35 and z_score < -1.5:
            score += 30
        
        if bb_position < 0.1:
            score += 25
        elif bb_position < 0.2:
            score += 15
        
        if score >= 65:
            target = sma_20
            stop = current_price * 0.95
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=min(90, score + 5),
                strategy=StrategyType.MEAN_REVERSION, entry_price=current_price,
                target_price=target, stop_loss=stop,
                risk_reward_ratio=2.0 if target > current_price else 1.5,
                position_size=25, time_frame='2-7 days',
                reasoning=f"Oversold: RSI {rsi:.0f}, BB% {bb_position*100:.0f}, Z-score {z_score:.1f}",
                urgency='high' if score > 80 else 'normal'
            )
        return None
    
    def _calculate_rsi(self, closes: List[float]) -> float:
        gains, losses = [], []
        for i in range(1, min(15, len(closes))):
            change = closes[-i] - closes[-i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        avg_gain = np.mean(gains) if gains else 0.001
        avg_loss = np.mean(losses) if losses else 0.001
        return 100 - (100 / (1 + avg_gain / avg_loss))

# ============ 3. BREAKOUT TRADER ============
class BreakoutTrader(InstitutionalTrader):
    """Pattern breakout specialist"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Breakout Pro", StrategyType.BREAKOUT, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 30:
            return None
        
        closes = [float(bar['c']) for bar in data]
        highs = [float(bar['h']) for bar in data]
        lows = [float(bar['l']) for bar in data]
        volumes = [int(bar['v']) for bar in data]
        current_price = closes[-1]
        
        resistance = max(highs[-20:])
        support = min(lows[-20:])
        consolidation_range = (max(highs[-10:]) - min(lows[-10:])) / np.mean(closes[-10:]) * 100
        
        volume_ratio = volumes[-1] / np.mean(volumes[-20:])
        price_vs_resistance = ((current_price - resistance) / resistance) * 100
        
        score = 0
        if price_vs_resistance > 2 and consolidation_range < 8:
            score += 40
        elif price_vs_resistance > 1.5:
            score += 25
        
        if volume_ratio > 2.5:
            score += 30
        elif volume_ratio > 1.8:
            score += 20
        
        if consolidation_range < 5:
            score += 10
        
        if score >= 70:
            measured_move = resistance - support
            target = current_price + measured_move * 0.7
            stop = min(support, min(lows[-3:]))
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=min(92, score + 8),
                strategy=StrategyType.BREAKOUT, entry_price=current_price,
                target_price=target, stop_loss=stop,
                risk_reward_ratio=(target-current_price)/(current_price-stop) if (current_price-stop) > 0 else 2.5,
                position_size=20, time_frame='1-4 weeks',
                reasoning=f"Breakout: +{price_vs_resistance:.1f}% above resistance, {volume_ratio:.1f}x volume",
                urgency='high'
            )
        return None

# ============ 4. TREND FOLLOWING TRADER ============
class TrendFollowingTrader(InstitutionalTrader):
    """Long-term trend rider"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Trend Rider", StrategyType.TREND_FOLLOWING, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 50:
            return None
        
        closes = [float(bar['c']) for bar in data]
        highs = [float(bar['h']) for bar in data]
        lows = [float(bar['l']) for bar in data]
        current_price = closes[-1]
        
        ema_9 = self._calculate_ema(closes, 9)
        ema_21 = self._calculate_ema(closes, 21)
        ema_50 = self._calculate_ema(closes, 50)
        
        ma_aligned = current_price > ema_9 > ema_21 > ema_50
        
        price_slope_10 = np.polyfit(range(10), closes[-10:], 1)[0] if len(closes) >= 10 else 0
        price_slope_30 = np.polyfit(range(30), closes[-30:], 1)[0] if len(closes) >= 30 else 0
        trend_acceleration = price_slope_10 > price_slope_30 > 0
        
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        hh_hl_pattern = max(recent_highs[-10:]) > max(recent_highs[:10]) and min(recent_lows[-10:]) > min(recent_lows[:10])
        
        score = 0
        if ma_aligned and ema_9 > ema_21 > ema_50:
            score += 35
        elif current_price > ema_21 > ema_50:
            score += 25
        
        if trend_acceleration:
            score += 25
        elif price_slope_10 > 0:
            score += 15
        
        if hh_hl_pattern:
            score += 20
        
        if score >= 65:
            target = current_price * 1.20
            stop = ema_21 * 0.95
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=min(88, score),
                strategy=StrategyType.TREND_FOLLOWING, entry_price=current_price,
                target_price=target, stop_loss=stop,
                risk_reward_ratio=4.0,
                position_size=15, time_frame='2-8 weeks',
                reasoning="Strong uptrend with EMA alignment and HH/HL pattern",
                urgency='normal'
            )
        return None
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

# ============ 5. SWING TRADER ============
class SwingTrader(InstitutionalTrader):
    """3-10 day swing trading specialist"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Swing Trader", StrategyType.SWING_TRADING, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 30:
            return None
        
        closes = [float(bar['c']) for bar in data]
        highs = [float(bar['h']) for bar in data]
        lows = [float(bar['l']) for bar in data]
        current_price = closes[-1]
        
        # Find swing low setup
        recent_low = min(lows[-10:])
        bounce_from_low = ((current_price - recent_low) / recent_low) * 100
        
        # Check for support hold
        higher_lows = lows[-5] > lows[-10] if len(lows) >= 10 else False
        
        # Volume confirmation
        volumes = [int(bar['v']) for bar in data]
        vol_increase = volumes[-1] > np.mean(volumes[-5:]) * 1.3
        
        # Stochastic (simplified)
        lowest_low = min(lows[-14:])
        highest_high = max(highs[-14:])
        k_percent = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100 if (highest_high - lowest_low) > 0 else 50
        
        score = 0
        if bounce_from_low > 3 and bounce_from_low < 8:
            score += 30
        if higher_lows:
            score += 20
        if vol_increase:
            score += 15
        if k_percent > 20 and k_percent < 50:
            score += 20
        
        if score >= 60:
            recent_high = max(highs[-20:-5]) if len(highs) >= 20 else max(highs) * 1.05
            target = recent_high
            stop = recent_low * 0.98
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=min(85, score + 5),
                strategy=StrategyType.SWING_TRADING, entry_price=current_price,
                target_price=target, stop_loss=stop,
                risk_reward_ratio=2.5,
                position_size=25, time_frame='3-10 days',
                reasoning=f"Swing setup: Bounce {bounce_from_low:.1f}% from low, higher lows forming",
                urgency='normal'
            )
        return None

# ============ 6. SCALPING TRADER ============
class ScalpingTrader(InstitutionalTrader):
    """Quick intraday moves - rapid small gains"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Scalper", StrategyType.SCALPING, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        # Scalping requires intraday data - simplified for daily bars
        if len(data) < 5:
            return None
        
        closes = [float(bar['c']) for bar in data]
        volumes = [int(bar['v']) for bar in data]
        current_price = closes[-1]
        
        # Look for tight range breakout
        range_3d = (max(closes[-3:]) - min(closes[-3:])) / np.mean(closes[-3:]) * 100
        volume_burst = volumes[-1] > np.mean(volumes[-10:]) * 2
        
        # Quick momentum
        change_1d = ((closes[-1] - closes[-2]) / closes[-2]) * 100
        
        score = 0
        if range_3d < 2 and volume_burst and change_1d > 1.5:
            score = 75
        
        if score >= 70:
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=80,
                strategy=StrategyType.SCALPING, entry_price=current_price,
                target_price=current_price * 1.03, stop_loss=current_price * 0.99,
                risk_reward_ratio=3.0,
                position_size=50, time_frame='1-3 days',
                reasoning=f"Scalp setup: Tight range breakout, volume burst {volume_burst}",
                urgency='immediate'
            )
        return None

# ============ 7. POSITION TRADER ============
class PositionTrader(InstitutionalTrader):
    """Long-term position holding - weeks to months"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Position Trader", StrategyType.POSITION_TRADING, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 100:
            return None
        
        closes = [float(bar['c']) for bar in data]
        current_price = closes[-1]
        
        # Long-term trend
        ma_50 = np.mean(closes[-50:])
        ma_200 = np.mean(closes[-200:])
        
        # Golden cross check (simplified)
        golden_cross = ma_50 > ma_200 and np.mean(closes[-60:-50]) < np.mean(closes[-10:])
        
        # Long-term momentum
        momentum_50d = ((closes[-1] - closes[-50]) / closes[-50]) * 100
        
        score = 0
        if golden_cross:
            score += 40
        if ma_50 > ma_200:
            score += 20
        if momentum_50d > 15:
            score += 25
        elif momentum_50d > 8:
            score += 15
        
        if score >= 65:
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=min(90, score),
                strategy=StrategyType.POSITION_TRADING, entry_price=current_price,
                target_price=current_price * 1.35, stop_loss=ma_200 * 0.95,
                risk_reward_ratio=5.0,
                position_size=10, time_frame='1-6 months',
                reasoning=f"Long-term trend: 50d momentum {momentum_50d:.1f}%, MA alignment",
                urgency='low'
            )
        return None

# ============ 8. ARBITRAGE TRADER ============
class ArbitrageTrader(InstitutionalTrader):
    """Statistical arbitrage and mispricings"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Arbitrage Hunter", StrategyType.ARBITRAGE, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 50:
            return None
        
        closes = [float(bar['c']) for bar in data]
        current_price = closes[-1]
        
        # Look for deviation from mean
        ma_50 = np.mean(closes[-50:])
        deviation = ((current_price - ma_50) / ma_50) * 100
        
        # Historical volatility
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100
        
        # If price deviates significantly but not extremely
        if -8 < deviation < -3 and volatility < 40:
            score = 70 + abs(deviation) * 2
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=85,
                strategy=StrategyType.ARBITRAGE, entry_price=current_price,
                target_price=ma_50, stop_loss=current_price * 0.95,
                risk_reward_ratio=2.0,
                position_size=30, time_frame='5-15 days',
                reasoning=f"Mean reversion arbitrage: {deviation:.1f}% below 50d MA",
                urgency='normal'
            )
        return None

# ============ 9. GAP TRADER ============
class GapTrader(InstitutionalTrader):
    """Trading overnight/weekly gaps"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Gap Filler", StrategyType.GAP_TRADING, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 10:
            return None
        
        closes = [float(bar['c']) for bar in data]
        opens = [float(bar['o']) for bar in data]
        lows = [float(bar['l']) for bar in data]
        current_price = closes[-1]
        
        # Check for gap down
        prev_close = closes[-2] if len(closes) >= 2 else closes[-1]
        gap_down = ((opens[-1] - prev_close) / prev_close) * 100
        
        # Gap fill potential (if price is moving back toward prev close)
        fill_progress = ((current_price - opens[-1]) / (prev_close - opens[-1])) * 100 if (prev_close - opens[-1]) != 0 else 0
        
        score = 0
        if gap_down < -3 and fill_progress > 30:
            score = 65 + abs(gap_down) * 2
        
        if score >= 65:
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=80,
                strategy=StrategyType.GAP_TRADING, entry_price=current_price,
                target_price=prev_close, stop_loss=min(lows[-2:]) * 0.99,
                risk_reward_ratio=2.5,
                position_size=35, time_frame='1-5 days',
                reasoning=f"Gap fill play: {gap_down:.1f}% gap, {fill_progress:.0f}% fill progress",
                urgency='high'
            )
        return None

# ============ 10. PAIRS TRADING ============
class PairsTrader(InstitutionalTrader):
    """Correlation-based pairs trading"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Pairs Trader", StrategyType.PAIRS_TRADING, logger)
        self.pairs = {
            'AAPL': 'MSFT', 'MSFT': 'AAPL',
            'JPM': 'BAC', 'BAC': 'JPM',
            'XOM': 'CVX', 'CVX': 'XOM',
            'AMD': 'NVDA', 'NVDA': 'AMD',
            'COIN': 'MSTR', 'MSTR': 'COIN'
        }
    
    def analyze(self, symbol: str, data: List[Dict], pair_data: List[Dict] = None) -> Optional[TradeSignal]:
        # Simplified - would need both stocks' data
        return None

# ============ 11. SECTOR ROTATION TRADER ============
class SectorRotationTrader(InstitutionalTrader):
    """Rotates between sectors based on strength"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Sector Rotator", StrategyType.SECTOR_ROTATION, logger)
        self.sectors = {
            'technology': ['AAPL', 'MSFT', 'NVDA', 'AMD'],
            'finance': ['JPM', 'BAC', 'V', 'MA'],
            'healthcare': ['JNJ', 'PFE', 'UNH'],
            'energy': ['XOM', 'CVX']
        }
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 40:
            return None
        
        closes = [float(bar['c']) for bar in data]
        current_price = closes[-1]
        
        # Sector momentum
        momentum_20d = ((closes[-1] - closes[-20]) / closes[-20]) * 100
        relative_strength = momentum_20d - 5  # vs market average assumption
        
        # RSI for timing
        rsi = self._calculate_rsi(closes)
        
        score = 0
        if relative_strength > 8 and 40 < rsi < 70:
            score = 70 + min(relative_strength, 15)
        
        if score >= 70:
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=85,
                strategy=StrategyType.SECTOR_ROTATION, entry_price=current_price,
                target_price=current_price * 1.12, stop_loss=current_price * 0.95,
                risk_reward_ratio=2.4,
                position_size=20, time_frame='2-6 weeks',
                reasoning=f"Sector leader: RS +{relative_strength:.1f}%, RSI {rsi:.0f}",
                urgency='normal'
            )
        return None
    
    def _calculate_rsi(self, closes: List[float]) -> float:
        gains, losses = [], []
        for i in range(1, min(15, len(closes))):
            change = closes[-i] - closes[-i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        avg_gain = np.mean(gains) if gains else 0.001
        avg_loss = np.mean(losses) if losses else 0.001
        return 100 - (100 / (1 + avg_gain / avg_loss))

# ============ 12. VOLATILITY TRADER ============
class VolatilityTrader(InstitutionalTrader):
    """Trades volatility expansion and contraction"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Volatility Master", StrategyType.VOLATILITY, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 30:
            return None
        
        closes = [float(bar['c']) for bar in data]
        highs = [float(bar['h']) for bar in data]
        lows = [float(bar['l']) for bar in data]
        current_price = closes[-1]
        
        # ATR-based volatility
        atrs = []
        for i in range(1, min(15, len(closes))):
            tr = max(highs[-i] - lows[-i], abs(highs[-i] - closes[-i-1]), abs(lows[-i] - closes[-i-1]))
            atrs.append(tr)
        current_atr = np.mean(atrs) if atrs else 1
        
        # Historical ATR
        hist_atrs = []
        for i in range(15, min(30, len(closes))):
            if len(highs) > i and len(lows) > i and len(closes) > i:
                tr = max(highs[-i] - lows[-i], abs(highs[-i] - closes[-i-1]), abs(lows[-i] - closes[-i-1]))
                hist_atrs.append(tr)
        hist_atr = np.mean(hist_atrs) if hist_atrs else current_atr
        
        # Volatility expansion
        vol_expansion = current_atr / hist_atr if hist_atr > 0 else 1
        
        # Bollinger Band squeeze breakout
        sma_20 = np.mean(closes[-20:])
        std_20 = np.std(closes[-20:])
        bb_width = (std_20 * 4) / sma_20 * 100 if sma_20 > 0 else 10
        
        score = 0
        if vol_expansion > 1.5 and closes[-1] > closes[-2]:
            score = 70 + min((vol_expansion - 1.5) * 20, 20)
        elif bb_width < 5 and vol_expansion > 1.3:  # Squeeze breakout
            score = 75
        
        if score >= 65:
            return TradeSignal(
                symbol=symbol, side='buy', score=score, confidence=80,
                strategy=StrategyType.VOLATILITY, entry_price=current_price,
                target_price=current_price + current_atr * 2, stop_loss=current_price - current_atr,
                risk_reward_ratio=2.0,
                position_size=25, time_frame='2-7 days',
                reasoning=f"Volatility expansion: {vol_expansion:.1f}x normal ATR",
                urgency='high'
            )
        return None

# ============ 13. NEWS/EVENT TRADER ============
class NewsEventTrader(InstitutionalTrader):
    """Post-earnings and event-driven trading"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Event Trader", StrategyType.NEWS_EVENT, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 20:
            return None
        
        closes = [float(bar['c']) for bar in data]
        volumes = [int(bar['v']) for bar in data]
        current_price = closes[-1]
        
        # Look for post-event consolidation (earnings drift pattern)
        big_volume_day = max(volumes[-10:])
        big_volume_idx = len(volumes) - 10 + volumes[-10:].index(big_volume_day)
        days_since_event = len(volumes) - big_volume_idx
        
        if 2 <= days_since_event <= 5:
            # Post-event consolidation
            consolidation_range = (max(closes[-days_since_event:]) - min(closes[-days_since_event:])) / np.mean(closes[-days_since_event:]) * 100
            
            if consolidation_range < 5:
                score = 75 - days_since_event * 2
                
                return TradeSignal(
                    symbol=symbol, side='buy', score=score, confidence=78,
                    strategy=StrategyType.NEWS_EVENT, entry_price=current_price,
                    target_price=current_price * 1.08, stop_loss=current_price * 0.96,
                    risk_reward_ratio=2.0,
                    position_size=30, time_frame='3-10 days',
                    reasoning=f"Post-event consolidation after volume spike {days_since_event} days ago",
                    urgency='normal'
                )
        return None

# ============ 14. ALGORITHMIC/MACHINE LEARNING TRADER ============
class AlgorithmicTrader(InstitutionalTrader):
    """Pattern recognition and algorithmic signals"""
    def __init__(self, logger: logging.Logger):
        super().__init__("Algo Master", StrategyType.ALGORITHMIC, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 40:
            return None
        
        closes = [float(bar['c']) for bar in data]
        volumes = [int(bar['v']) for bar in data]
        current_price = closes[-1]
        
        # Composite score using multiple factors
        scores = []
        
        # 1. Price pattern (higher lows)
        if len(closes) >= 20:
            lows_1 = min(closes[-20:-10])
            lows_2 = min(closes[-10:])
            if lows_2 > lows_1 * 0.98:
                scores.append(15)
        
        # 2. Volume pattern (accumulation)
        if len(volumes) >= 10:
            vol_recent = np.mean(volumes[-5:])
            vol_old = np.mean(volumes[-10:-5])
            if vol_recent > vol_old * 1.2:
                scores.append(15)
        
        # 3. Moving average cluster
        ma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else current_price
        ma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else current_price
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        
        ma_cluster = abs(ma_5 - ma_10) / ma_10 * 100 + abs(ma_10 - ma_20) / ma_20 * 100
        if ma_cluster < 3:  # Tight MA cluster
            scores.append(20)
        
        # 4. Momentum alignment
        mom_5 = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
        mom_10 = ((closes[-1] - closes[-10]) / closes[-10]) * 100 if len(closes) >= 10 else 0
        
        if mom_5 > 2 and mom_10 > 1:
            scores.append(20)
        
        # 5. Support level test
        support = min(closes[-20:]) if len(closes) >= 20 else min(closes)
        support_test = abs(current_price - support) / support * 100
        if support_test < 2:
            scores.append(20)
        
        total_score = sum(scores)
        
        if total_score >= 60:
            return TradeSignal(
                symbol=symbol, side='buy', score=total_score, confidence=min(85, total_score + 5),
                strategy=StrategyType.ALGORITHMIC, entry_price=current_price,
                target_price=current_price * 1.10, stop_loss=current_price * 0.95,
                risk_reward_ratio=2.0,
                position_size=25, time_frame='3-15 days',
                reasoning=f"Algo composite: {len(scores)}/5 factors positive, score {total_score}",
                urgency='normal'
            )
        return None

# ============ TRADING TEAM MANAGER ============
class CompleteTradingTeam:
    """All 14 traders working together"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.traders = {
            'momentum': MomentumTrader(logger),
            'mean_reversion': MeanReversionTrader(logger),
            'breakout': BreakoutTrader(logger),
            'trend_following': TrendFollowingTrader(logger),
            'swing_trading': SwingTrader(logger),
            'scalping': ScalpingTrader(logger),
            'position_trading': PositionTrader(logger),
            'arbitrage': ArbitrageTrader(logger),
            'gap_trading': GapTrader(logger),
            'sector_rotation': SectorRotationTrader(logger),
            'volatility': VolatilityTrader(logger),
            'news_event': NewsEventTrader(logger),
            'algorithmic': AlgorithmicTrader(logger),
            'pairs_trading': PairsTrader(logger)
        }
        self.logger.info(f"🎩 Complete Trading Team initialized: {len(self.traders)} specialized traders")
    
    def analyze_stock(self, symbol: str, data: List[Dict]) -> List[TradeSignal]:
        """Get signals from all applicable traders"""
        all_signals = []
        
        for name, trader in self.traders.items():
            try:
                signal = trader.analyze(symbol, data)
                if signal:
                    all_signals.append(signal)
                    self.logger.info(
                        f"🎯 {trader.name}: {symbol} Score {signal.score:.0f} | "
                        f"{signal.strategy.value} | R/R {signal.risk_reward_ratio:.1f}"
                    )
            except Exception as e:
                self.logger.debug(f"{trader.name} error on {symbol}: {e}")
        
        return all_signals
    
    def get_consensus_opportunities(self, symbol_data: Dict[str, List[Dict]], min_score: int = 70) -> List[TradeSignal]:
        """Find high-conviction opportunities"""
        opportunities = []
        
        for symbol, data in symbol_data.items():
            signals = self.analyze_stock(symbol, data)
            
            if signals:
                # Get best signal
                best = max(signals, key=lambda x: x.score)
                
                # Boost if multiple traders agree
                if len(signals) >= 2:
                    best.score += len(signals) * 5
                    best.confidence = min(98, best.confidence + 10)
                    best.reasoning += f" [CONSENSUS: {len(signals)} traders]"
                    self.logger.info(f"🔥 CONSENSUS: {symbol} - {len(signals)} traders agree, Score {best.score:.0f}")
                
                if best.score >= min_score:
                    opportunities.append(best)
        
        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities
    
    def get_team_performance(self) -> Dict:
        return {
            name: {
                'name': trader.name,
                'strategy': trader.strategy.value,
                'performance': trader.performance
            }
            for name, trader in self.traders.items()
        }

# Singleton
team_instance = None

def get_complete_team(logger: logging.Logger = None) -> CompleteTradingTeam:
    global team_instance
    if team_instance is None:
        if logger is None:
            logger = logging.getLogger('CompleteTradingTeam')
        team_instance = CompleteTradingTeam(logger)
    return team_instance
