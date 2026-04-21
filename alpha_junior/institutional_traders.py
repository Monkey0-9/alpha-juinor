#!/usr/bin/env python3
"""
Alpha Junior - INSTITUTIONAL TRADING TEAM
Elite group of AI traders working together like top 1% hedge funds
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    ARBITRAGE = "arbitrage"
    SWING_TRADING = "swing_trading"
    SCALPING = "scalping"
    POSITION_TRADING = "position_trading"

@dataclass
class TradeSignal:
    symbol: str
    side: str  # buy or sell
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

class InstitutionalTrader:
    """Base class for institutional-grade traders"""
    
    def __init__(self, name: str, strategy: StrategyType, logger: logging.Logger):
        self.name = name
        self.strategy = strategy
        self.logger = logger
        self.performance = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'total_pnl': 0.0
        }
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        """Analyze stock and return trade signal if opportunity found"""
        raise NotImplementedError
    
    def update_performance(self, pnl: float):
        """Update trader performance metrics"""
        self.performance['trades'] += 1
        if pnl > 0:
            self.performance['wins'] += 1
        else:
            self.performance['losses'] += 1
        
        self.performance['win_rate'] = (
            self.performance['wins'] / self.performance['trades'] * 100
        )
        self.performance['total_pnl'] += pnl
        
        # Calculate average return
        if self.performance['trades'] > 0:
            self.performance['avg_return'] = (
                self.performance['total_pnl'] / self.performance['trades']
            )

class MomentumTrader(InstitutionalTrader):
    """
    Top 1% Momentum Trader
    Specializes in high-momentum stocks with institutional backing
    Uses: Volume Profile, Relative Strength, Money Flow Index
    """
    
    def __init__(self, logger: logging.Logger):
        super().__init__("Momentum Master", StrategyType.MOMENTUM, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 20:
            return None
        
        closes = [float(bar['c']) for bar in data]
        volumes = [int(bar['v']) for bar in data]
        highs = [float(bar['h']) for bar in data]
        lows = [float(bar['l']) for bar in data]
        
        current_price = closes[-1]
        
        # 1. Price Momentum (multi-timeframe)
        momentum_5d = ((closes[-1] - closes[-5]) / closes[-5]) * 100
        momentum_10d = ((closes[-1] - closes[-10]) / closes[-10]) * 100
        momentum_20d = ((closes[-1] - closes[-20]) / closes[-20]) * 100
        
        # 2. Volume Analysis (Institutional Money Flow)
        avg_volume_20d = np.mean(volumes[-20:])
        volume_today = volumes[-1]
        volume_surge = volume_today / avg_volume_20d
        
        # Calculate Volume Weighted Average Price (VWAP)
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:])]
        vwap = np.average(typical_prices, weights=volumes[-20:])
        
        price_vs_vwap = ((current_price - vwap) / vwap) * 100
        
        # 3. Money Flow Index (MFI) - Volume-weighted RSI
        raw_money_flow = [tp * v for tp, v in zip(typical_prices[-14:], volumes[-14:])]
        positive_flow = sum([mf for mf, tp, prev_tp in zip(
            raw_money_flow[1:], typical_prices[-13:], typical_prices[-14:-1]
        ) if tp > prev_tp])
        negative_flow = sum([mf for mf, tp, prev_tp in zip(
            raw_money_flow[1:], typical_prices[-13:], typical_prices[-14:-1]
        ) if tp < prev_tp])
        
        money_flow_ratio = positive_flow / negative_flow if negative_flow > 0 else 999
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        # 4. Relative Strength vs Market (simplified - assume SPY benchmark)
        # In real implementation, compare to SPY performance
        
        # 5. Institutional Accumulation Pattern
        # Look for steady volume increase with price stability
        volume_trend = np.polyfit(range(5), volumes[-5:], 1)[0]
        price_stability = np.std(closes[-5:]) / np.mean(closes[-5:]) * 100
        
        # Calculate composite score
        score = 0
        
        # Momentum score (0-35)
        if momentum_20d > 15 and momentum_10d > 8 and momentum_5d > 3:
            score += 35  # Strong multi-timeframe momentum
        elif momentum_20d > 10 and momentum_10d > 5:
            score += 25
        elif momentum_20d > 5:
            score += 15
        
        # Volume/Institutional score (0-30)
        if volume_surge > 2.5 and volume_trend > 0:
            score += 30  # Heavy institutional buying
        elif volume_surge > 1.8:
            score += 20
        elif volume_surge > 1.3:
            score += 10
        
        # MFI score (0-20)
        if 40 < mfi < 70:  # Sweet spot - strong but not overbought
            score += 20
        elif 30 < mfi < 80:
            score += 10
        
        # VWAP score (0-15)
        if price_vs_vwap > 2:  # Trading above VWAP - bullish
            score += 15
        elif price_vs_vwap > 0:
            score += 8
        
        # Determine signal
        if score >= 75:
            # Calculate targets
            atr = self._calculate_atr(highs[-14:], lows[-14:], closes[-14:])
            
            entry = current_price
            target = entry * (1 + (atr * 3 / entry))  # 3 ATR target
            stop = entry * (1 - (atr * 1.5 / entry))  # 1.5 ATR stop
            
            risk_reward = (target - entry) / (entry - stop)
            
            # Position sizing based on volatility
            position_size = max(1, min(50, int(5000 / (atr * 2))))  # Risk-based sizing
            
            return TradeSignal(
                symbol=symbol,
                side='buy',
                score=score,
                confidence=min(95, score + 10),
                strategy=StrategyType.MOMENTUM,
                entry_price=entry,
                target_price=target,
                stop_loss=stop,
                risk_reward_ratio=risk_reward,
                position_size=position_size,
                time_frame='1-5 days',
                reasoning=f"Strong momentum ({momentum_20d:.1f}% 20d), Volume surge {volume_surge:.1f}x, MFI {mfi:.0f}"
            )
        
        return None
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        """Calculate Average True Range"""
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        return np.mean(trs[-14:]) if trs else 1.0

class MeanReversionTrader(InstitutionalTrader):
    """
    Top 1% Mean Reversion Trader
    Specializes in oversold bounces with statistical edge
    Uses: Bollinger Bands, RSI, Standard Deviations, Statistical Arbitrage
    """
    
    def __init__(self, logger: logging.Logger):
        super().__init__("Reversion King", StrategyType.MEAN_REVERSION, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 30:
            return None
        
        closes = [float(bar['c']) for bar in data]
        highs = [float(bar['h']) for bar in data]
        lows = [float(bar['l']) for bar in data]
        volumes = [int(bar['v']) for bar in data]
        
        current_price = closes[-1]
        
        # 1. Bollinger Bands
        sma_20 = np.mean(closes[-20:])
        std_20 = np.std(closes[-20:])
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        
        # 2. RSI
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
        
        # 3. Statistical Analysis
        # Z-score (how many standard deviations from mean)
        z_score = (current_price - sma_20) / std_20 if std_20 > 0 else 0
        
        # 4. Rate of Change (ROC)
        roc_10 = ((closes[-1] - closes[-10]) / closes[-10]) * 100
        
        # 5. Oversold Bounce Setup
        # Look for extreme oversold + volume capitulation + RSI divergence
        volume_avg = np.mean(volumes[-20:])
        volume_recent = np.mean(volumes[-3:])
        volume_cap = volume_recent / volume_avg if volume_avg > 0 else 1
        
        # 6. Williams %R
        highest_high = max(highs[-14:])
        lowest_low = min(lows[-14:])
        williams_r = ((highest_high - current_price) / (highest_high - lowest_low)) * -100 if (highest_high - lowest_low) > 0 else -50
        
        # Calculate score (for mean reversion, we want oversold conditions)
        score = 0
        
        # Oversold conditions (0-40)
        if rsi < 25 and z_score < -2.0:
            score += 40  # Extremely oversold
        elif rsi < 35 and z_score < -1.5:
            score += 30
        elif rsi < 40 and z_score < -1.0:
            score += 20
        
        # Bollinger Band position (0-25)
        if bb_position < 0.1:  # Near bottom band
            score += 25
        elif bb_position < 0.2:
            score += 15
        elif bb_position < 0.3:
            score += 10
        
        # Capitulation volume (0-20)
        if volume_cap > 2.0:  # High volume on decline = capitulation
            score += 20
        elif volume_cap > 1.5:
            score += 12
        
        # Williams %R (0-15)
        if williams_r < -90:
            score += 15
        elif williams_r < -80:
            score += 10
        
        # Signal generation
        if score >= 70:
            atr = self._calculate_atr(highs[-14:], lows[-14:], closes[-14:])
            
            entry = current_price
            # Target is mean reversion to 20 SMA
            target = sma_20
            stop = entry - (atr * 2)  # Tight stop for mean reversion
            
            # Risk/Reward
            if target > entry and stop < entry:
                risk_reward = (target - entry) / (entry - stop)
            else:
                risk_reward = 1.5
            
            return TradeSignal(
                symbol=symbol,
                side='buy',
                score=score,
                confidence=min(90, score + 5),
                strategy=StrategyType.MEAN_REVERSION,
                entry_price=entry,
                target_price=target,
                stop_loss=stop,
                risk_reward_ratio=risk_reward,
                position_size=max(1, int(3000 / (atr * 2))),
                time_frame='2-7 days',
                reasoning=f"Oversold bounce setup: RSI {rsi:.0f}, BB% {bb_position*100:.0f}%, Z-score {z_score:.1f}"
            )
        
        # Also check for overbought (short opportunities)
        if rsi > 75 and z_score > 2.0 and bb_position > 0.9:
            # Short signal (for advanced accounts)
            pass
        
        return None
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        return np.mean(trs[-14:]) if trs else 1.0

class BreakoutTrader(InstitutionalTrader):
    """
    Top 1% Breakout Trader
    Specializes in momentum breakouts from consolidation
    Uses: Support/Resistance, Volume Confirmation, Price Patterns
    """
    
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
        current_high = highs[-1]
        current_low = lows[-1]
        
        # 1. Find key resistance levels (recent highs)
        recent_highs = sorted(highs[-20:], reverse=True)[:5]
        resistance_level = np.mean(recent_highs)
        
        # 2. Find key support levels
        recent_lows = sorted(lows[-20:])[:5]
        support_level = np.mean(recent_lows)
        
        # 3. Check for consolidation (tight range)
        consolidation_range = (max(highs[-10:]) - min(lows[-10:])) / np.mean(closes[-10:]) * 100
        
        # 4. Volume on breakout
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume
        
        # 5. Breakout confirmation
        price_vs_resistance = ((current_price - resistance_level) / resistance_level) * 100
        
        # 6. ADX for trend strength (simplified calculation)
        # In real implementation, use proper ADX
        dx_values = []
        for i in range(1, min(15, len(closes))):
            plus_dm = highs[-i] - highs[-i-1] if highs[-i] > highs[-i-1] else 0
            minus_dm = lows[-i-1] - lows[-i] if lows[-i] < lows[-i-1] else 0
            
            if plus_dm > minus_dm and plus_dm > 0:
                dx = 100
            elif minus_dm > plus_dm and minus_dm > 0:
                dx = 0
            else:
                dx = 50
            dx_values.append(dx)
        
        trend_strength = np.mean(dx_values) if dx_values else 50
        
        # Calculate score
        score = 0
        
        # Breakout quality (0-40)
        if price_vs_resistance > 2 and consolidation_range < 8:
            score += 40  # Clean breakout from tight consolidation
        elif price_vs_resistance > 1.5 and consolidation_range < 12:
            score += 30
        elif price_vs_resistance > 1:
            score += 20
        
        # Volume confirmation (0-30)
        if volume_ratio > 2.5:
            score += 30
        elif volume_ratio > 1.8:
            score += 22
        elif volume_ratio > 1.3:
            score += 12
        
        # Trend strength (0-20)
        if trend_strength > 60:
            score += 20
        elif trend_strength > 50:
            score += 15
        elif trend_strength > 40:
            score += 10
        
        # Prior consolidation quality (0-10)
        if consolidation_range < 5:  # Very tight consolidation
            score += 10
        elif consolidation_range < 8:
            score += 5
        
        # Generate signal
        if score >= 75:
            atr = self._calculate_atr(highs[-14:], lows[-14:], closes[-14:])
            
            entry = current_price
            # Target based on measured move (consolidation height projected up)
            measured_move = max(highs[-20:]) - min(lows[-20:])
            target = entry + measured_move * 0.7  # 70% of measured move
            
            # Stop below support or recent low
            stop = min(support_level, min(lows[-3:]))
            
            risk_reward = (target - entry) / (entry - stop) if (entry - stop) > 0 else 2.0
            
            return TradeSignal(
                symbol=symbol,
                side='buy',
                score=score,
                confidence=min(92, score + 8),
                strategy=StrategyType.BREAKOUT,
                entry_price=entry,
                target_price=target,
                stop_loss=stop,
                risk_reward_ratio=risk_reward,
                position_size=max(1, int(4000 / (atr * 2))),
                time_frame='1-4 weeks',
                reasoning=f"Breakout from consolidation: +{price_vs_resistance:.1f}% above resistance, {volume_ratio:.1f}x volume"
            )
        
        return None
    
    def _calculate_atr(self, highs, lows, closes):
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return np.mean(trs[-14:]) if trs else 1.0

class TrendFollowingTrader(InstitutionalTrader):
    """
    Top 1% Trend Following Trader
    Specializes in riding major trends with institutional backing
    Uses: Moving Averages, ADX, MACD, Trend Strength
    """
    
    def __init__(self, logger: logging.Logger):
        super().__init__("Trend Rider", StrategyType.TREND_FOLLOWING, logger)
    
    def analyze(self, symbol: str, data: List[Dict]) -> Optional[TradeSignal]:
        if len(data) < 50:
            return None
        
        closes = [float(bar['c']) for bar in data]
        highs = [float(bar['h']) for bar in data]
        lows = [float(bar['l']) for bar in data]
        volumes = [int(bar['v']) for bar in data]
        
        current_price = closes[-1]
        
        # Multiple timeframe moving averages
        ema_9 = self._calculate_ema(closes, 9)
        ema_21 = self._calculate_ema(closes, 21)
        ema_50 = self._calculate_ema(closes, 50)
        
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        
        # Moving Average alignment (bullish stack)
        ma_aligned = current_price > ema_9 > ema_21 > ema_50 > sma_50
        
        # MACD (simplified)
        ema_12 = self._calculate_ema(closes, 12)
        ema_26 = self._calculate_ema(closes, 26)
        macd_line = ema_12 - ema_26
        signal_line = self._calculate_ema([macd_line] * len(closes), 9)  # Simplified
        macd_histogram = macd_line - signal_line
        macd_bullish = macd_line > signal_line and macd_histogram > 0
        
        # Trend strength (using slopes)
        price_slope_10 = np.polyfit(range(10), closes[-10:], 1)[0]
        price_slope_30 = np.polyfit(range(30), closes[-30:], 1)[0]
        
        trend_acceleration = price_slope_10 > price_slope_30 > 0
        
        # Volume trend
        volume_slope = np.polyfit(range(20), volumes[-20:], 1)[0]
        volume_increasing = volume_slope > 0
        
        # Pullback to EMA (entry opportunity)
        distance_from_ema9 = ((current_price - ema_9) / ema_9) * 100
        pullback_to_ema = -2 < distance_from_ema9 < 2  # Near EMA
        
        # Higher highs and higher lows
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        hh_hl_pattern = (
            max(recent_highs[-10:]) > max(recent_highs[:10]) and
            min(recent_lows[-10:]) > min(recent_lows[:10])
        )
        
        # Calculate score
        score = 0
        
        # MA alignment (0-35)
        if ma_aligned and ema_9 > ema_21 > ema_50:
            score += 35
        elif current_price > ema_21 > ema_50:
            score += 25
        elif current_price > ema_50:
            score += 15
        
        # Trend strength (0-25)
        if trend_acceleration and price_slope_10 > 0.5:
            score += 25
        elif price_slope_10 > 0.3:
            score += 18
        elif price_slope_10 > 0.1:
            score += 10
        
        # MACD (0-20)
        if macd_bullish and macd_line > 0:
            score += 20
        elif macd_bullish:
            score += 12
        
        # Pattern (0-15)
        if hh_hl_pattern:
            score += 15
        elif current_price > max(highs[-10:-1]):
            score += 8
        
        # Entry timing (0-5 bonus for pullbacks)
        if pullback_to_ema and ma_aligned:
            score += 5
        
        # Generate signal
        if score >= 70 and ma_aligned:
            atr = self._calculate_atr(highs[-14:], lows[-14:], closes[-14:])
            
            entry = current_price
            # Trend following - wider targets
            target = entry * 1.15  # 15% target for trend
            stop = ema_21 * 0.98  # Stop below 21 EMA
            
            risk_reward = (target - entry) / (entry - stop) if (entry - stop) > 0 else 3.0
            
            return TradeSignal(
                symbol=symbol,
                side='buy',
                score=score,
                confidence=min(88, score),
                strategy=StrategyType.TREND_FOLLOWING,
                entry_price=entry,
                target_price=target,
                stop_loss=stop,
                risk_reward_ratio=risk_reward,
                position_size=max(1, int(5000 / (atr * 3))),  # Larger positions for trends
                time_frame='2-8 weeks',
                reasoning=f"Strong uptrend: EMAs aligned, MACD bullish, HH/HL pattern confirmed"
            )
        
        return None
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_atr(self, highs, lows, closes):
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return np.mean(trs[-14:]) if trs else 1.0

class InstitutionalTradingTeam:
    """
    Team of elite AI traders working together
    Like a top hedge fund with specialized PMs
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
        # Initialize specialized traders
        self.traders = {
            'momentum': MomentumTrader(logger),
            'mean_reversion': MeanReversionTrader(logger),
            'breakout': BreakoutTrader(logger),
            'trend_following': TrendFollowingTrader(logger)
        }
        
        self.consensus_threshold = 2  # Minimum number of traders agreeing
        
    def analyze_stock(self, symbol: str, data: List[Dict]) -> List[TradeSignal]:
        """
        Get analysis from all traders
        Returns signals with consensus from multiple strategies
        """
        all_signals = []
        
        for name, trader in self.traders.items():
            try:
                signal = trader.analyze(symbol, data)
                if signal:
                    all_signals.append(signal)
                    self.logger.info(
                        f"🎯 {trader.name} found signal: {symbol} "
                        f"Score: {signal.score:.0f} | "
                        f"{signal.strategy.value} | "
                        f"R/R: {signal.risk_reward_ratio:.1f}"
                    )
            except Exception as e:
                self.logger.warning(f"Error from {trader.name}: {e}")
        
        return all_signals
    
    def get_consensus_opportunities(self, symbol_data: Dict[str, List[Dict]]) -> List[TradeSignal]:
        """
        Find opportunities with multiple traders agreeing
        Higher confidence when multiple strategies align
        """
        consensus_trades = []
        
        for symbol, data in symbol_data.items():
            signals = self.analyze_stock(symbol, data)
            
            if len(signals) >= self.consensus_threshold:
                # Multiple traders agree - high confidence opportunity
                best_signal = max(signals, key=lambda x: x.score)
                
                # Boost score based on consensus
                consensus_bonus = len(signals) * 5
                best_signal.score += consensus_bonus
                best_signal.confidence = min(98, best_signal.confidence + consensus_bonus)
                best_signal.reasoning += f" [CONSENSUS: {len(signals)} traders agree]"
                
                consensus_trades.append(best_signal)
                
                self.logger.info(
                    f"🔥 CONSENSUS OPPORTUNITY: {symbol} | "
                    f"Score: {best_signal.score:.0f} | "
                    f"{len(signals)} strategies aligned"
                )
            elif signals:
                # Single strong signal
                best_signal = max(signals, key=lambda x: x.score)
                if best_signal.score >= 80:  # High conviction single signal
                    consensus_trades.append(best_signal)
        
        # Sort by score
        consensus_trades.sort(key=lambda x: x.score, reverse=True)
        
        return consensus_trades
    
    def get_team_performance(self) -> Dict:
        """Get performance of all traders"""
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

def get_trading_team(logger: logging.Logger = None) -> InstitutionalTradingTeam:
    """Get trading team instance"""
    global team_instance
    if team_instance is None:
        if logger is None:
            logger = logging.getLogger('InstitutionalTeam')
        team_instance = InstitutionalTradingTeam(logger)
    return team_instance
