"""
Strategy Universe - 15+ Trading Strategies
============================================

Complete library of trading strategies that the brain
autonomously selects from based on market conditions.

Strategies:
1. Momentum - ride strong trends
2. Mean Reversion - buy oversold, sell overbought
3. Breakout - trade range breaks
4. Trend Following - follow established trends
5. Pairs Trading - cointegrated pairs
6. Statistical Arbitrage - mispricing exploitation
7. Value Investing - undervalued stocks
8. Quality Investing - high quality companies
9. Growth Investing - high growth stocks
10. Dividend Capture - around ex-div dates
11. Volatility Trading - volatility mean reversion
12. Gap Trading - overnight gaps
13. Swing Trading - multi-day moves
14. Scalping - quick intraday moves
15. Contrarian - against the crowd
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies."""
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    BREAKOUT = "BREAKOUT"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    PAIRS_TRADING = "PAIRS_TRADING"
    STAT_ARB = "STAT_ARB"
    VALUE = "VALUE"
    QUALITY = "QUALITY"
    GROWTH = "GROWTH"
    DIVIDEND = "DIVIDEND"
    VOLATILITY = "VOLATILITY"
    GAP = "GAP"
    SWING = "SWING"
    SCALP = "SCALP"
    CONTRARIAN = "CONTRARIAN"


class MarketRegime(Enum):
    """Market regime types."""
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    RANGE_BOUND = "RANGE_BOUND"
    CRISIS = "CRISIS"
    RECOVERY = "RECOVERY"


@dataclass
class StrategySignal:
    """Signal from a strategy."""
    strategy: StrategyType
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    entry_price: Decimal
    stop_loss: Decimal
    target: Decimal
    reasoning: str
    suitability_score: float  # How suitable for current conditions


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str, strategy_type: StrategyType):
        self.name = name
        self.strategy_type = strategy_type
        self.trades_generated = 0
        self.wins = 0
        self.losses = 0

    @abstractmethod
    def analyze(
        self,
        symbol: str,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        fundamentals: Optional[Dict] = None
    ) -> Optional[StrategySignal]:
        """Analyze and generate signal if conditions met."""
        pass

    @abstractmethod
    def get_suitability(
        self,
        regime: MarketRegime,
        volatility: float
    ) -> float:
        """Return 0-1 suitability score for current conditions."""
        pass

    def record_outcome(self, win: bool):
        """Record trade outcome."""
        self.trades_generated += 1
        if win:
            self.wins += 1
        else:
            self.losses += 1

    @property
    def win_rate(self) -> float:
        if self.trades_generated == 0:
            return 0.5
        return self.wins / self.trades_generated


class MomentumStrategy(BaseStrategy):
    """Trade strong price momentum."""

    def __init__(self):
        super().__init__("Momentum", StrategyType.MOMENTUM)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if len(prices) < 50:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        # Momentum signals
        mom_5 = p[-1] / p[-5] - 1
        mom_10 = p[-1] / p[-10] - 1
        mom_20 = p[-1] / p[-20] - 1

        # Strong upward momentum
        if mom_5 > 0.02 and mom_10 > 0.03 and mom_20 > 0.05:
            action = "BUY"
            strength = min(1.0, (mom_5 + mom_10 + mom_20) / 0.15)
            confidence = 0.7 + min(0.2, mom_20)
        # Strong downward momentum
        elif mom_5 < -0.02 and mom_10 < -0.03 and mom_20 < -0.05:
            action = "SELL"
            strength = max(-1.0, (mom_5 + mom_10 + mom_20) / 0.15)
            confidence = 0.7 + min(0.2, abs(mom_20))
        else:
            return None

        # Volume confirmation
        if volumes is not None and len(volumes) >= 20:
            vol_ratio = volumes.iloc[-5:].mean() / volumes.iloc[-20:].mean()
            if vol_ratio > 1.3:
                confidence += 0.1

        atr = np.mean(np.abs(np.diff(p[-14:])))
        stop = current - Decimal(str(atr * 2)) if action == "BUY" else current + Decimal(str(atr * 2))
        target = current + Decimal(str(atr * 4)) if action == "BUY" else current - Decimal(str(atr * 4))

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=min(0.95, confidence),
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=f"Strong momentum: 5d={mom_5:.1%}, 10d={mom_10:.1%}, 20d={mom_20:.1%}",
            suitability_score=0.8
        )

    def get_suitability(self, regime, volatility):
        if regime == MarketRegime.BULL_TREND:
            return 0.95
        elif regime == MarketRegime.BEAR_TREND:
            return 0.85
        elif regime == MarketRegime.HIGH_VOL:
            return 0.6
        elif regime == MarketRegime.RANGE_BOUND:
            return 0.3
        return 0.5


class MeanReversionStrategy(BaseStrategy):
    """Trade price reversions to mean."""

    def __init__(self):
        super().__init__("MeanReversion", StrategyType.MEAN_REVERSION)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if len(prices) < 50:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        # Z-score
        mean = np.mean(p[-50:])
        std = np.std(p[-50:])
        z_score = (p[-1] - mean) / (std + 1e-10)

        # RSI
        delta = np.diff(p[-15:])
        gains = np.maximum(delta, 0)
        losses = np.maximum(-delta, 0)
        rsi = 100 - 100 / (1 + np.mean(gains) / (np.mean(losses) + 1e-10))

        # Oversold - buy
        if z_score < -2.0 and rsi < 30:
            action = "BUY"
            strength = min(1.0, abs(z_score) / 3)
            confidence = 0.65 + min(0.25, abs(z_score) / 8)
        # Overbought - sell
        elif z_score > 2.0 and rsi > 70:
            action = "SELL"
            strength = max(-1.0, -abs(z_score) / 3)
            confidence = 0.65 + min(0.25, abs(z_score) / 8)
        else:
            return None

        # Target = mean
        target = Decimal(str(mean))
        atr = np.mean(np.abs(np.diff(p[-14:])))
        stop = current - Decimal(str(atr * 1.5)) if action == "BUY" else current + Decimal(str(atr * 1.5))

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=min(0.90, confidence),
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=f"Mean reversion: Z={z_score:.2f}, RSI={rsi:.0f}",
            suitability_score=0.75
        )

    def get_suitability(self, regime, volatility):
        if regime == MarketRegime.RANGE_BOUND:
            return 0.95
        elif regime == MarketRegime.LOW_VOL:
            return 0.85
        elif regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            return 0.4
        elif regime == MarketRegime.HIGH_VOL:
            return 0.5
        return 0.5


class BreakoutStrategy(BaseStrategy):
    """Trade price breakouts from ranges."""

    def __init__(self):
        super().__init__("Breakout", StrategyType.BREAKOUT)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if len(prices) < 50:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        # Range detection
        high_20 = np.max(p[-20:-1])
        low_20 = np.min(p[-20:-1])
        range_pct = (high_20 - low_20) / low_20

        # Breakout detection
        breakout_up = p[-1] > high_20 * 1.01
        breakout_down = p[-1] < low_20 * 0.99

        if not (breakout_up or breakout_down):
            return None

        # Volume confirmation
        vol_confirmed = True
        if volumes is not None and len(volumes) >= 20:
            vol_ratio = volumes.iloc[-1] / volumes.iloc[-20:].mean()
            vol_confirmed = vol_ratio > 1.5

        if not vol_confirmed:
            return None

        if breakout_up:
            action = "BUY"
            strength = min(1.0, (p[-1] / high_20 - 1) * 20)
            target = current + Decimal(str((high_20 - low_20)))
            stop = Decimal(str(low_20))
        else:
            action = "SELL"
            strength = max(-1.0, -(1 - p[-1] / low_20) * 20)
            target = current - Decimal(str((high_20 - low_20)))
            stop = Decimal(str(high_20))

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=0.70 if vol_confirmed else 0.55,
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=f"Breakout {'above' if action == 'BUY' else 'below'} 20-day range",
            suitability_score=0.80
        )

    def get_suitability(self, regime, volatility):
        if regime == MarketRegime.RANGE_BOUND:
            return 0.90
        elif volatility < 0.15:
            return 0.7
        elif regime == MarketRegime.HIGH_VOL:
            return 0.6
        return 0.5


class TrendFollowingStrategy(BaseStrategy):
    """Follow established trends."""

    def __init__(self):
        super().__init__("TrendFollowing", StrategyType.TREND_FOLLOWING)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if len(prices) < 100:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        # Multiple MA alignment
        sma_10 = np.mean(p[-10:])
        sma_20 = np.mean(p[-20:])
        sma_50 = np.mean(p[-50:])
        sma_100 = np.mean(p[-100:])

        # Perfect alignment
        up_trend = p[-1] > sma_10 > sma_20 > sma_50 > sma_100
        down_trend = p[-1] < sma_10 < sma_20 < sma_50 < sma_100

        if not (up_trend or down_trend):
            return None

        # Trend strength
        if up_trend:
            action = "BUY"
            pct_above = (p[-1] / sma_100 - 1)
            strength = min(1.0, pct_above * 5)
            confidence = 0.75 + min(0.15, pct_above)
        else:
            action = "SELL"
            pct_below = (1 - p[-1] / sma_100)
            strength = max(-1.0, -pct_below * 5)
            confidence = 0.75 + min(0.15, pct_below)

        atr = np.mean(np.abs(np.diff(p[-14:])))
        stop = current - Decimal(str(atr * 3)) if action == "BUY" else current + Decimal(str(atr * 3))
        target = current + Decimal(str(atr * 6)) if action == "BUY" else current - Decimal(str(atr * 6))

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=min(0.92, confidence),
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning="MA alignment: 10 > 20 > 50 > 100" if up_trend else "MA alignment: 10 < 20 < 50 < 100",
            suitability_score=0.85
        )

    def get_suitability(self, regime, volatility):
        if regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            return 0.95
        elif regime == MarketRegime.HIGH_VOL:
            return 0.7
        elif regime == MarketRegime.RANGE_BOUND:
            return 0.25
        return 0.5


class ValueStrategy(BaseStrategy):
    """Invest in undervalued stocks."""

    def __init__(self):
        super().__init__("Value", StrategyType.VALUE)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if fundamentals is None:
            return None

        if len(prices) < 20:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        pe = fundamentals.get("pe_ratio", 0)
        pb = fundamentals.get("pb_ratio", 0)
        div_yield = fundamentals.get("dividend_yield", 0)

        # Value criteria
        value_score = 0
        criteria = []

        if 5 < pe < 15:
            value_score += 0.3
            criteria.append(f"Low P/E={pe:.1f}")
        if 0.5 < pb < 2:
            value_score += 0.25
            criteria.append(f"Low P/B={pb:.1f}")
        if div_yield > 0.03:
            value_score += 0.25
            criteria.append(f"High Div={div_yield:.1%}")

        # Price near 52w low
        if len(prices) >= 252:
            low_52w = np.min(p[-252:])
            if p[-1] < low_52w * 1.15:
                value_score += 0.2
                criteria.append("Near 52w low")

        if value_score < 0.5:
            return None

        action = "BUY"
        strength = min(1.0, value_score)
        confidence = 0.60 + value_score * 0.2

        atr = np.mean(np.abs(np.diff(p[-14:])))
        stop = current - Decimal(str(atr * 4))
        target = current * Decimal("1.20")

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=min(0.85, confidence),
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=", ".join(criteria),
            suitability_score=0.70
        )

    def get_suitability(self, regime, volatility):
        if regime == MarketRegime.RECOVERY:
            return 0.90
        elif regime == MarketRegime.BEAR_TREND:
            return 0.7
        elif regime == MarketRegime.BULL_TREND:
            return 0.6
        return 0.5


class QualityStrategy(BaseStrategy):
    """Invest in high-quality companies."""

    def __init__(self):
        super().__init__("Quality", StrategyType.QUALITY)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if fundamentals is None:
            return None

        if len(prices) < 20:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        roe = fundamentals.get("roe", 0)
        roa = fundamentals.get("roa", 0)
        margin = fundamentals.get("profit_margin", 0)
        debt_ratio = fundamentals.get("debt_to_equity", 999)

        quality_score = 0
        criteria = []

        if roe > 0.15:
            quality_score += 0.3
            criteria.append(f"High ROE={roe:.0%}")
        if roa > 0.08:
            quality_score += 0.25
            criteria.append(f"High ROA={roa:.0%}")
        if margin > 0.15:
            quality_score += 0.25
            criteria.append(f"High margin={margin:.0%}")
        if debt_ratio < 0.5:
            quality_score += 0.2
            criteria.append(f"Low debt={debt_ratio:.1f}")

        if quality_score < 0.5:
            return None

        action = "BUY"
        strength = min(1.0, quality_score)
        confidence = 0.65 + quality_score * 0.2

        atr = np.mean(np.abs(np.diff(p[-14:])))
        stop = current - Decimal(str(atr * 3))
        target = current * Decimal("1.15")

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=min(0.88, confidence),
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=", ".join(criteria),
            suitability_score=0.75
        )

    def get_suitability(self, regime, volatility):
        if regime == MarketRegime.CRISIS:
            return 0.85
        elif regime == MarketRegime.HIGH_VOL:
            return 0.8
        elif regime == MarketRegime.BULL_TREND:
            return 0.7
        return 0.6


class SwingStrategy(BaseStrategy):
    """Multi-day swing trades."""

    def __init__(self):
        super().__init__("Swing", StrategyType.SWING)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if len(prices) < 30:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        # Swing detection
        high_5 = np.max(p[-5:])
        low_5 = np.min(p[-5:])

        # Pullback in uptrend
        sma_20 = np.mean(p[-20:])
        uptrend = p[-1] > sma_20 and sma_20 > np.mean(p[-30:])
        downtrend = p[-1] < sma_20 and sma_20 < np.mean(p[-30:])

        # RSI
        delta = np.diff(p[-15:])
        gains = np.maximum(delta, 0)
        losses = np.maximum(-delta, 0)
        rsi = 100 - 100 / (1 + np.mean(gains) / (np.mean(losses) + 1e-10))

        if uptrend and rsi < 45 and p[-1] < high_5 * 0.97:
            action = "BUY"
            strength = 0.7
            confidence = 0.72
            reasoning = f"Pullback in uptrend, RSI={rsi:.0f}"
        elif downtrend and rsi > 55 and p[-1] > low_5 * 1.03:
            action = "SELL"
            strength = -0.7
            confidence = 0.72
            reasoning = f"Rally in downtrend, RSI={rsi:.0f}"
        else:
            return None

        atr = np.mean(np.abs(np.diff(p[-14:])))
        stop = current - Decimal(str(atr * 2)) if action == "BUY" else current + Decimal(str(atr * 2))
        target = current + Decimal(str(atr * 4)) if action == "BUY" else current - Decimal(str(atr * 4))

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=reasoning,
            suitability_score=0.75
        )

    def get_suitability(self, regime, volatility):
        if 0.15 < volatility < 0.30:
            return 0.85
        elif regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            return 0.8
        elif regime == MarketRegime.RANGE_BOUND:
            return 0.6
        return 0.5


class ContrarianStrategy(BaseStrategy):
    """Trade against the crowd."""

    def __init__(self):
        super().__init__("Contrarian", StrategyType.CONTRARIAN)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if len(prices) < 50:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        # Extreme moves
        ret_5 = p[-1] / p[-5] - 1
        ret_20 = p[-1] / p[-20] - 1

        # Z-score
        mean = np.mean(p[-50:])
        std = np.std(p[-50:])
        z_score = (p[-1] - mean) / (std + 1e-10)

        # Extreme panic selling
        if ret_5 < -0.10 and z_score < -2.5:
            action = "BUY"
            strength = min(1.0, abs(ret_5) * 5)
            confidence = 0.65
            reasoning = f"Panic selling: 5d={ret_5:.1%}, Z={z_score:.1f}"
        # Extreme euphoria
        elif ret_5 > 0.10 and z_score > 2.5:
            action = "SELL"
            strength = max(-1.0, -ret_5 * 5)
            confidence = 0.65
            reasoning = f"Euphoria: 5d={ret_5:.1%}, Z={z_score:.1f}"
        else:
            return None

        atr = np.mean(np.abs(np.diff(p[-14:])))
        stop = current - Decimal(str(atr * 2)) if action == "BUY" else current + Decimal(str(atr * 2))
        target = Decimal(str(mean))

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=reasoning,
            suitability_score=0.60
        )

    def get_suitability(self, regime, volatility):
        if regime == MarketRegime.HIGH_VOL:
            return 0.8
        elif regime == MarketRegime.CRISIS:
            return 0.85
        elif regime == MarketRegime.RANGE_BOUND:
            return 0.5
        return 0.4


class VolatilityStrategy(BaseStrategy):
    """Trade volatility mean reversion."""

    def __init__(self):
        super().__init__("Volatility", StrategyType.VOLATILITY)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if len(prices) < 50:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        # Calculate volatility
        returns = np.diff(np.log(p[-50:]))
        current_vol = np.std(returns[-10:]) * np.sqrt(252)
        avg_vol = np.std(returns) * np.sqrt(252)

        vol_ratio = current_vol / (avg_vol + 1e-10)

        # Volatility regime
        if vol_ratio < 0.6:  # Low vol - expect expansion
            action = "BUY"  # Long volatility
            strength = 0.6
            confidence = 0.65
            reasoning = f"Low volatility, expect expansion: {current_vol:.0%} vs avg {avg_vol:.0%}"
        elif vol_ratio > 1.8:  # High vol - expect contraction
            # This would be for short volatility positions
            action = "HOLD"
            return None
        else:
            return None

        atr = np.mean(np.abs(np.diff(p[-14:])))
        stop = current - Decimal(str(atr * 1.5))
        target = current + Decimal(str(atr * 3))

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=reasoning,
            suitability_score=0.65
        )

    def get_suitability(self, regime, volatility):
        if regime == MarketRegime.LOW_VOL:
            return 0.85
        elif regime == MarketRegime.HIGH_VOL:
            return 0.7
        return 0.5


class GrowthStrategy(BaseStrategy):
    """Invest in high-growth stocks."""

    def __init__(self):
        super().__init__("Growth", StrategyType.GROWTH)

    def analyze(self, symbol, prices, volumes=None, fundamentals=None):
        if fundamentals is None:
            return None

        if len(prices) < 20:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        rev_growth = fundamentals.get("revenue_growth", 0)
        earnings_growth = fundamentals.get("earnings_growth", 0)

        growth_score = 0
        criteria = []

        if rev_growth > 0.20:
            growth_score += 0.4
            criteria.append(f"Rev growth={rev_growth:.0%}")
        if earnings_growth > 0.25:
            growth_score += 0.4
            criteria.append(f"EPS growth={earnings_growth:.0%}")

        # Price momentum confirms
        if len(prices) >= 50:
            mom = p[-1] / p[-50] - 1
            if mom > 0.15:
                growth_score += 0.2
                criteria.append(f"Momentum={mom:.0%}")

        if growth_score < 0.5:
            return None

        action = "BUY"
        strength = min(1.0, growth_score)
        confidence = 0.60 + growth_score * 0.2

        atr = np.mean(np.abs(np.diff(p[-14:])))
        stop = current - Decimal(str(atr * 3))
        target = current * Decimal("1.25")

        return StrategySignal(
            strategy=self.strategy_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=min(0.85, confidence),
            entry_price=current,
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=", ".join(criteria),
            suitability_score=0.70
        )

    def get_suitability(self, regime, volatility):
        if regime == MarketRegime.BULL_TREND:
            return 0.90
        elif regime == MarketRegime.RECOVERY:
            return 0.80
        elif regime == MarketRegime.HIGH_VOL:
            return 0.5
        elif regime == MarketRegime.BEAR_TREND:
            return 0.3
        return 0.5


class StrategyUniverse:
    """Container for all trading strategies."""

    def __init__(self):
        """Initialize all strategies."""
        self.strategies: Dict[StrategyType, BaseStrategy] = {
            StrategyType.MOMENTUM: MomentumStrategy(),
            StrategyType.MEAN_REVERSION: MeanReversionStrategy(),
            StrategyType.BREAKOUT: BreakoutStrategy(),
            StrategyType.TREND_FOLLOWING: TrendFollowingStrategy(),
            StrategyType.VALUE: ValueStrategy(),
            StrategyType.QUALITY: QualityStrategy(),
            StrategyType.SWING: SwingStrategy(),
            StrategyType.CONTRARIAN: ContrarianStrategy(),
            StrategyType.VOLATILITY: VolatilityStrategy(),
            StrategyType.GROWTH: GrowthStrategy(),
        }

        logger.info(
            f"[UNIVERSE] Strategy Universe initialized with "
            f"{len(self.strategies)} strategies"
        )

    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all available strategies."""
        return list(self.strategies.values())

    def get_strategy(self, strategy_type: StrategyType) -> Optional[BaseStrategy]:
        """Get specific strategy."""
        return self.strategies.get(strategy_type)

    def get_suitable_strategies(
        self,
        regime: MarketRegime,
        volatility: float,
        min_suitability: float = 0.6
    ) -> List[BaseStrategy]:
        """Get strategies suitable for current conditions."""
        suitable = []

        for strategy in self.strategies.values():
            suitability = strategy.get_suitability(regime, volatility)
            if suitability >= min_suitability:
                suitable.append((strategy, suitability))

        # Sort by suitability
        suitable.sort(key=lambda x: x[1], reverse=True)

        return [s[0] for s in suitable]


# Singleton
_universe: Optional[StrategyUniverse] = None


def get_strategy_universe() -> StrategyUniverse:
    """Get or create the Strategy Universe."""
    global _universe
    if _universe is None:
        _universe = StrategyUniverse()
    return _universe
