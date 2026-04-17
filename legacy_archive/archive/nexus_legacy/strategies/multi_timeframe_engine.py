"""
Multi-Timeframe Strategy Engine.

Orchestrates strategies across multiple timeframes:
- Scalping (Seconds-Minutes)
- Intraday (Minutes-Hours)
- Swing (Days-Weeks)
- Momentum (Days-Months)
- Position (Months-Years)

Implements conflict resolution when signals disagree.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Trading timeframe classifications."""
    SCALPING = "scalping"          # Seconds to Minutes
    INTRADAY = "intraday"          # Minutes to Hours
    SWING = "swing"                # Days to Weeks
    MOMENTUM = "momentum"          # Days to Months
    POSITION = "position"          # Months to Years


@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe strategy."""
    timeframe: Timeframe
    symbol: str
    direction: int  # 1=Long, -1=Short, 0=Neutral
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedSignal:
    """Combined signal from multiple timeframes."""
    symbol: str
    final_direction: int
    final_strength: float
    final_confidence: float
    contributing_timeframes: List[Timeframe]
    dominant_timeframe: Timeframe
    action: str  # BUY, SELL, HOLD, WAIT
    reasoning: List[str]


class MultiTimeframeEngine:
    """
    Central orchestrator for multi-timeframe trading.

    Decision Logic:
    1. Generate signals from each timeframe strategy.
    2. Weight signals by timeframe appropriateness (regime-based).
    3. Resolve conflicts when timeframes disagree.
    4. Output actionable aggregate signal.
    """

    # Timeframe weights (adjustable by regime)
    DEFAULT_WEIGHTS = {
        Timeframe.SCALPING: 0.05,
        Timeframe.INTRADAY: 0.15,
        Timeframe.SWING: 0.30,
        Timeframe.MOMENTUM: 0.30,
        Timeframe.POSITION: 0.20,
    }

    # Regime-based weight adjustments
    REGIME_WEIGHTS = {
        "BULL_QUIET": {
            Timeframe.MOMENTUM: 0.40,
            Timeframe.POSITION: 0.30,
            Timeframe.SWING: 0.20,
        },
        "BEAR_VOLATILE": {
            Timeframe.SCALPING: 0.15,
            Timeframe.INTRADAY: 0.30,
            Timeframe.SWING: 0.40,
        },
        "CRISIS": {
            Timeframe.SCALPING: 0.0,
            Timeframe.INTRADAY: 0.10,
            Timeframe.SWING: 0.20,
            Timeframe.POSITION: 0.70,
        },
    }

    def __init__(self, regime: str = "NEUTRAL"):
        self.regime = regime
        self.signals: Dict[str, List[TimeframeSignal]] = {}
        self.active_weights = self._get_weights_for_regime(regime)

    def _get_weights_for_regime(self, regime: str) -> Dict[Timeframe, float]:
        """Get timeframe weights based on market regime."""
        base = self.DEFAULT_WEIGHTS.copy()
        if regime in self.REGIME_WEIGHTS:
            base.update(self.REGIME_WEIGHTS[regime])

        # Normalize
        total = sum(base.values())
        return {k: v / total for k, v in base.items()}

    def update_regime(self, regime: str):
        """Update the market regime and recalculate weights."""
        self.regime = regime
        self.active_weights = self._get_weights_for_regime(regime)
        logger.info(f"MTF Engine: Regime updated to {regime}")

    def add_signal(self, signal: TimeframeSignal):
        """Add a signal from a timeframe strategy."""
        if signal.symbol not in self.signals:
            self.signals[signal.symbol] = []
        self.signals[signal.symbol].append(signal)

    def clear_signals(self, symbol: Optional[str] = None):
        """Clear signals for a symbol or all symbols."""
        if symbol:
            self.signals.pop(symbol, None)
        else:
            self.signals.clear()

    def aggregate(self, symbol: str) -> Optional[AggregatedSignal]:
        """
        Aggregate signals across timeframes for a symbol.

        Uses weighted voting with conflict resolution.
        """
        if symbol not in self.signals or not self.signals[symbol]:
            return None

        sigs = self.signals[symbol]
        reasoning = []

        # Group by direction
        long_weight = 0.0
        short_weight = 0.0
        neutral_weight = 0.0
        contributing = []

        for sig in sigs:
            w = self.active_weights.get(sig.timeframe, 0.1)
            score = w * sig.strength * sig.confidence

            if sig.direction > 0:
                long_weight += score
                contributing.append(sig.timeframe)
                reasoning.append(
                    f"{sig.timeframe.value}: LONG "
                    f"(str={sig.strength:.2f}, conf={sig.confidence:.2f})"
                )
            elif sig.direction < 0:
                short_weight += score
                contributing.append(sig.timeframe)
                reasoning.append(
                    f"{sig.timeframe.value}: SHORT "
                    f"(str={sig.strength:.2f}, conf={sig.confidence:.2f})"
                )
            else:
                neutral_weight += score
                reasoning.append(f"{sig.timeframe.value}: NEUTRAL")

        # Determine dominant direction
        total_weight = long_weight + short_weight + neutral_weight
        if total_weight == 0:
            return None

        # Conflict Resolution
        long_pct = long_weight / total_weight
        short_pct = short_weight / total_weight
        neutral_pct = neutral_weight / total_weight

        # Decision thresholds
        CONVICTION_THRESHOLD = 0.4  # Need 40% agreement minimum
        CONFLICT_THRESHOLD = 0.3   # If opposing >30%, wait

        if long_pct >= CONVICTION_THRESHOLD and short_pct < CONFLICT_THRESHOLD:
            direction = 1
            action = "BUY"
            strength = long_pct
        elif short_pct >= CONVICTION_THRESHOLD and long_pct < CONFLICT_THRESHOLD:
            direction = -1
            action = "SELL"
            strength = short_pct
        elif neutral_pct > 0.5:
            direction = 0
            action = "HOLD"
            strength = neutral_pct
        else:
            # Conflict - wait for clarity
            direction = 0
            action = "WAIT"
            strength = 0.0
            reasoning.append(
                f"CONFLICT: Long={long_pct:.0%}, Short={short_pct:.0%}"
            )

        # Find dominant timeframe
        dominant = max(
            sigs,
            key=lambda s: self.active_weights.get(s.timeframe, 0) * s.strength
        ).timeframe

        return AggregatedSignal(
            symbol=symbol,
            final_direction=direction,
            final_strength=strength,
            final_confidence=strength,
            contributing_timeframes=list(set(contributing)),
            dominant_timeframe=dominant,
            action=action,
            reasoning=reasoning
        )

    def generate_intraday_signal(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[TimeframeSignal]:
        """
        Generate intraday signal from minute/hour data.

        Strategy: VWAP reversion + volume breakout.
        """
        if data.empty or len(data) < 20:
            return None

        try:
            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values

            # VWAP Calculation
            typical_price = (high + low + close) / 3
            vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
            current_vwap = vwap[-1]
            current_price = close[-1]

            # Price vs VWAP
            vwap_deviation = (current_price - current_vwap) / current_vwap

            # Volume breakout check
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Signal logic
            direction = 0
            strength = 0.0

            if vwap_deviation < -0.02 and volume_ratio > 1.5:
                # Below VWAP with high volume = potential reversion long
                direction = 1
                strength = min(abs(vwap_deviation) * 20, 1.0)
            elif vwap_deviation > 0.02 and volume_ratio > 1.5:
                # Above VWAP with high volume = potential reversion short
                direction = -1
                strength = min(abs(vwap_deviation) * 20, 1.0)

            if direction == 0:
                return None

            return TimeframeSignal(
                timeframe=Timeframe.INTRADAY,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=min(volume_ratio / 2, 1.0),
                metadata={"vwap_dev": vwap_deviation, "vol_ratio": volume_ratio}
            )
        except Exception as e:
            logger.warning(f"Intraday signal error for {symbol}: {e}")
            return None

    def generate_swing_signal(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[TimeframeSignal]:
        """
        Generate swing signal from daily data.

        Strategy: RSI divergence + support/resistance.
        """
        if data.empty or len(data) < 50:
            return None

        try:
            close = data['close'].values

            # RSI Calculation
            delta = np.diff(close)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)

            period = 14
            avg_gain = np.convolve(
                gains, np.ones(period)/period, mode='valid'
            )[-1]
            avg_loss = np.convolve(
                losses, np.ones(period)/period, mode='valid'
            )[-1]

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # Support/Resistance (simple pivot)
            recent_low = np.min(close[-20:])
            recent_high = np.max(close[-20:])
            current = close[-1]

            # Signal logic
            direction = 0
            strength = 0.0

            if rsi < 30 and current <= recent_low * 1.02:
                # Oversold near support
                direction = 1
                strength = (30 - rsi) / 30
            elif rsi > 70 and current >= recent_high * 0.98:
                # Overbought near resistance
                direction = -1
                strength = (rsi - 70) / 30

            if direction == 0:
                return None

            return TimeframeSignal(
                timeframe=Timeframe.SWING,
                symbol=symbol,
                direction=direction,
                strength=min(strength, 1.0),
                confidence=0.7,
                stop_loss=recent_low * 0.95 if direction > 0 else recent_high * 1.05,
                take_profit=recent_high if direction > 0 else recent_low,
                metadata={"rsi": rsi}
            )
        except Exception as e:
            logger.warning(f"Swing signal error for {symbol}: {e}")
            return None

    def generate_momentum_signal(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[TimeframeSignal]:
        """
        Generate momentum signal from daily/weekly data.

        Strategy: Cross-sectional momentum (12-1 month return).
        """
        if data.empty or len(data) < 252:
            return None

        try:
            close = data['close'].values

            # 12-month return excluding last month (12-1)
            ret_12m = (close[-21] / close[-252]) - 1 if close[-252] > 0 else 0
            ret_1m = (close[-1] / close[-21]) - 1 if close[-21] > 0 else 0
            momentum = ret_12m - ret_1m

            # Recent volatility
            returns = np.diff(np.log(close[-63:]))
            volatility = np.std(returns) * np.sqrt(252)

            # Risk-adjusted momentum
            if volatility > 0:
                risk_adj_mom = momentum / volatility
            else:
                risk_adj_mom = momentum

            # Signal logic
            direction = 0
            strength = 0.0

            if risk_adj_mom > 0.5:
                direction = 1
                strength = min(risk_adj_mom / 2, 1.0)
            elif risk_adj_mom < -0.5:
                direction = -1
                strength = min(abs(risk_adj_mom) / 2, 1.0)

            if direction == 0:
                return None

            return TimeframeSignal(
                timeframe=Timeframe.MOMENTUM,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=0.65,
                metadata={"momentum": momentum, "vol": volatility}
            )
        except Exception as e:
            logger.warning(f"Momentum signal error for {symbol}: {e}")
            return None

    def generate_position_signal(
        self, symbol: str, data: pd.DataFrame,
        fundamental_score: float = 0.5
    ) -> Optional[TimeframeSignal]:
        """
        Generate position (long-term) signal.

        Strategy: Fundamental value + macro trend alignment.
        """
        if data.empty or len(data) < 252:
            return None

        try:
            close = data['close'].values

            # Long-term trend (200 SMA)
            sma_200 = np.mean(close[-200:])
            current = close[-1]
            trend = (current - sma_200) / sma_200

            # Combine with fundamental score
            combined = (trend * 0.4) + ((fundamental_score - 0.5) * 0.6)

            direction = 0
            strength = 0.0

            if combined > 0.1 and current > sma_200:
                direction = 1
                strength = min(combined * 2, 1.0)
            elif combined < -0.1 and current < sma_200:
                direction = -1
                strength = min(abs(combined) * 2, 1.0)

            if direction == 0:
                return None

            return TimeframeSignal(
                timeframe=Timeframe.POSITION,
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=0.6,
                metadata={"trend": trend, "fundamental": fundamental_score}
            )
        except Exception as e:
            logger.warning(f"Position signal error for {symbol}: {e}")
            return None

    def run_full_analysis(
        self, symbol: str,
        intraday_data: Optional[pd.DataFrame] = None,
        daily_data: Optional[pd.DataFrame] = None,
        fundamental_score: float = 0.5
    ) -> Optional[AggregatedSignal]:
        """
        Run complete multi-timeframe analysis for a symbol.
        """
        self.clear_signals(symbol)

        # Generate signals from each available timeframe
        if intraday_data is not None and not intraday_data.empty:
            sig = self.generate_intraday_signal(symbol, intraday_data)
            if sig:
                self.add_signal(sig)

        if daily_data is not None and not daily_data.empty:
            # Swing (needs ~50 days)
            if len(daily_data) >= 50:
                sig = self.generate_swing_signal(symbol, daily_data)
                if sig:
                    self.add_signal(sig)

            # Momentum (needs ~252 days)
            if len(daily_data) >= 252:
                sig = self.generate_momentum_signal(symbol, daily_data)
                if sig:
                    self.add_signal(sig)

                sig = self.generate_position_signal(
                    symbol, daily_data, fundamental_score
                )
                if sig:
                    self.add_signal(sig)

        return self.aggregate(symbol)


# Global instance
_mtf_engine: Optional[MultiTimeframeEngine] = None


def get_mtf_engine() -> MultiTimeframeEngine:
    """Get or create global MTF engine."""
    global _mtf_engine
    if _mtf_engine is None:
        _mtf_engine = MultiTimeframeEngine()
    return _mtf_engine
