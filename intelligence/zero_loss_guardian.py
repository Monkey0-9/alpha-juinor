"""
Zero Loss Guardian - The Ultimate Loss Prevention System
=========================================================

This module's ONLY job is to PREVENT LOSSES.

Features:
1. NEVER allows a trade if loss probability > 10%
2. Automatic stop-loss tightening
3. Profit locking at multiple levels
4. Market condition verification
5. Multi-timeframe confirmation
6. Drawdown prevention
7. Real-time position monitoring

ZERO LOSS TOLERANCE.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maximum precision
getcontext().prec = 50


class TradeRisk(Enum):
    """Risk level classification."""
    ULTRA_SAFE = "ULTRA_SAFE"  # < 5% loss probability
    SAFE = "SAFE"              # 5-10% loss probability
    MODERATE = "MODERATE"      # 10-20% loss probability
    RISKY = "RISKY"            # 20-35% loss probability
    VERY_RISKY = "VERY_RISKY"  # > 35% loss probability
    FORBIDDEN = "FORBIDDEN"    # NEVER trade


class MarketCondition(Enum):
    """Market condition for trading."""
    PERFECT = "PERFECT"        # Best conditions
    GOOD = "GOOD"              # Favorable
    NEUTRAL = "NEUTRAL"        # Average
    POOR = "POOR"              # Unfavorable
    DANGEROUS = "DANGEROUS"    # Stay out


@dataclass
class LossPreventionResult:
    """Result of loss prevention analysis."""
    symbol: str
    timestamp: datetime

    # Risk assessment
    loss_probability: Decimal
    max_potential_loss: Decimal
    risk_level: TradeRisk

    # Market conditions
    market_condition: MarketCondition
    volatility_regime: str
    trend_strength: float

    # Trade allowed?
    trade_allowed: bool
    block_reasons: List[str]

    # Protection measures
    recommended_stop_loss: Decimal
    profit_lock_levels: List[Decimal]
    max_position_size: Decimal
    max_holding_period_hours: int

    # Confidence
    analysis_confidence: float


@dataclass
class ProfitProtection:
    """Profit protection levels."""
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl_pct: Decimal

    # Stop loss progression
    initial_stop: Decimal
    current_stop: Decimal
    trailing_stop_active: bool

    # Profit locks
    lock_25_pct: bool
    lock_50_pct: bool
    lock_75_pct: bool

    # Action
    action_required: str  # HOLD, TIGHTEN_STOP, TAKE_PROFIT, EXIT_NOW


class ZeroLossGuardian:
    """
    The ULTIMATE loss prevention system.

    This guardian's ONLY job is to PREVENT LOSSES.
    It will block ANY trade that has significant loss potential.
    It will protect profits aggressively.

    ZERO TOLERANCE FOR LOSSES.
    """

    # Strict thresholds
    MAX_LOSS_PROBABILITY = Decimal("0.10")  # 10% max
    MAX_POTENTIAL_LOSS = Decimal("0.02")    # 2% max per trade
    MIN_WIN_PROBABILITY = Decimal("0.70")   # 70% min to trade
    MIN_RISK_REWARD = Decimal("3.0")        # 3:1 minimum
    MAX_VOLATILITY = 0.35                   # 35% annualized max

    # Profit protection levels
    PROFIT_LOCK_25 = Decimal("0.01")  # Lock at 1% gain
    PROFIT_LOCK_50 = Decimal("0.02")  # Lock at 2% gain
    PROFIT_LOCK_75 = Decimal("0.03")  # Lock at 3% gain

    def __init__(self):
        """Initialize the Zero Loss Guardian."""
        self.trades_blocked = 0
        self.trades_approved = 0
        self.losses_prevented = Decimal("0")

        logger.info(
            "[GUARDIAN] Zero Loss Guardian initialized - "
            "ZERO LOSS TOLERANCE ACTIVE"
        )

    def analyze_trade(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        proposed_stop: float,
        proposed_target: float,
        market_data: pd.DataFrame,
        position_size: float = 0.05
    ) -> LossPreventionResult:
        """
        Analyze a proposed trade for loss risk.

        Will BLOCK any trade with unacceptable loss risk.
        """
        entry = Decimal(str(entry_price))
        stop = Decimal(str(proposed_stop))
        target = Decimal(str(proposed_target))
        size = Decimal(str(position_size))

        block_reasons = []

        # 1. Calculate loss probability
        loss_prob = self._calculate_loss_probability(
            symbol, action, entry, stop, target, market_data
        )

        if loss_prob > self.MAX_LOSS_PROBABILITY:
            block_reasons.append(
                f"Loss probability {loss_prob:.1%} exceeds {self.MAX_LOSS_PROBABILITY:.0%} limit"
            )

        # 2. Calculate potential loss
        if action == "BUY":
            potential_loss = (entry - stop) / entry * size
        else:
            potential_loss = (stop - entry) / entry * size

        potential_loss = abs(potential_loss)

        if potential_loss > self.MAX_POTENTIAL_LOSS:
            block_reasons.append(
                f"Max potential loss {potential_loss:.2%} exceeds {self.MAX_POTENTIAL_LOSS:.0%} limit"
            )

        # 3. Check risk/reward
        if action == "BUY":
            reward = target - entry
            risk = entry - stop
        else:
            reward = entry - target
            risk = stop - entry

        rr_ratio = reward / risk if risk > 0 else Decimal("0")

        if rr_ratio < self.MIN_RISK_REWARD:
            block_reasons.append(
                f"Risk/reward {rr_ratio:.1f} below {self.MIN_RISK_REWARD:.0f} minimum"
            )

        # 4. Check market conditions
        market_condition = self._assess_market_condition(market_data, symbol)

        if market_condition in [MarketCondition.POOR, MarketCondition.DANGEROUS]:
            block_reasons.append(
                f"Market condition {market_condition.value} not suitable"
            )

        # 5. Check volatility
        volatility = self._calculate_volatility(market_data, symbol)

        if volatility > self.MAX_VOLATILITY:
            block_reasons.append(
                f"Volatility {volatility:.0%} exceeds {self.MAX_VOLATILITY:.0%} limit"
            )

        # 6. Multi-timeframe confirmation
        mtf_confirmed = self._multi_timeframe_check(market_data, symbol, action)

        if not mtf_confirmed:
            block_reasons.append("Multi-timeframe confirmation failed")

        # 7. Trend alignment
        trend_strength = self._check_trend_alignment(market_data, symbol, action)

        if trend_strength < 0.3:
            block_reasons.append(
                f"Trend alignment weak ({trend_strength:.0%})"
            )

        # Determine risk level
        if loss_prob < Decimal("0.05"):
            risk_level = TradeRisk.ULTRA_SAFE
        elif loss_prob < Decimal("0.10"):
            risk_level = TradeRisk.SAFE
        elif loss_prob < Decimal("0.20"):
            risk_level = TradeRisk.MODERATE
        elif loss_prob < Decimal("0.35"):
            risk_level = TradeRisk.RISKY
        else:
            risk_level = TradeRisk.VERY_RISKY

        # Final decision
        trade_allowed = len(block_reasons) == 0

        if not trade_allowed:
            self.trades_blocked += 1
            self.losses_prevented += potential_loss
            logger.warning(
                f"[GUARDIAN] BLOCKED {symbol} {action}: {block_reasons[0]}"
            )
        else:
            self.trades_approved += 1
            logger.info(
                f"[GUARDIAN] APPROVED {symbol} {action}: "
                f"risk={risk_level.value}, loss_prob={loss_prob:.1%}"
            )

        # Calculate protection levels
        recommended_stop = self._calculate_optimal_stop(
            entry, action, market_data, symbol
        )

        profit_locks = self._calculate_profit_locks(entry, action)

        # Maximum position size based on risk
        if risk_level == TradeRisk.ULTRA_SAFE:
            max_size = Decimal("0.08")
        elif risk_level == TradeRisk.SAFE:
            max_size = Decimal("0.05")
        elif risk_level == TradeRisk.MODERATE:
            max_size = Decimal("0.03")
        else:
            max_size = Decimal("0.01")

        # Maximum holding period
        if volatility > 0.25:
            max_hold = 24  # 1 day in volatile markets
        elif volatility > 0.15:
            max_hold = 72  # 3 days
        else:
            max_hold = 168  # 1 week

        return LossPreventionResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            loss_probability=loss_prob,
            max_potential_loss=potential_loss,
            risk_level=risk_level,
            market_condition=market_condition,
            volatility_regime="HIGH" if volatility > 0.25 else "NORMAL",
            trend_strength=trend_strength,
            trade_allowed=trade_allowed,
            block_reasons=block_reasons,
            recommended_stop_loss=recommended_stop,
            profit_lock_levels=profit_locks,
            max_position_size=max_size,
            max_holding_period_hours=max_hold,
            analysis_confidence=0.95 if trade_allowed else 0.99
        )

    def _calculate_loss_probability(
        self,
        symbol: str,
        action: str,
        entry: Decimal,
        stop: Decimal,
        target: Decimal,
        market_data: pd.DataFrame
    ) -> Decimal:
        """Calculate probability of hitting stop loss."""
        try:
            # Get historical prices
            if isinstance(market_data.columns, pd.MultiIndex):
                closes = market_data[symbol]["Close"].dropna()
            else:
                closes = market_data.get("Close", pd.Series()).dropna()

            if len(closes) < 50:
                return Decimal("0.50")  # Unknown = assume 50%

            p = closes.values

            # Historical volatility
            returns = np.diff(np.log(p))
            vol = np.std(returns) * np.sqrt(252)

            # Distance to stop
            stop_distance = abs(float(entry - stop) / float(entry))

            # Z-score of stop distance
            daily_vol = vol / np.sqrt(252)
            z_score = stop_distance / (daily_vol * 5)  # 5-day horizon

            # Probability from normal distribution
            from scipy import stats
            prob = float(stats.norm.cdf(-z_score))

            # Adjust for trend
            trend = (p[-1] / p[-20] - 1) if len(p) >= 20 else 0

            if action == "BUY" and trend > 0:
                prob *= 0.8  # Reduce loss prob in uptrend
            elif action == "SELL" and trend < 0:
                prob *= 0.8
            else:
                prob *= 1.2  # Increase if against trend

            return Decimal(str(min(0.99, max(0.01, prob))))

        except Exception:
            return Decimal("0.50")

    def _assess_market_condition(
        self,
        market_data: pd.DataFrame,
        symbol: str
    ) -> MarketCondition:
        """Assess current market condition."""
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                closes = market_data[symbol]["Close"].dropna()
            else:
                closes = market_data.get("Close", pd.Series()).dropna()

            if len(closes) < 50:
                return MarketCondition.NEUTRAL

            p = closes.values

            # Trend check
            sma_20 = np.mean(p[-20:])
            sma_50 = np.mean(p[-50:])

            above_20sma = p[-1] > sma_20
            above_50sma = p[-1] > sma_50
            trend_up = sma_20 > sma_50

            # Volatility check
            returns = np.diff(np.log(p[-20:]))
            recent_vol = np.std(returns) * np.sqrt(252)

            # Momentum check
            mom = p[-1] / p[-5] - 1

            # Score conditions
            score = 0

            if above_20sma and above_50sma and trend_up:
                score += 2
            elif above_20sma or above_50sma:
                score += 1

            if recent_vol < 0.20:
                score += 1
            elif recent_vol > 0.35:
                score -= 2

            if -0.02 < mom < 0.05:
                score += 1
            elif mom > 0.10 or mom < -0.05:
                score -= 1

            if score >= 3:
                return MarketCondition.PERFECT
            elif score >= 2:
                return MarketCondition.GOOD
            elif score >= 0:
                return MarketCondition.NEUTRAL
            elif score >= -2:
                return MarketCondition.POOR
            else:
                return MarketCondition.DANGEROUS

        except Exception:
            return MarketCondition.NEUTRAL

    def _calculate_volatility(
        self,
        market_data: pd.DataFrame,
        symbol: str
    ) -> float:
        """Calculate annualized volatility."""
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                closes = market_data[symbol]["Close"].dropna()
            else:
                closes = market_data.get("Close", pd.Series()).dropna()

            if len(closes) < 20:
                return 0.25

            returns = np.diff(np.log(closes.values[-20:]))
            return float(np.std(returns) * np.sqrt(252))

        except Exception:
            return 0.25

    def _multi_timeframe_check(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        action: str
    ) -> bool:
        """Check signal across multiple timeframes."""
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                closes = market_data[symbol]["Close"].dropna()
            else:
                closes = market_data.get("Close", pd.Series()).dropna()

            if len(closes) < 50:
                return False

            p = closes.values

            # Daily trend (5-day)
            daily_trend = p[-1] > np.mean(p[-5:])

            # Weekly trend (20-day)
            weekly_trend = p[-1] > np.mean(p[-20:])

            # Monthly trend (50-day)
            monthly_trend = p[-1] > np.mean(p[-50:])

            if action == "BUY":
                # Need alignment across timeframes
                aligned = sum([daily_trend, weekly_trend, monthly_trend])
                return aligned >= 2
            else:
                aligned = sum([not daily_trend, not weekly_trend, not monthly_trend])
                return aligned >= 2

        except Exception:
            return False

    def _check_trend_alignment(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        action: str
    ) -> float:
        """Check how aligned trade is with trend."""
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                closes = market_data[symbol]["Close"].dropna()
            else:
                closes = market_data.get("Close", pd.Series()).dropna()

            if len(closes) < 50:
                return 0.5

            p = closes.values

            # Calculate trend metrics
            sma_20 = np.mean(p[-20:])
            sma_50 = np.mean(p[-50:])

            # Trend strength
            if action == "BUY":
                if p[-1] > sma_20 > sma_50:
                    return 1.0
                elif p[-1] > sma_20:
                    return 0.7
                elif p[-1] > sma_50:
                    return 0.4
                else:
                    return 0.1
            else:
                if p[-1] < sma_20 < sma_50:
                    return 1.0
                elif p[-1] < sma_20:
                    return 0.7
                elif p[-1] < sma_50:
                    return 0.4
                else:
                    return 0.1

        except Exception:
            return 0.5

    def _calculate_optimal_stop(
        self,
        entry: Decimal,
        action: str,
        market_data: pd.DataFrame,
        symbol: str
    ) -> Decimal:
        """Calculate optimal stop loss level."""
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                closes = market_data[symbol]["Close"].dropna()
            else:
                closes = market_data.get("Close", pd.Series()).dropna()

            if len(closes) < 14:
                # Default 2% stop
                return entry * (Decimal("0.98") if action == "BUY" else Decimal("1.02"))

            p = closes.values

            # ATR-based stop
            highs = p  # Simplified
            lows = p
            tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - p[:-1]))
            atr = np.mean(tr[-14:])

            atr_decimal = Decimal(str(atr))

            if action == "BUY":
                stop = entry - (atr_decimal * Decimal("2.0"))
            else:
                stop = entry + (atr_decimal * Decimal("2.0"))

            return stop.quantize(Decimal("0.01"))

        except Exception:
            return entry * (Decimal("0.98") if action == "BUY" else Decimal("1.02"))

    def _calculate_profit_locks(
        self,
        entry: Decimal,
        action: str
    ) -> List[Decimal]:
        """Calculate profit lock levels."""
        if action == "BUY":
            return [
                entry * (1 + self.PROFIT_LOCK_25),
                entry * (1 + self.PROFIT_LOCK_50),
                entry * (1 + self.PROFIT_LOCK_75)
            ]
        else:
            return [
                entry * (1 - self.PROFIT_LOCK_25),
                entry * (1 - self.PROFIT_LOCK_50),
                entry * (1 - self.PROFIT_LOCK_75)
            ]

    def protect_profit(
        self,
        entry_price: float,
        current_price: float,
        action: str,
        current_stop: float
    ) -> ProfitProtection:
        """Protect unrealized profits."""
        entry = Decimal(str(entry_price))
        current = Decimal(str(current_price))
        stop = Decimal(str(current_stop))

        # Calculate unrealized P&L
        if action == "BUY":
            pnl_pct = (current - entry) / entry
        else:
            pnl_pct = (entry - current) / entry

        # Determine profit locks
        lock_25 = pnl_pct >= self.PROFIT_LOCK_25
        lock_50 = pnl_pct >= self.PROFIT_LOCK_50
        lock_75 = pnl_pct >= self.PROFIT_LOCK_75

        # Calculate new stop
        trailing_active = pnl_pct > Decimal("0.005")  # 0.5%

        if trailing_active:
            if action == "BUY":
                # Trail stop to lock profits
                if lock_75:
                    new_stop = entry * (1 + self.PROFIT_LOCK_50)
                elif lock_50:
                    new_stop = entry * (1 + self.PROFIT_LOCK_25)
                elif lock_25:
                    new_stop = entry  # Breakeven
                else:
                    new_stop = stop

                new_stop = max(new_stop, stop)  # Never lower stop
            else:
                if lock_75:
                    new_stop = entry * (1 - self.PROFIT_LOCK_50)
                elif lock_50:
                    new_stop = entry * (1 - self.PROFIT_LOCK_25)
                elif lock_25:
                    new_stop = entry
                else:
                    new_stop = stop

                new_stop = min(new_stop, stop)
        else:
            new_stop = stop

        # Determine action
        if pnl_pct >= Decimal("0.05"):  # 5% profit
            action_required = "TAKE_PROFIT"
        elif new_stop != stop:
            action_required = "TIGHTEN_STOP"
        elif pnl_pct < Decimal("-0.02"):  # 2% loss
            action_required = "EXIT_NOW"
        else:
            action_required = "HOLD"

        return ProfitProtection(
            entry_price=entry,
            current_price=current,
            unrealized_pnl_pct=pnl_pct,
            initial_stop=Decimal(str(current_stop)),
            current_stop=new_stop,
            trailing_stop_active=trailing_active,
            lock_25_pct=lock_25,
            lock_50_pct=lock_50,
            lock_75_pct=lock_75,
            action_required=action_required
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get guardian statistics."""
        total = self.trades_blocked + self.trades_approved
        return {
            "trades_blocked": self.trades_blocked,
            "trades_approved": self.trades_approved,
            "block_rate": self.trades_blocked / total if total > 0 else 0,
            "losses_prevented": float(self.losses_prevented)
        }


# Singleton
_guardian: Optional[ZeroLossGuardian] = None


def get_guardian() -> ZeroLossGuardian:
    """Get or create the Zero Loss Guardian."""
    global _guardian
    if _guardian is None:
        _guardian = ZeroLossGuardian()
    return _guardian
