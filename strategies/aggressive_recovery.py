"""
Aggressive Recovery Hunter - High Risk, High Reward
=====================================================

This module finds BEATEN DOWN stocks and buys them for
MASSIVE RETURNS when they recover.

Strategy:
1. Find stocks that have crashed/dropped significantly
2. Buy at the bottom with strict stop loss
3. Hold for big recovery
4. Take HIGH RISK but with LIMITED LOSS

Rules:
- Buy when stock is 20-50% below its high (beaten down)
- Set tight stop loss (max 5% loss per trade)
- Target 20-50%+ returns
- Take LARGER positions for bigger profits
- Risk/Reward: 1:4 to 1:10 (risk 5% to make 20-50%)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

getcontext().prec = 50


@dataclass
class RecoveryOpportunity:
    """A beaten down stock opportunity."""
    symbol: str
    timestamp: datetime

    # Current state
    current_price: Decimal
    high_52w: Decimal
    low_52w: Decimal

    # How beaten down
    drop_from_high: Decimal  # Percentage drop from high
    above_low: Decimal       # Percentage above low

    # Recovery potential
    recovery_target: Decimal
    upside_potential: Decimal  # Percentage upside

    # Risk
    stop_loss: Decimal
    max_loss_pct: Decimal

    # Risk/Reward
    risk_reward_ratio: Decimal

    # Quality score
    recovery_score: float  # 0 to 1

    # Why this opportunity
    reasons: List[str]


@dataclass
class AggressiveTrade:
    """An aggressive high-reward trade."""
    symbol: str
    timestamp: datetime

    # Trade details
    action: str  # Always BUY for recovery
    entry_price: Decimal
    position_size: Decimal  # Larger for high reward

    # Exits
    stop_loss: Decimal          # Tight to limit loss
    take_profit_1: Decimal      # First target
    take_profit_2: Decimal      # Second target
    take_profit_3: Decimal      # Maximum target

    # Risk/Reward
    risk_pct: Decimal           # Max loss %
    reward_pct: Decimal         # Expected gain %
    risk_reward: Decimal        # Must be 4:1+

    # Trade quality
    grade: str                  # A+, A, B
    confidence: float

    # Reasoning
    setup_type: str             # CRASH_RECOVERY, OVERSOLD_BOUNCE, etc
    reasons: List[str]


class AggressiveRecoveryHunter:
    """
    Finds beaten down stocks for HIGH REWARD trades.

    Strategy:
    - Find stocks that crashed 20-50%
    - Buy near the bottom
    - Tight stop loss (5% max loss)
    - Target 20-50%+ returns
    - Position size: LARGER than normal

    Risk/Reward: 1:4 to 1:10
    """

    # Aggressive thresholds
    MIN_DROP_FROM_HIGH = 0.20   # Stock must be 20%+ below high
    MAX_DROP_FROM_HIGH = 0.60   # Not more than 60% (avoid dying companies)

    # Risk limits - STRICT
    MAX_LOSS_PER_TRADE = Decimal("0.05")   # 5% max loss

    # Reward targets - HIGH
    MIN_UPSIDE_TARGET = 0.20    # 20% minimum upside
    TARGET_UPSIDE = 0.40        # 40% target

    # Position sizing - AGGRESSIVE
    BASE_POSITION_SIZE = Decimal("0.08")   # 8% base (larger than normal)
    MAX_POSITION_SIZE = Decimal("0.12")    # 12% max (aggressive)

    # Minimum risk/reward
    MIN_RISK_REWARD = 4.0       # At least 4:1

    def __init__(self):
        """Initialize the hunter."""
        self.opportunities_found = 0
        self.trades_generated = 0

        logger.info(
            "[AGGRESSIVE] Recovery Hunter initialized - "
            "HIGH RISK, HIGH REWARD MODE"
        )

    def find_opportunities(
        self,
        market_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict]] = None
    ) -> List[RecoveryOpportunity]:
        """Find beaten down stocks ready for recovery."""
        opportunities = []

        if isinstance(market_data.columns, pd.MultiIndex):
            symbols = list(market_data.columns.get_level_values(0).unique())
        else:
            symbols = []

        for symbol in symbols:
            try:
                opp = self._analyze_recovery_potential(
                    symbol, market_data, fundamentals
                )
                if opp:
                    opportunities.append(opp)
            except Exception:
                continue

        # Sort by recovery score
        opportunities.sort(key=lambda x: x.recovery_score, reverse=True)

        self.opportunities_found += len(opportunities)

        logger.info(
            f"[AGGRESSIVE] Found {len(opportunities)} recovery opportunities"
        )

        return opportunities

    def _analyze_recovery_potential(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict]]
    ) -> Optional[RecoveryOpportunity]:
        """Analyze if stock is beaten down and ready for recovery."""
        if isinstance(market_data.columns, pd.MultiIndex):
            prices = market_data[symbol]["Close"].dropna()
            volumes = market_data[symbol].get("Volume", pd.Series())
        else:
            return None

        if len(prices) < 50:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        # 52-week high/low (or max available)
        lookback = min(252, len(p))
        high_52w = Decimal(str(np.max(p[-lookback:])))
        low_52w = Decimal(str(np.min(p[-lookback:])))

        # How much has it dropped from high?
        drop_from_high = (high_52w - current) / high_52w

        # How far above low?
        if high_52w > low_52w:
            above_low = (current - low_52w) / (high_52w - low_52w)
        else:
            above_low = Decimal("0.5")

        # Check if beaten down enough
        if float(drop_from_high) < self.MIN_DROP_FROM_HIGH:
            return None  # Not beaten down enough

        if float(drop_from_high) > self.MAX_DROP_FROM_HIGH:
            return None  # Too risky (dying company)

        reasons = []
        score = 0.0

        # Score based on drop
        if float(drop_from_high) >= 0.40:
            score += 0.35
            reasons.append(f"Crashed {float(drop_from_high):.0%} from high")
        elif float(drop_from_high) >= 0.30:
            score += 0.25
            reasons.append(f"Dropped {float(drop_from_high):.0%} from high")
        else:
            score += 0.15
            reasons.append(f"Down {float(drop_from_high):.0%} from high")

        # Score based on position in range
        if float(above_low) < 0.20:
            score += 0.25
            reasons.append("Near 52-week low")
        elif float(above_low) < 0.35:
            score += 0.15
            reasons.append("In lower range")

        # RSI - oversold is good
        if len(p) >= 14:
            delta = np.diff(p[-15:])
            gains = np.maximum(delta, 0)
            losses = np.maximum(-delta, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses) + 0.0001
            rsi = 100 - 100 / (1 + avg_gain / avg_loss)

            if rsi < 30:
                score += 0.20
                reasons.append(f"Oversold RSI={rsi:.0f}")
            elif rsi < 40:
                score += 0.10
                reasons.append(f"Low RSI={rsi:.0f}")

        # Volume spike - could indicate capitulation
        if len(volumes) >= 20 and len(volumes.dropna()) >= 20:
            v = volumes.dropna().values
            vol_ratio = v[-1] / np.mean(v[-20:])
            if vol_ratio > 2.0:
                score += 0.15
                reasons.append("Volume spike (possible capitulation)")

        # Fundamentals check (company not dying)
        if fundamentals and symbol in fundamentals:
            fund = fundamentals[symbol]

            # Check it's not a dying company
            if fund.get("revenue_growth", 0) > -0.30:
                score += 0.05

            if fund.get("debt_to_equity", 999) < 2.0:
                score += 0.05
                reasons.append("Manageable debt")

        if score < 0.30:
            return None

        # Calculate recovery target and stop loss
        # Target: 40% of the way back to high
        recovery_target = current + (high_52w - current) * Decimal("0.40")
        upside = (recovery_target - current) / current

        # Stop loss: 5% below entry
        stop_loss = current * Decimal("0.95")
        max_loss = Decimal("0.05")

        # Risk/Reward
        risk_reward = upside / max_loss if max_loss > 0 else Decimal("0")

        # Must meet minimum risk/reward
        if float(risk_reward) < self.MIN_RISK_REWARD:
            return None

        return RecoveryOpportunity(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            current_price=current.quantize(Decimal("0.01")),
            high_52w=high_52w.quantize(Decimal("0.01")),
            low_52w=low_52w.quantize(Decimal("0.01")),
            drop_from_high=(drop_from_high * 100).quantize(Decimal("0.1")),
            above_low=(above_low * 100).quantize(Decimal("0.1")),
            recovery_target=recovery_target.quantize(Decimal("0.01")),
            upside_potential=(upside * 100).quantize(Decimal("0.1")),
            stop_loss=stop_loss.quantize(Decimal("0.01")),
            max_loss_pct=max_loss * 100,
            risk_reward_ratio=risk_reward.quantize(Decimal("0.1")),
            recovery_score=min(1.0, score),
            reasons=reasons
        )

    def generate_trade(
        self,
        opportunity: RecoveryOpportunity,
        portfolio_value: float = 100000
    ) -> AggressiveTrade:
        """Generate aggressive trade from opportunity."""
        entry = opportunity.current_price
        stop = opportunity.stop_loss

        # Multiple take profit levels
        t1 = entry * Decimal("1.15")  # 15% - first target
        t2 = opportunity.recovery_target  # Main target
        t3 = entry + (opportunity.high_52w - entry) * Decimal("0.60")
        # 60% back to high

        # Risk/Reward
        risk_pct = Decimal("5.0")  # 5% max loss
        reward_pct = opportunity.upside_potential  # From opportunity
        rr = reward_pct / risk_pct

        # Position size - AGGRESSIVE
        # Higher score = larger position
        score = opportunity.recovery_score

        if score >= 0.7:
            pos_size = self.MAX_POSITION_SIZE
            grade = "A+"
        elif score >= 0.5:
            pos_size = self.BASE_POSITION_SIZE
            grade = "A"
        else:
            pos_size = self.BASE_POSITION_SIZE * Decimal("0.8")
            grade = "B"

        # Adjust for risk/reward
        if float(rr) >= 8:
            pos_size *= Decimal("1.2")  # More size for better R/R

        pos_size = min(pos_size, self.MAX_POSITION_SIZE)

        self.trades_generated += 1

        return AggressiveTrade(
            symbol=opportunity.symbol,
            timestamp=datetime.utcnow(),
            action="BUY",
            entry_price=entry,
            position_size=pos_size.quantize(Decimal("0.001")),
            stop_loss=stop,
            take_profit_1=t1.quantize(Decimal("0.01")),
            take_profit_2=t2.quantize(Decimal("0.01")),
            take_profit_3=t3.quantize(Decimal("0.01")),
            risk_pct=risk_pct,
            reward_pct=reward_pct,
            risk_reward=rr.quantize(Decimal("0.1")),
            grade=grade,
            confidence=opportunity.recovery_score,
            setup_type="CRASH_RECOVERY",
            reasons=opportunity.reasons
        )

    def hunt(
        self,
        market_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict]] = None,
        top_n: int = 5
    ) -> List[AggressiveTrade]:
        """
        Find and generate aggressive trades.

        This is the main method that:
        1. Finds all beaten down stocks
        2. Scores them
        3. Generates trades for top opportunities
        """
        # Find opportunities
        opportunities = self.find_opportunities(market_data, fundamentals)

        # Generate trades for top ones
        trades = []
        for opp in opportunities[:top_n]:
            trade = self.generate_trade(opp)
            trades.append(trade)

            logger.info(
                f"[AGGRESSIVE] {trade.symbol}: BUY @ {trade.entry_price} | "
                f"Stop={trade.stop_loss} | Target={trade.take_profit_2} | "
                f"R/R={trade.risk_reward} | Grade={trade.grade}"
            )

        return trades

    def get_stats(self) -> Dict[str, Any]:
        """Get hunter statistics."""
        return {
            "opportunities_found": self.opportunities_found,
            "trades_generated": self.trades_generated
        }


class HighRiskHighRewardEngine:
    """
    Complete engine for high-risk, high-reward trading.

    Combines:
    1. Recovery Hunter - buy crashed stocks
    2. Momentum Reversal - catch the bounce
    3. Breakout Hunter - ride explosive moves

    Risk Rules:
    - MAX 5% loss per trade (STRICT)
    - Target 20-50%+ returns
    - R/R must be 4:1 or better
    - Larger positions for higher rewards
    """

    def __init__(self):
        """Initialize the engine."""
        self.recovery_hunter = AggressiveRecoveryHunter()

        self.total_trades = 0
        self.winning_trades = 0

        logger.info(
            "[HRHR] High Risk High Reward Engine initialized - "
            "MAXIMUM PROFIT MODE"
        )

    def find_best_trades(
        self,
        market_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict]] = None,
        top_n: int = 5
    ) -> List[AggressiveTrade]:
        """Find the best high-reward trades."""
        all_trades = []

        # 1. Recovery trades (buy crashed stocks)
        recovery_trades = self.recovery_hunter.hunt(
            market_data, fundamentals, top_n
        )
        all_trades.extend(recovery_trades)

        # 2. Also look for oversold bounces
        bounce_trades = self._find_bounce_trades(market_data)
        all_trades.extend(bounce_trades)

        # Sort by reward potential
        all_trades.sort(
            key=lambda t: float(t.reward_pct) * t.confidence,
            reverse=True
        )

        self.total_trades += len(all_trades[:top_n])

        return all_trades[:top_n]

    def _find_bounce_trades(
        self,
        market_data: pd.DataFrame
    ) -> List[AggressiveTrade]:
        """Find oversold bounce opportunities."""
        trades = []

        if isinstance(market_data.columns, pd.MultiIndex):
            symbols = list(market_data.columns.get_level_values(0).unique())
        else:
            return trades

        for symbol in symbols[:50]:
            try:
                prices = market_data[symbol]["Close"].dropna()

                if len(prices) < 20:
                    continue

                p = prices.values
                current = Decimal(str(p[-1]))

                # RSI
                delta = np.diff(p[-15:])
                gains = np.maximum(delta, 0)
                losses = np.maximum(-delta, 0)
                rsi = 100 - 100 / (
                    1 + np.mean(gains) / (np.mean(losses) + 0.0001)
                )

                # Bollinger position
                sma = np.mean(p[-20:])
                std = np.std(p[-20:])
                bb_lower = sma - 2 * std

                # Extreme oversold + below Bollinger
                if rsi < 25 and p[-1] < bb_lower:
                    # This is a bounce setup
                    stop = current * Decimal("0.95")
                    target = current * Decimal("1.25")  # 25% target

                    trade = AggressiveTrade(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        action="BUY",
                        entry_price=current.quantize(Decimal("0.01")),
                        position_size=Decimal("0.08"),
                        stop_loss=stop.quantize(Decimal("0.01")),
                        take_profit_1=(
                            current * Decimal("1.10")
                        ).quantize(Decimal("0.01")),
                        take_profit_2=(
                            current * Decimal("1.20")
                        ).quantize(Decimal("0.01")),
                        take_profit_3=target.quantize(Decimal("0.01")),
                        risk_pct=Decimal("5.0"),
                        reward_pct=Decimal("25.0"),
                        risk_reward=Decimal("5.0"),
                        grade="A",
                        confidence=0.70,
                        setup_type="OVERSOLD_BOUNCE",
                        reasons=[
                            f"Over RSI={rsi:.0f}", "Below Bollinger"
                        ]
                    )
                    trades.append(trade)

            except Exception:
                continue

        return trades[:3]


# Singletons
_hunter: Optional[AggressiveRecoveryHunter] = None
_engine: Optional[HighRiskHighRewardEngine] = None


def get_recovery_hunter() -> AggressiveRecoveryHunter:
    """Get or create the Recovery Hunter."""
    global _hunter
    if _hunter is None:
        _hunter = AggressiveRecoveryHunter()
    return _hunter


def get_hrhr_engine() -> HighRiskHighRewardEngine:
    """Get or create the High Risk High Reward Engine."""
    global _engine
    if _engine is None:
        _engine = HighRiskHighRewardEngine()
    return _engine
