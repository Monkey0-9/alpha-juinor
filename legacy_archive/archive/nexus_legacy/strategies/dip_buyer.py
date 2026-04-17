"""
Smart Dip Buyer - Buy the Dip for Profit
==========================================

Buys stocks when they DIP but with strict rules:
1. Only buy quality dips (not dying stocks)
2. Strict stop loss (max 5% loss)
3. Target big recovery bounces
4. Sell when in profit

This is the "Buy Low, Sell High" strategy executed precisely.
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
class DipSignal:
    """A dip buying signal."""
    symbol: str
    timestamp: datetime

    # Dip details
    current_price: Decimal
    dip_from_recent: Decimal  # % dip
    dip_type: str             # FLASH_CRASH, GRADUAL_DIP, OVERSOLD

    # Quality
    dip_quality: str          # A, B, C
    is_quality_stock: bool    # Good fundamentals

    # Entry
    entry_price: Decimal
    stop_loss: Decimal

    # Targets
    target_1: Decimal         # Quick profit
    target_2: Decimal         # Main target
    target_3: Decimal         # Extended target

    # Risk/Reward
    risk_pct: Decimal
    reward_pct: Decimal
    risk_reward: Decimal

    # Position
    position_size: Decimal

    # Reasoning
    reasons: List[str]


class SmartDipBuyer:
    """
    Buys quality dips for high profits.

    Rules:
    1. Stock has dipped 5-20% from recent high
    2. Fundamentals are still good (not dying)
    3. Entry at support level
    4. Max 5% loss per trade
    5. Target 15-40% profit
    """

    # Dip thresholds
    MIN_DIP = 0.05           # At least 5% dip
    MAX_DIP = 0.25           # Not more than 25% (might be dead)
    FLASH_CRASH_DIP = 0.08   # 8%+ in 1 day = flash crash

    # Risk management
    MAX_LOSS = Decimal("0.05")    # 5% max loss

    # Position sizing
    BASE_POSITION = Decimal("0.06")   # 6% base
    MAX_POSITION = Decimal("0.10")    # 10% max

    def __init__(self):
        """Initialize the dip buyer."""
        self.dips_found = 0
        self.trades_made = 0

        logger.info(
            "[DIPBUYER] Smart Dip Buyer initialized - "
            "BUY LOW SELL HIGH"
        )

    def find_dips(
        self,
        market_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict]] = None
    ) -> List[DipSignal]:
        """Find quality dip opportunities."""
        signals = []

        if isinstance(market_data.columns, pd.MultiIndex):
            symbols = list(market_data.columns.get_level_values(0).unique())
        else:
            return signals

        for symbol in symbols:
            try:
                signal = self._analyze_dip(symbol, market_data, fundamentals)
                if signal:
                    signals.append(signal)
            except Exception:
                continue

        # Sort by quality
        quality_order = {"A": 0, "B": 1, "C": 2}
        signals.sort(key=lambda x: (quality_order.get(x.dip_quality, 3), -float(x.reward_pct)))

        self.dips_found += len(signals)

        return signals

    def _analyze_dip(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict]]
    ) -> Optional[DipSignal]:
        """Analyze if there's a quality dip to buy."""
        prices = market_data[symbol]["Close"].dropna()

        if len(prices) < 30:
            return None

        p = prices.values
        current = Decimal(str(p[-1]))

        # Calculate dip from various timeframes
        high_20d = np.max(p[-20:])
        high_10d = np.max(p[-10:])
        high_5d = np.max(p[-5:])

        dip_20d = (high_20d - p[-1]) / high_20d
        dip_10d = (high_10d - p[-1]) / high_10d
        dip_5d = (high_5d - p[-1]) / high_5d
        dip_1d = (p[-2] - p[-1]) / p[-2] if len(p) >= 2 else 0

        # Determine dip type
        if dip_1d >= self.FLASH_CRASH_DIP:
            dip_type = "FLASH_CRASH"
            dip_pct = dip_1d
        elif dip_5d >= self.MIN_DIP:
            dip_type = "QUICK_DIP"
            dip_pct = dip_5d
        elif dip_20d >= self.MIN_DIP:
            dip_type = "GRADUAL_DIP"
            dip_pct = dip_20d
        else:
            return None  # No significant dip

        # Check dip is not too severe
        if dip_pct > self.MAX_DIP:
            return None  # Too risky

        reasons = []
        score = 0.0

        # Flash crash is best opportunity
        if dip_type == "FLASH_CRASH":
            score += 0.35
            reasons.append(f"Flash crash: -{dip_1d:.1%} today")
        elif dip_type == "QUICK_DIP":
            score += 0.25
            reasons.append(f"Quick dip: -{dip_5d:.1%} in 5 days")
        else:
            score += 0.15
            reasons.append(f"Dip: -{dip_20d:.1%} from high")

        # RSI check
        if len(p) >= 14:
            delta = np.diff(p[-15:])
            gains = np.maximum(delta, 0)
            losses = np.maximum(-delta, 0)
            rsi = 100 - 100 / (1 + np.mean(gains) / (np.mean(losses) + 0.0001))

            if rsi < 30:
                score += 0.25
                reasons.append(f"Oversold RSI={rsi:.0f}")
            elif rsi < 40:
                score += 0.15
                reasons.append(f"Low RSI={rsi:.0f}")

        # Check if at support
        support = np.min(p[-30:])
        if p[-1] <= support * 1.03:
            score += 0.15
            reasons.append("At support level")

        # Volume spike (panic selling = opportunity)
        volumes = market_data[symbol].get("Volume", pd.Series()).dropna()
        if len(volumes) >= 20:
            v = volumes.values
            if v[-1] > np.mean(v[-20:]) * 1.5:
                score += 0.10
                reasons.append("Volume spike (panic)")

        # Fundamental check
        is_quality = False
        if fundamentals and symbol in fundamentals:
            fund = fundamentals[symbol]

            # Quality checks
            if fund.get("revenue_growth", -1) > -0.10:
                score += 0.05
                is_quality = True

            if fund.get("debt_to_equity", 999) < 1.5:
                score += 0.05
                is_quality = True
                reasons.append("Low debt")

            if fund.get("profit_margin", 0) > 0.05:
                is_quality = True

        if score < 0.30:
            return None

        # Determine quality grade
        if score >= 0.60:
            quality = "A"
        elif score >= 0.40:
            quality = "B"
        else:
            quality = "C"

        # Calculate trade parameters
        stop = current * (1 - self.MAX_LOSS)

        # Target based on dip recovery
        recovery_factor = Decimal(str(dip_pct))
        t1 = current * Decimal("1.10")  # 10% - quick profit
        t2 = current * (1 + recovery_factor * Decimal("0.7"))  # 70% of dip recovery
        t3 = current * (1 + recovery_factor)  # Full recovery

        risk_pct = self.MAX_LOSS * 100
        reward_pct = ((t2 - current) / current) * 100
        rr = reward_pct / risk_pct if risk_pct > 0 else Decimal("0")

        # Position size based on quality
        if quality == "A":
            pos = self.MAX_POSITION
        elif quality == "B":
            pos = self.BASE_POSITION
        else:
            pos = self.BASE_POSITION * Decimal("0.7")

        return DipSignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            current_price=current.quantize(Decimal("0.01")),
            dip_from_recent=(Decimal(str(dip_pct)) * 100).quantize(Decimal("0.1")),
            dip_type=dip_type,
            dip_quality=quality,
            is_quality_stock=is_quality,
            entry_price=current.quantize(Decimal("0.01")),
            stop_loss=stop.quantize(Decimal("0.01")),
            target_1=t1.quantize(Decimal("0.01")),
            target_2=t2.quantize(Decimal("0.01")),
            target_3=t3.quantize(Decimal("0.01")),
            risk_pct=risk_pct.quantize(Decimal("0.1")),
            reward_pct=reward_pct.quantize(Decimal("0.1")),
            risk_reward=rr.quantize(Decimal("0.1")),
            position_size=pos.quantize(Decimal("0.001")),
            reasons=reasons
        )

    def buy_the_dip(
        self,
        market_data: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict]] = None,
        top_n: int = 5
    ) -> List[DipSignal]:
        """Find and return best dip opportunities."""
        dips = self.find_dips(market_data, fundamentals)

        # Only return A and B quality dips
        quality_dips = [d for d in dips if d.dip_quality in ["A", "B"]]

        for dip in quality_dips[:top_n]:
            logger.info(
                f"[DIPBUYER] {dip.symbol}: {dip.dip_type} | "
                f"Dip={float(dip.dip_from_recent):.1f}% | "
                f"Target={dip.target_2} | R/R={dip.risk_reward}"
            )

        self.trades_made += len(quality_dips[:top_n])

        return quality_dips[:top_n]


class MomentumReversalTrader:
    """
    Catches momentum reversals for quick profits.

    When stocks drop sharply but momentum is shifting:
    - Enter on reversal confirmation
    - Tight stop at recent low
    - Quick profit targets
    """

    def __init__(self):
        """Initialize the trader."""
        self.reversals_found = 0

        logger.info(
            "[REVERSAL] Momentum Reversal Trader initialized"
        )

    def find_reversals(
        self,
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Find momentum reversal opportunities."""
        reversals = []

        if isinstance(market_data.columns, pd.MultiIndex):
            symbols = list(market_data.columns.get_level_values(0).unique())
        else:
            return reversals

        for symbol in symbols[:50]:
            try:
                prices = market_data[symbol]["Close"].dropna()

                if len(prices) < 20:
                    continue

                p = prices.values

                # Conditions for reversal:
                # 1. Price dropped recently
                # 2. RSI is low but turning up
                # 3. Price closed above previous close (first green)

                drop_5d = (np.max(p[-5:-1]) - p[-1]) / np.max(p[-5:-1])

                if drop_5d < 0.05:  # Need 5%+ drop
                    continue

                # RSI
                delta = np.diff(p[-15:])
                rsi = 100 - 100 / (1 + np.mean(np.maximum(delta, 0)) / (np.mean(np.maximum(-delta, 0)) + 0.0001))

                if rsi > 45:  # Not oversold enough
                    continue

                # Green candle (price up from yesterday)
                if p[-1] <= p[-2]:
                    continue

                # This is a potential reversal
                current = Decimal(str(p[-1]))
                stop = Decimal(str(min(p[-5:])))  # Recent low
                target = current * Decimal("1.15")

                risk = current - stop
                reward = target - current
                rr = reward / risk if risk > 0 else 0

                if float(rr) < 3:  # Need good R/R
                    continue

                reversals.append({
                    "symbol": symbol,
                    "entry": float(current),
                    "stop": float(stop),
                    "target": float(target),
                    "rsi": rsi,
                    "drop_5d": drop_5d,
                    "risk_reward": float(rr),
                    "type": "MOMENTUM_REVERSAL"
                })

            except Exception:
                continue

        self.reversals_found += len(reversals)

        return reversals


# Singletons
_dip_buyer: Optional[SmartDipBuyer] = None
_reversal_trader: Optional[MomentumReversalTrader] = None


def get_dip_buyer() -> SmartDipBuyer:
    """Get or create the Smart Dip Buyer."""
    global _dip_buyer
    if _dip_buyer is None:
        _dip_buyer = SmartDipBuyer()
    return _dip_buyer


def get_reversal_trader() -> MomentumReversalTrader:
    """Get or create the Reversal Trader."""
    global _reversal_trader
    if _reversal_trader is None:
        _reversal_trader = MomentumReversalTrader()
    return _reversal_trader
