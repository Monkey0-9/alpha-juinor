"""
=============================================================================
VOLATILITY-SCALED POSITION SIZING ENGINE
=============================================================================

Implements institutional-grade dynamic position sizing:
- Scales positions based on volatility (ATR)
- Scales based on signal confidence
- Respects concentration limits
- Validates against ADV (Average Daily Volume)
- Manages sector correlation risk

This replaces hardcoded position sizing with adaptive risk-adjusted sizing.
"""

import logging
from decimal import Decimal
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolatilityScaledAllocator:
    """
    Dynamic position sizing based on volatility and risk parameters.

    Formula:
    position_size = (nav * vol_adjusted_leverage * signal_confidence * sector_adjust) / symbol_count

    Where:
    - vol_adjusted_leverage = base_leverage / (1 + volatility) - reduces size when vol high
    - signal_confidence = 0.5 to 2.0x multiplier based on signal strength
    - sector_adjust = reducer if sector exposure already high
    """

    def __init__(
        self,
        nav: float,
        base_leverage: float = 3.0,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.20,
        max_correlation: float = 0.90,
    ):
        self.nav = nav
        self.base_leverage = base_leverage
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_correlation = max_correlation

        self.current_positions = {}
        self.sector_exposure = {}
        self.correlation_matrix = {}

    def calculate_dynamic_position_size(
        self,
        symbol: str,
        signal_confidence: float,  # 0.0 to 1.0
        volatility: float,  # e.g., 0.20 for 20% realized vol
        sector: str,
        correlation_to_largest: float = 0.0,
        adv: float = 1_000_000.0,
        price: float = 100.0,
    ) -> Tuple[int, str]:
        """
        Calculate dynamic position size with all risk adjustments.

        Returns:
            (quantity, reason_for_size)
        """

        # 1. VOL ADJUSTMENT: Higher vol = smaller position
        #    Normal conditions: vol=0.20 -> leverage=2.5x
        #    Stressed: vol=0.50 -> leverage=2.0x
        vol_adjusted_leverage = self.base_leverage / (1.0 + volatility)
        logger.debug(
            f"[VOL] {symbol}: vol={volatility:.2%} -> leverage={vol_adjusted_leverage:.2f}x"
        )

        # 2. SIGNAL CONFIDENCE: 0.5x to 2.0x
        #    Low confidence (0.5): size reduced by 50%
        #    High confidence (1.0): normal size
        confidence_multiplier = 0.5 + (1.5 * signal_confidence)
        logger.debug(
            f"[SIG] {symbol}: confidence={signal_confidence:.2f} -> mult={confidence_multiplier:.2f}x"
        )

        # 3. BASE position notional
        target_notional = self.nav * vol_adjusted_leverage * confidence_multiplier

        # 4. CONCENTRATION GATE: Max 5% of NAV per position
        max_notional_concentration = self.nav * self.max_position_pct
        if target_notional > max_notional_concentration:
            logger.info(
                f"[CONC] {symbol}: target ${target_notional:,.0f} > max ${max_notional_concentration:,.0f} - reducing"
            )
            target_notional = max_notional_concentration
            reason = "concentration_limit"
        else:
            reason = "normal_sizing"

        # 5. SECTOR EXPOSURE CHECK
        current_sector_exposure = self.sector_exposure.get(sector, 0)
        sector_available = self.nav * self.max_sector_pct - current_sector_exposure

        if target_notional > sector_available:
            logger.warning(
                f"[SECTOR] {symbol}: sector={sector} exposure would exceed {self.max_sector_pct:.0%}. Reducing from ${target_notional:,.0f} to ${sector_available:,.0f}"
            )
            target_notional = sector_available
            reason = "sector_limit"

        # 6. CORRELATION CHECK: If highly correlated with largest position, reduce
        if correlation_to_largest > self.max_correlation:
            correlation_reduction = (correlation_to_largest - self.max_correlation) / (
                1.0 - self.max_correlation
            )  # 0 to 1
            target_notional *= 1.0 - correlation_reduction * 0.5  # Reduce by up to 50%
            logger.info(
                f"[CORR] {symbol}: corr={correlation_to_largest:.2f} > max {self.max_correlation:.2f} - reducing size"
            )
            reason = "correlation_limit"

        # 7. ADV VALIDATION: Don't order > 10% of ADV
        max_shares_adv = int(adv * 0.10)
        max_notional_adv = max_shares_adv * price

        if target_notional > max_notional_adv:
            logger.warning(
                f"[ADV] {symbol}: order ${target_notional:,.0f} > 10% ADV ${max_notional_adv:,.0f} - rejecting"
            )
            return 0, "exceeds_adv_limit"

        # 8. FINAL QUANTITY
        quantity = int(target_notional / price)

        if quantity <= 0:
            return 0, "size_too_small"

        logger.info(
            f"[SIZE] {symbol}: qty={quantity} (${target_notional:,.0f} notional) - {reason}"
        )

        # Update tracking
        self.current_positions[symbol] = {
            "notional": target_notional,
            "quantity": quantity,
            "price": price,
            "sector": sector,
            "confidence": signal_confidence,
            "volatility": volatility,
        }

        self.sector_exposure[sector] = (
            self.sector_exposure.get(sector, 0) + target_notional
        )

        return quantity, reason

    def calculate_stop_loss_dynamic(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        volatility: float,
        trading_type: str = "swing",
    ) -> float:
        """
        Calculate dynamic stop loss based on ATR and volatility.

        Formula:
        stop_loss = entry_price - (ATR_multiple * ATR)

        ATR_multiple depends on:
        - Trading type (day=1.5x, swing=2.0x, position=3.0x)
        - Volatility (high vol = wider stops)
        """

        # Base ATR multiplier by type
        atr_multiples = {
            "day": 1.5,
            "swing": 2.0,
            "scalp": 1.0,
            "position": 3.0,
            "momentum": 1.8,
        }

        base_multiple = atr_multiples.get(trading_type, 2.0)

        # Volatility adjustment: High vol = wider stops
        vol_adjustment = 1.0 + (volatility * 2.0)  # Add up to 2x for 100% vol

        final_multiple = base_multiple * vol_adjustment
        stop_loss = entry_price - (final_multiple * atr)

        logger.info(
            f"[SL] {symbol}: SL=${stop_loss:.2f} (entry=${entry_price:.2f}, ATR={atr:.2f}, "
            f"type={trading_type}, vol={volatility:.2%})"
        )

        return stop_loss

    def calculate_take_profit_dynamic(
        self,
        entry_price: float,
        atr: float,
        volatility: float,
        trading_type: str = "swing",
    ) -> list:
        """
        Calculate dynamic take profit levels based on ATR and type.

        Returns list of take profit prices.
        """

        # Risk-reward ratios by type
        rr_ratios = {
            "day": [2.0, 3.0],  # 2:1 and 3:1
            "swing": [2.5, 4.0],  # 2.5:1 and 4:1
            "scalp": [0.5, 1.0],  # Limited TP for scalps
            "position": [3.0, 5.0],  # 3:1 and 5:1
            "momentum": [2.0, 4.0],
        }

        base_ratios = rr_ratios.get(trading_type, [2.0, 3.0])

        # Volatility impact: High vol = wider targets
        vol_adjustment = 1.0 + volatility
        tp_levels = []

        for ratio in base_ratios:
            tp_distance = ratio * atr * vol_adjustment
            tp_price = entry_price + tp_distance
            tp_levels.append(tp_price)

        logger.info(
            f"[TP] Entry=${entry_price:.2f}, ATR={atr:.2f}: TPs={[f'${tp:.2f}' for tp in tp_levels]}"
        )

        return tp_levels

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        volatility: float,
        unrealized_pnl_pct: float,
    ) -> float:
        """
        Calculate trailing stop that moves up with profits.

        Tightens as profits accumulate.
        """

        # Start with wide stop, tighten as profits grow
        if unrealized_pnl_pct < 0.05:
            # No profit yet or small loss
            return entry_price * (1.0 - 0.10)  # 10% trail
        elif unrealized_pnl_pct < 0.10:
            # 5-10% profit: trail 5%
            return current_price * (1.0 - 0.05)
        elif unrealized_pnl_pct < 0.20:
            # 10-20% profit: trail 3%
            return current_price * (1.0 - 0.03)
        else:
            # 20%+ profit: trail 2% (tight)
            return current_price * (1.0 - 0.02)

    def get_portfolio_stats(self) -> Dict:
        """Return current portfolio sizing statistics"""

        total_notional = sum(p["notional"] for p in self.current_positions.values())
        total_leverage = total_notional / self.nav

        return {
            "nav": self.nav,
            "positions_count": len(self.current_positions),
            "total_notional": total_notional,
            "gross_leverage": total_leverage,
            "max_position": max(
                (p["notional"] for p in self.current_positions.values()), default=0
            ),
            "sector_exposure": self.sector_exposure,
            "largest_sector": (
                max(self.sector_exposure.items(), key=lambda x: x[1])[0]
                if self.sector_exposure
                else None
            ),
        }


# Helper function for integrating into main.py
def calculate_position_size_for_symbol(
    symbol: str,
    nav: float,
    signal_confidence: float,
    volatility: float,
    sector: str,
    adv: float,
    current_price: float,
    allocator: VolatilityScaledAllocator,
    correlation_to_largest: float = 0.0,
) -> Tuple[int, str]:
    """
    Wrapper function for easy integration into existing code.

    Usage in main.py:
        qty, reason = calculate_position_size_for_symbol(
            symbol="AAPL",
            nav=nav,
            signal_confidence=0.85,
            volatility=0.22,
            sector="Technology",
            adv=avg_volume,
            current_price=price,
            allocator=allocator,
        )
    """

    return allocator.calculate_dynamic_position_size(
        symbol=symbol,
        signal_confidence=signal_confidence,
        volatility=volatility,
        sector=sector,
        correlation_to_largest=correlation_to_largest,
        adv=adv,
        price=current_price,
    )
