"""
=============================================================================
ENHANCED 7-GATE RISK MANAGEMENT SYSTEM
=============================================================================

Implements 7 institutional-grade risk gates for trade validation:

1. LIQUIDITY GATE - Reject if order > 10% of ADV
2. VOLATILITY GATE - Reject if VIX > 50 or realized_vol > 3 sigma
3. CONCENTRATION GATE - Reject if position > 5% of NAV
4. LEVERAGE GATE - Reject if gross leverage > 3x
5. SECTOR GATE - Reject if sector exposure > 20%
6. CORRELATION GATE - Reject if position correlation > 0.9 with largest
7. MARKET HOURS GATE - Strict market hours (fail-safe = FALSE)

Each gate can either REJECT or SCALE the order down.
"""

import logging
from datetime import datetime, time
from typing import Dict, Optional, Tuple

import numpy as np
import pytz

logger = logging.getLogger(__name__)


class SevenGateRiskManager:
    """
    7-gate risk management system for institutional trading.

    Each gate is independent and can reject or scale orders.
    """

    def __init__(
        self,
        nav: float,
        max_gross_leverage: float = 3.0,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.20,
        max_correlation: float = 0.90,
        adv_limit_pct: float = 0.10,
        max_vix: float = 50.0,
        max_realized_vol: float = 0.50,
        test_mode: bool = False,
    ):
        self.nav = nav
        self.max_gross_leverage = max_gross_leverage
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_correlation = max_correlation
        self.adv_limit_pct = adv_limit_pct
        self.max_vix = max_vix
        self.max_realized_vol = max_realized_vol
        self.test_mode = test_mode

        # Track current state
        self.current_positions = {}
        self.sector_exposure = {}

    def validate_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,  # "BUY" or "SELL"
        sector: str,
        adv: float,
        vix: float = 20.0,
        realized_volatility: float = 0.20,
        correlation_largest: float = 0.0,
    ) -> Tuple[bool, str, int]:
        """
        Validate order through all 7 gates.

        Returns:
            (is_approved, reason_code, final_quantity)

        Gates applied in priority:
        1. Market hours (hard reject)
        2. Liquidity (hard reject or scale)
        3. Volatility (hard reject)
        4. Concentration (scale)
        5. Leverage (scale)
        6. Sector (scale)
        7. Correlation (scale)
        """

        notional = abs(quantity * price)
        final_quantity = quantity

        # =====================================================================
        # GATE 1: MARKET HOURS (Hard Reject - Fail Safe = FALSE)
        # =====================================================================
        if not self._gate_market_hours():
            if not self.test_mode:
                logger.critical(
                    f"[GATE-1] MARKET HOURS: Market is CLOSED - REJECTING {symbol}"
                )
                return False, "GATE_1_MARKET_CLOSED", 0
            else:
                logger.debug(
                    f"[GATE-1] MARKET HOURS: Market is CLOSED - but TEST_MODE enabled, bypassing"
                )

        # =====================================================================
        # GATE 2: LIQUIDITY (ADV > 10%)
        # =====================================================================
        max_notional_adv = adv * self.adv_limit_pct * price
        if notional > max_notional_adv:
            logger.warning(
                f"[GATE-2] LIQUIDITY: {symbol} order ${notional:,.0f} > 10% ADV ${max_notional_adv:,.0f}"
            )

            # For highly illiquid symbols, reject entirely
            if max_notional_adv < 100_000:  # Less than $100k ADV
                logger.critical(
                    f"[GATE-2] LIQUIDITY: {symbol} too illiquid - REJECTING"
                )
                return False, "GATE_2_ILLIQUID", 0

            # Otherwise scale down
            final_quantity = int((max_notional_adv * 0.9) / price)
            logger.info(
                f"[GATE-2] Scaled {symbol} from {quantity} to {final_quantity} shares"
            )

        # =====================================================================
        # GATE 3: VOLATILITY (VIX > 50 or realized_vol > 50%)
        # =====================================================================
        if vix > self.max_vix:
            logger.critical(
                f"[GATE-3] VOLATILITY: VIX={vix:.0f} > {self.max_vix:.0f} - REJECTING all trades"
            )
            return False, f"GATE_3_VIX_TOO_HIGH_{vix:.0f}", 0

        if realized_volatility > self.max_realized_vol:
            logger.critical(
                f"[GATE-3] VOLATILITY: Realized_vol={realized_volatility:.2%} > max {self.max_realized_vol:.2%} - REJECTING"
            )
            return False, f"GATE_3_REALIZED_VOL_TOO_HIGH_{realized_volatility:.2%}", 0

        # =====================================================================
        # GATE 4: CONCENTRATION (Max 5% of NAV per position)
        # =====================================================================
        max_notional_concentration = self.nav * self.max_position_pct
        if notional > max_notional_concentration:
            logger.warning(
                f"[GATE-4] CONCENTRATION: {symbol} ${notional:,.0f} > 5% NAV limit ${max_notional_concentration:,.0f}"
            )
            final_quantity = int(max_notional_concentration * 0.95 / price)
            logger.info(
                f"[GATE-4] Scaled {symbol} to {final_quantity} shares for concentration"
            )

        # =====================================================================
        # GATE 5: LEVERAGE (Max 3x gross leverage)
        # =====================================================================
        current_gross_notional = sum(
            abs(p["notional"]) for p in self.current_positions.values()
        )
        proposed_gross_notional = current_gross_notional + notional
        proposed_gross_leverage = proposed_gross_notional / self.nav

        if proposed_gross_leverage > self.max_gross_leverage:
            logger.warning(
                f"[GATE-5] LEVERAGE: Would be {proposed_gross_leverage:.2f}x > {self.max_gross_leverage:.2f}x max"
            )

            # Calculate how much we can add
            available_notional = (
                self.nav * self.max_gross_leverage - current_gross_notional
            )

            if available_notional < 100_000:  # Less than $100k available
                logger.critical(
                    f"[GATE-5] LEVERAGE: Out of leverage capacity - REJECTING"
                )
                return False, "GATE_5_LEVERAGE_EXHAUSTED", 0

            final_quantity = int(available_notional * 0.95 / price)
            logger.info(
                f"[GATE-5] Scaled {symbol} to {final_quantity} due to leverage limit"
            )

        # =====================================================================
        # GATE 6: SECTOR EXPOSURE (Max 20% per sector)
        # =====================================================================
        current_sector_exposure = self.sector_exposure.get(sector, 0)
        max_sector_notional = self.nav * self.max_sector_pct
        proposed_sector_notional = current_sector_exposure + notional

        if proposed_sector_notional > max_sector_notional:
            logger.warning(
                f"[GATE-6] SECTOR: {sector} would be {proposed_sector_notional / self.nav:.2%} > 20% limit"
            )

            available_sector = max_sector_notional - current_sector_exposure
            if available_sector < 50_000:
                logger.warning(f"[GATE-6] SECTOR: Sector {sector} full - REJECTING")
                return False, f"GATE_6_SECTOR_{sector}_FULL", 0

            # Scale to available
            final_quantity = int(available_sector * 0.95 / price)
            logger.info(
                f"[GATE-6] Scaled {symbol} to {final_quantity} for sector {sector}"
            )

        # =====================================================================
        # GATE 7: CORRELATION (Max 0.90 correlation with largest position)
        # =====================================================================
        if correlation_largest > self.max_correlation:
            logger.warning(
                f"[GATE-7] CORRELATION: {symbol} has {correlation_largest:.2f} correlation > 0.90"
            )

            # Apply correlation penalty to size
            correlation_penalty = (correlation_largest - self.max_correlation) / (
                1.0 - self.max_correlation
            )
            size_multiplier = 1.0 - (correlation_penalty * 0.5)  # Up to 50% reduction
            final_quantity = int(final_quantity * size_multiplier)
            logger.info(
                f"[GATE-7] Scaled {symbol} to {final_quantity} due to {correlation_largest:.2f} correlation"
            )

        # =====================================================================
        # ALL GATES PASSED - APPROVE
        # =====================================================================

        if final_quantity <= 0:
            logger.warning(
                f"[GATES] Final quantity <= 0 after scaling - REJECTING {symbol}"
            )
            return False, "GATES_SIZE_TOO_SMALL", 0

        logger.info(
            f"[GATES] ✓ APPROVED {symbol}: qty={final_quantity} @ ${price:.2f} = ${final_quantity * price:,.0f} "
            f"(VIX={vix:.0f}, Vol={realized_volatility:.2%}, Leverage={proposed_gross_leverage:.2f}x)"
        )

        return True, "GATES_APPROVED", final_quantity

    def _gate_market_hours(self) -> bool:
        """
        GATE 1: Market Hours Check (9:30 AM - 4:00 PM ET, Mon-Fri)

        Conservative default: FALSE (don't trade if uncertain)
        """
        try:
            now = datetime.now(pytz.timezone("US/Eastern"))

            # Weekend check
            if now.weekday() >= 5:
                logger.debug(f"[GATE-1] Weekend - market closed")
                return False

            # Time check: 9:30 AM - 4:00 PM
            market_open = time(9, 30)
            market_close = time(16, 0)
            current_time = now.time()

            is_open = market_open <= current_time <= market_close

            logger.debug(f"[GATE-1] Time={current_time}, Open={is_open}")
            return is_open

        except Exception as e:
            logger.critical(
                f"[GATE-1] Market hours check FAILED: {e} - defaulting to CLOSED (safe)"
            )
            return False  # CRITICAL: Default to FALSE (don't trade)

    def update_position_state(
        self, symbol: str, quantity: int, price: float, sector: str
    ):
        """Update tracking of positions for gate calculations"""
        notional = abs(quantity) * price

        if symbol not in self.current_positions:
            self.current_positions[symbol] = {
                "quantity": quantity,
                "price": price,
                "notional": notional,
                "sector": sector,
            }
        else:
            self.current_positions[symbol]["quantity"] = quantity
            self.current_positions[symbol]["notional"] = notional

        # Update sector exposure
        if sector not in self.sector_exposure:
            self.sector_exposure[sector] = notional
        else:
            # Recalculate sector total
            sector_notional = sum(
                p["notional"]
                for p in self.current_positions.values()
                if p.get("sector") == sector
            )
            self.sector_exposure[sector] = sector_notional

    def get_gate_status(self) -> Dict:
        """Return current status of all gates"""

        total_notional = sum(p["notional"] for p in self.current_positions.values())
        gross_leverage = total_notional / self.nav

        return {
            "positions_count": len(self.current_positions),
            "total_notional": total_notional,
            "gross_leverage": gross_leverage,
            "leverage_headroom": self.max_gross_leverage - gross_leverage,
            "sector_exposure": self.sector_exposure,
            "max_sector": max(
                (s / self.nav for s in self.sector_exposure.values()), default=0.0
            ),
        }


# Example usage in main.py
def check_order_with_7_gates(
    symbol: str,
    quantity: int,
    price: float,
    side: str,
    sector: str,
    adv: float,
    gate_manager: SevenGateRiskManager,
    vix: float = 20.0,
    realized_vol: float = 0.20,
) -> Tuple[bool, str, int]:
    """
    Wrapper function for easy mainpy integration.

    Usage:
        is_ok, reason, final_qty = check_order_with_7_gates(
            symbol="AAPL",
            quantity=1000,
            price=150,
            side="BUY",
            sector="Technology",
            adv=50_000_000,
            gate_manager=gate_mgr,
            vix=vix_value,
            realized_vol=vol_value,
        )

        if is_ok:
            place_order(symbol, final_qty, price, side)
        else:
            logger.warning(f"Order rejected: {reason}")
    """

    is_approved, reason, final_quantity = gate_manager.validate_order(
        symbol=symbol,
        quantity=quantity,
        price=price,
        side=side,
        sector=sector,
        adv=adv,
        vix=vix,
        realized_volatility=realized_vol,
    )

    if is_approved:
        gate_manager.update_position_state(symbol, final_quantity, price, sector)

    return is_approved, reason, final_quantity
