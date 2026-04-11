"""
Multi-Currency Settlement Engine
=================================
Handles FX conversion, settlement, and P&L
aggregation across multiple currencies.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CurrencySettlement:
    """
    Multi-currency settlement for global trading.

    Features:
    - Real-time FX rate management
    - Settlement in base currency (USD)
    - Unrealized P&L in local + base currency
    - FX hedging recommendations
    - Currency exposure tracking
    """

    # Default FX rates (USD-based) — updated at runtime
    DEFAULT_RATES = {
        "USD": 1.0,
        "EUR": 1.08,
        "GBP": 1.27,
        "JPY": 0.0067,
        "AUD": 0.65,
        "CAD": 0.74,
        "CHF": 1.13,
        "HKD": 0.128,
        "SGD": 0.74,
        "NZD": 0.61,
        "ZAR": 0.055,
        "MXN": 0.059,
        "BRL": 0.20,
        "TRY": 0.031,
        "INR": 0.012,
        "TWD": 0.031,
    }

    def __init__(
        self, base_currency: str = "USD"
    ):
        self.base_currency = base_currency
        self._fx_rates = dict(self.DEFAULT_RATES)
        self._exposures: Dict[str, float] = {}
        self._last_update = datetime.utcnow()

    def update_rate(
        self, currency: str, rate_vs_usd: float
    ):
        """Update FX rate (currency per 1 USD)."""
        self._fx_rates[currency] = rate_vs_usd
        self._last_update = datetime.utcnow()

    def update_rates_batch(
        self, rates: Dict[str, float]
    ):
        """Batch update FX rates."""
        self._fx_rates.update(rates)
        self._last_update = datetime.utcnow()

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str = "",
    ) -> float:
        """
        Convert amount between currencies.

        Args:
            amount: Amount in from_currency
            from_currency: Source currency code
            to_currency: Target (default: base_currency)

        Returns:
            Converted amount
        """
        to_curr = to_currency or self.base_currency
        if from_currency == to_curr:
            return amount

        # Convert to USD first, then to target
        from_rate = self._fx_rates.get(
            from_currency, 1.0
        )
        to_rate = self._fx_rates.get(to_curr, 1.0)

        usd_amount = amount * from_rate
        return usd_amount / to_rate

    def settle_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        local_currency: str,
    ) -> Dict:
        """
        Settle a trade and compute base currency
        equivalent.
        """
        local_value = quantity * price
        base_value = self.convert(
            local_value, local_currency
        )

        # Track exposure
        curr_exposure = self._exposures.get(
            local_currency, 0
        )
        self._exposures[local_currency] = (
            curr_exposure + local_value
        )

        return {
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "local_currency": local_currency,
            "local_value": round(local_value, 2),
            "base_currency": self.base_currency,
            "base_value": round(base_value, 2),
            "fx_rate": self._fx_rates.get(
                local_currency, 1.0
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_currency_exposure(self) -> Dict[str, Dict]:
        """
        Get current currency exposure breakdown.

        Returns:
            Dict with local and base currency values
        """
        result = {}
        for curr, local_val in self._exposures.items():
            base_val = self.convert(local_val, curr)
            result[curr] = {
                "local_value": round(local_val, 2),
                "base_value": round(base_val, 2),
                "fx_rate": self._fx_rates.get(curr, 1.0),
                "pct_of_total": 0,  # Filled below
            }

        total = sum(
            v["base_value"]
            for v in result.values()
        )
        if total:
            for curr in result:
                result[curr]["pct_of_total"] = round(
                    result[curr]["base_value"]
                    / total * 100, 2
                )

        return result

    def fx_hedge_recommendation(
        self, threshold_pct: float = 10.0
    ) -> List[Dict]:
        """
        Recommend FX hedges for overexposed currencies.

        Args:
            threshold_pct: Max acceptable exposure (%)
        """
        exposure = self.get_currency_exposure()
        recs = []

        for curr, info in exposure.items():
            if (
                curr != self.base_currency
                and info["pct_of_total"] > threshold_pct
            ):
                hedge_amount = (
                    info["base_value"]
                    * (
                        info["pct_of_total"]
                        - threshold_pct
                    )
                    / 100
                )
                recs.append({
                    "currency": curr,
                    "current_pct": info["pct_of_total"],
                    "threshold_pct": threshold_pct,
                    "recommend_hedge_usd": round(
                        hedge_amount, 2
                    ),
                    "instrument": (
                        f"{curr}{self.base_currency}"
                    ),
                })

        return recs

    def get_rates(self) -> Dict[str, float]:
        """Get current FX rates."""
        return dict(self._fx_rates)

    def aggregate_pnl(
        self,
        positions: Dict[str, Dict],
    ) -> Dict:
        """
        Aggregate P&L across currencies.

        Args:
            positions: {symbol: {pnl, currency}}

        Returns:
            Total P&L in base currency
        """
        total_base = 0
        by_currency = {}

        for symbol, pos in positions.items():
            pnl = pos.get("pnl", 0)
            curr = pos.get("currency", "USD")
            base_pnl = self.convert(pnl, curr)
            total_base += base_pnl

            if curr not in by_currency:
                by_currency[curr] = 0
            by_currency[curr] += pnl

        return {
            "total_base_pnl": round(total_base, 2),
            "base_currency": self.base_currency,
            "by_currency": {
                k: round(v, 2)
                for k, v in by_currency.items()
            },
        }


# Singleton
_settlement: Optional[CurrencySettlement] = None


def get_settlement() -> CurrencySettlement:
    """Get or create settlement engine."""
    global _settlement
    if _settlement is None:
        _settlement = CurrencySettlement()
    return _settlement
