"""
PACK 5B: CURRENCY RISK MANAGEMENT
Manages multi-currency portfolio with FX concentration and correlation limits
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CurrencyRiskManager:
    """Manage FX exposure and currency correlation risk"""

    MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "HKD"]

    def __init__(
        self,
        base_currency: str = "USD",
        max_unhedged_pct: float = 0.10,
        max_concentration: float = 0.15,
        max_correlation: float = 0.85,
        hedging_trigger: float = 0.08,
    ):
        self.base_currency = base_currency
        self.max_unhedged_pct = max_unhedged_pct  # Max 10% unhedged
        self.max_concentration = max_concentration  # Max 15% in one currency
        self.max_correlation = max_correlation  # > 0.85 triggers action
        self.hedging_trigger = hedging_trigger  # Trigger at 8%

        # FX Rates (stub - would be real-time in production)
        self.fx_rates = {
            "EURUSD": 1.095,
            "GBPUSD": 1.265,
            "JPYUSD": 0.0067,
            "AUDUSD": 0.65,
            "CADUSD": 0.74,
            "HKDUSD": 0.128,
        }

        # Track exposure per currency
        self.fx_exposure = {}  # currency -> notional in base currency
        self.nav = 1_000_000

    def get_currency_from_symbol(self, symbol: str) -> str:
        """Extract currency from symbol"""
        # Map symbols to currencies
        currency_map = {
            "JPY": "JPY",
            "EUR": "EUR",
            "GBP": "GBP",
            "AUD": "AUD",
            "CAD": "CAD",
            "HKD": "HKD",
            "EURUSD=X": "EUR",
            "GBPUSD=X": "GBP",
            "JPY=X": "JPY",
            "AUDUSD=X": "AUD",
            "CAD=X": "CAD",
            "HKD=X": "HKD",
        }

        for key, val in currency_map.items():
            if key in symbol:
                return val

        return self.base_currency  # Default to USD for unknown symbols

    def add_position(
        self, symbol: str, notional_usd: float, currency: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Add position and apply FX constraints.
        Returns (approved_notional, reason)
        """
        if currency is None:
            currency = self.get_currency_from_symbol(symbol)

        # Check concentration: max 15% of NAV in any single currency
        current_exposure = self.fx_exposure.get(currency, 0)
        max_single_currency = self.nav * self.max_concentration

        if current_exposure + notional_usd > max_single_currency:
            allowed = max_single_currency - current_exposure
            logger.warning(
                f"[FX] {currency}: Would exceed concentration limit. "
                f"Reducing from ${notional_usd:,.0f} to ${allowed:,.0f}"
            )
            self.fx_exposure[currency] = max_single_currency
            return allowed, "currency_concentration_limit"

        # Check overall unhedged: max 10% of NAV
        total_unhedged = sum(
            v for k, v in self.fx_exposure.items() if k != self.base_currency
        )
        total_unhedged += notional_usd

        max_unhedged_total = self.nav * self.max_unhedged_pct

        if total_unhedged > max_unhedged_total:
            logger.warning(
                f"[FX] Total unhedged FX: ${total_unhedged:,.0f} > "
                f"limit ${max_unhedged_total:,.0f}"
            )
            trigger_hedging_signal = total_unhedged > self.nav * self.hedging_trigger

            if trigger_hedging_signal:
                logger.critical(
                    f"[FX] Hedging signal triggered at {total_unhedged / self.nav:.2%} exposure"
                )

            # Allow only portion of position
            allowed = max_unhedged_total - (total_unhedged - notional_usd)
            self.fx_exposure[currency] = self.fx_exposure.get(currency, 0) + allowed
            return allowed, "fx_unhedged_limit"

        # Position approved
        self.fx_exposure[currency] = self.fx_exposure.get(currency, 0) + notional_usd
        logger.info(
            f"[FX] Added {currency} position: ${notional_usd:,.0f}, "
            f"Total exposure: {sum(self.fx_exposure.values()) / self.nav:.2%}"
        )
        return notional_usd, "approved"

    def get_fx_correlation(self, currency1: str, currency2: str) -> float:
        """Get correlation between two currencies (stub)"""
        # In production, would calculate from historical data
        # For now, return predefined correlations

        correlations = {
            ("EUR", "GBP"): 0.75,
            ("EUR", "JPY"): 0.45,
            ("GBP", "JPY"): 0.35,
            ("AUD", "JPY"): 0.65,
            ("CAD", "USD"): 0.82,
            ("HKD", "USD"): 0.90,  # High correlation
        }

        key = tuple(sorted([currency1, currency2]))
        return correlations.get(key, 0.5)

    def check_fx_correlation_risk(self) -> Dict:
        """Check for high FX correlation risks"""
        risks = {}

        currencies_exposed = list(self.fx_exposure.keys())

        for i, curr1 in enumerate(currencies_exposed):
            for curr2 in currencies_exposed[i + 1 :]:
                corr = self.get_fx_correlation(curr1, curr2)

                if corr > self.max_correlation:
                    risks[f"{curr1}-{curr2}"] = {
                        "correlation": corr,
                        "exposure_1": self.fx_exposure.get(curr1, 0),
                        "exposure_2": self.fx_exposure.get(curr2, 0),
                        "action": "REDUCE_CONCENTRATED_PAIR",
                    }

        if risks:
            logger.warning(
                f"[FX] High correlation risks detected: {list(risks.keys())}"
            )

        return risks

    def get_hedging_recommendations(self) -> Dict:
        """Get FX hedging recommendations"""
        recommendations = {}

        total_unhedged = sum(
            v for k, v in self.fx_exposure.items() if k != self.base_currency
        )
        unhedged_pct = total_unhedged / self.nav

        if unhedged_pct > self.hedging_trigger:
            # Recommend hedging
            for currency, exposure in self.fx_exposure.items():
                if currency != self.base_currency:
                    exposure_pct = exposure / self.nav
                    if exposure_pct > 0.05:
                        hedge_amount = exposure * 0.50  # Hedge 50%

                        recommendations[currency] = {
                            "exposure": exposure,
                            "exposure_pct": exposure_pct,
                            "hedge_amount": hedge_amount,
                            "hedge_ratio": 0.50,
                            "action": f"Short {currency} forward",
                        }

        return recommendations

    def get_fx_exposure_report(self) -> Dict:
        """Get comprehensive FX exposure report"""
        total_exposed = sum(
            v for k, v in self.fx_exposure.items() if k != self.base_currency
        )

        report = {
            "base_currency": self.base_currency,
            "nav": self.nav,
            "exposures": {},
            "total_unhedged": total_exposed,
            "total_unhedged_pct": total_exposed / self.nav if self.nav > 0 else 0,
            "max_allowed_unhedged": self.nav * self.max_unhedged_pct,
            "status": (
                "OK"
                if total_exposed <= self.nav * self.max_unhedged_pct
                else "OVER_LIMIT"
            ),
        }

        for currency, exposure in self.fx_exposure.items():
            report["exposures"][currency] = {
                "notional": exposure,
                "pct_of_nav": exposure / self.nav if self.nav > 0 else 0,
                "max_allowed": self.nav * self.max_concentration,
            }

        return report


if __name__ == "__main__":
    import json

    fx = CurrencyRiskManager(nav=1_000_000)

    # Test positions
    fx.add_position("ASML.AS", 100_000, "EUR")
    fx.add_position("SHELL", 80_000, "GBP")
    fx.add_position("9984.T", 60_000, "JPY")

    print("FX Exposure Report:")
    print(json.dumps(fx.get_fx_exposure_report(), indent=2))

    print("\nFX Correlation Risks:")
    print(json.dumps(fx.check_fx_correlation_risk(), indent=2))

    print("\nHedging Recommendations:")
    print(json.dumps(fx.get_hedging_recommendations(), indent=2))
