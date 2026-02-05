"""
Stress Testing Framework
========================

Comprehensive stress testing for institutional risk management.

Scenarios:
- 2008 Financial Crisis
- 2020 COVID Crash
- Custom scenarios
- Monte Carlo simulations

Phase 3.1: Advanced Risk Framework
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    description: str
    equity_shock: float  # e.g., -0.40 = 40% drop
    vol_multiplier: float
    correlation_shift: float
    liquidity_factor: float


@dataclass
class StressTestResult:
    """Result from stress test."""
    scenario_name: str
    portfolio_pnl: float
    portfolio_pnl_pct: float
    worst_position: str
    worst_position_loss: float
    var_breach: bool
    survival: bool


class StressTestingFramework:
    """
    Institutional stress testing framework.
    """

    def __init__(self):
        self.scenarios: Dict[str, StressScenario] = {}
        self._register_default_scenarios()
        logger.info(f"Stress Testing: {len(self.scenarios)} scenarios loaded")

    def _register_default_scenarios(self):
        """Register standard stress scenarios."""
        self.scenarios["2008_FINANCIAL_CRISIS"] = StressScenario(
            name="2008 Financial Crisis",
            description="Lehman collapse scenario",
            equity_shock=-0.55,
            vol_multiplier=4.0,
            correlation_shift=0.3,
            liquidity_factor=0.3
        )

        self.scenarios["2020_COVID_CRASH"] = StressScenario(
            name="2020 COVID Crash",
            description="March 2020 rapid selloff",
            equity_shock=-0.35,
            vol_multiplier=5.0,
            correlation_shift=0.4,
            liquidity_factor=0.5
        )

        self.scenarios["FLASH_CRASH"] = StressScenario(
            name="Flash Crash",
            description="2010-style flash crash",
            equity_shock=-0.10,
            vol_multiplier=10.0,
            correlation_shift=0.5,
            liquidity_factor=0.1
        )

        self.scenarios["RATE_SHOCK"] = StressScenario(
            name="Interest Rate Shock",
            description="300bp rate spike",
            equity_shock=-0.15,
            vol_multiplier=2.0,
            correlation_shift=0.2,
            liquidity_factor=0.7
        )

        self.scenarios["GEOPOLITICAL"] = StressScenario(
            name="Geopolitical Crisis",
            description="Major geopolitical event",
            equity_shock=-0.20,
            vol_multiplier=3.0,
            correlation_shift=0.25,
            liquidity_factor=0.6
        )

        self.scenarios["SECTOR_MELTDOWN"] = StressScenario(
            name="Sector Meltdown",
            description="Single sector -50%",
            equity_shock=-0.25,
            vol_multiplier=2.5,
            correlation_shift=0.15,
            liquidity_factor=0.4
        )

        self.scenarios["STAGFLATION"] = StressScenario(
            name="Stagflation",
            description="High inflation + recession",
            equity_shock=-0.30,
            vol_multiplier=2.0,
            correlation_shift=0.1,
            liquidity_factor=0.8
        )

        self.scenarios["LIQUIDITY_CRISIS"] = StressScenario(
            name="Liquidity Crisis",
            description="Market-wide liquidity freeze",
            equity_shock=-0.20,
            vol_multiplier=3.5,
            correlation_shift=0.6,
            liquidity_factor=0.1
        )

        self.scenarios["CURRENCY_CRISIS"] = StressScenario(
            name="Currency Crisis",
            description="Major currency devaluation",
            equity_shock=-0.15,
            vol_multiplier=2.5,
            correlation_shift=0.2,
            liquidity_factor=0.5
        )

        self.scenarios["BLACK_SWAN"] = StressScenario(
            name="Black Swan",
            description="Extreme tail event",
            equity_shock=-0.60,
            vol_multiplier=6.0,
            correlation_shift=0.5,
            liquidity_factor=0.2
        )

    def run_stress_test(
        self,
        scenario_name: str,
        positions: Dict[str, float],
        prices: Dict[str, float],
        betas: Dict[str, float],
        nav: float,
        var_limit: float = 0.15
    ) -> StressTestResult:
        """
        Run a single stress test.
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]

        # Calculate stressed P&L
        total_pnl = 0.0
        worst_sym = ""
        worst_loss = 0.0

        for symbol, qty in positions.items():
            price = prices.get(symbol, 100.0)
            beta = betas.get(symbol, 1.0)

            # Apply shock (beta-adjusted)
            symbol_shock = scenario.equity_shock * beta
            position_value = qty * price
            loss = position_value * symbol_shock

            total_pnl += loss

            if loss < worst_loss:
                worst_loss = loss
                worst_sym = symbol

        pnl_pct = total_pnl / nav if nav > 0 else 0.0
        var_breach = abs(pnl_pct) > var_limit
        survival = abs(pnl_pct) < 0.50  # Can survive up to 50% drawdown

        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_pnl=total_pnl,
            portfolio_pnl_pct=pnl_pct,
            worst_position=worst_sym,
            worst_position_loss=worst_loss,
            var_breach=var_breach,
            survival=survival
        )

    def run_all_scenarios(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        betas: Dict[str, float],
        nav: float
    ) -> List[StressTestResult]:
        """
        Run all registered stress scenarios.
        """
        results = []

        for scenario_name in self.scenarios:
            result = self.run_stress_test(
                scenario_name, positions, prices, betas, nav
            )
            results.append(result)

        return results

    def get_summary_report(
        self, results: List[StressTestResult]
    ) -> Dict[str, Any]:
        """
        Generate summary report from stress tests.
        """
        if not results:
            return {"status": "NO_TESTS_RUN"}

        worst = min(results, key=lambda r: r.portfolio_pnl_pct)
        breaches = [r for r in results if r.var_breach]
        failures = [r for r in results if not r.survival]

        return {
            "total_scenarios": len(results),
            "var_breaches": len(breaches),
            "survival_failures": len(failures),
            "worst_scenario": worst.scenario_name,
            "worst_pnl_pct": worst.portfolio_pnl_pct,
            "stress_test_passed": len(failures) == 0
        }


# Singleton
_stress_framework = None


def get_stress_framework() -> StressTestingFramework:
    global _stress_framework
    if _stress_framework is None:
        _stress_framework = StressTestingFramework()
    return _stress_framework
