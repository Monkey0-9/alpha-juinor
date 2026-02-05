"""
Stress Testing Framework - Real-Time Risk Scenarios
======================================================

Elite-tier risk management: simulate extreme events.

Features:
1. Historical stress scenarios (2008, COVID, Flash Crash)
2. Hypothetical tail events
3. Liquidity shock modeling
4. Flash crash simulation
5. Portfolio impact estimation

Don't wait for catastrophe. Simulate it.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StressScenario(Enum):
    """Pre-defined stress scenarios."""
    CRASH_2008 = "2008_FINANCIAL_CRISIS"
    FLASH_CRASH_2010 = "2010_FLASH_CRASH"
    COVID_2020 = "2020_COVID_CRASH"
    TAPER_TANTRUM = "2013_TAPER_TANTRUM"
    VOLMAGEDDON = "2018_VOL_EXPLOSION"
    DOT_COM = "2000_DOT_COM_BURST"
    BLACK_MONDAY = "1987_BLACK_MONDAY"

    # Hypothetical
    LIQUIDITY_DROUGHT = "LIQUIDITY_DROUGHT"
    OVERNIGHT_GAP_10PCT = "OVERNIGHT_GAP_10PCT"
    CORRELATION_BREAKDOWN = "CORRELATION_BREAKDOWN"
    SECTOR_COLLAPSE = "SECTOR_COLLAPSE"


@dataclass
class ScenarioDefinition:
    """Definition of a stress scenario."""
    name: str
    description: str

    # Market shocks
    equity_shock_pct: float
    volatility_multiplier: float
    liquidity_multiplier: float  # <1 means reduced liquidity

    # Duration
    shock_days: int
    recovery_days: int

    # Sector-specific shocks
    sector_shocks: Dict[str, float] = field(default_factory=dict)

    # Correlations
    correlation_override: Optional[float] = None  # Force correlation to this

    # Probability
    estimated_probability: float = 0.05  # Annual probability


@dataclass
class StressTestResult:
    """Result of a stress test."""
    timestamp: datetime
    scenario_name: str

    # Portfolio impact
    portfolio_value_before: float
    portfolio_value_after: float
    portfolio_loss: float
    portfolio_loss_pct: float

    # VaR comparison
    stressed_var: float
    normal_var: float
    var_ratio: float  # Stressed / Normal

    # Liquidity impact
    liquidation_cost: float
    liquidation_time_hours: float
    can_meet_margin: bool

    # Worst positions
    worst_positions: List[Dict[str, Any]]

    # Survival
    survives_scenario: bool
    margin_call_triggered: bool

    # Recommendations
    recommendations: List[str]


# Historical scenario definitions
HISTORICAL_SCENARIOS = {
    StressScenario.CRASH_2008: ScenarioDefinition(
        name="2008 Financial Crisis",
        description="Lehman collapse, credit freeze, -50% equity decline",
        equity_shock_pct=-0.50,
        volatility_multiplier=4.0,
        liquidity_multiplier=0.3,
        shock_days=120,
        recovery_days=400,
        sector_shocks={
            "FINANCIALS": -0.70,
            "REAL_ESTATE": -0.60,
            "CONSUMER_DISCRETIONARY": -0.55
        },
        correlation_override=0.95,
        estimated_probability=0.02
    ),

    StressScenario.FLASH_CRASH_2010: ScenarioDefinition(
        name="2010 Flash Crash",
        description="Rapid -9% drop in minutes, quick recovery",
        equity_shock_pct=-0.09,
        volatility_multiplier=10.0,
        liquidity_multiplier=0.1,
        shock_days=1,
        recovery_days=1,
        correlation_override=0.99,
        estimated_probability=0.05
    ),

    StressScenario.COVID_2020: ScenarioDefinition(
        name="COVID-19 Crash",
        description="Global pandemic, -34% decline in 23 days",
        equity_shock_pct=-0.34,
        volatility_multiplier=5.0,
        liquidity_multiplier=0.4,
        shock_days=23,
        recovery_days=150,
        sector_shocks={
            "ENERGY": -0.60,
            "TRAVEL": -0.70,
            "FINANCIALS": -0.45
        },
        correlation_override=0.90,
        estimated_probability=0.01
    ),

    StressScenario.BLACK_MONDAY: ScenarioDefinition(
        name="1987 Black Monday",
        description="Single-day -22.6% crash",
        equity_shock_pct=-0.226,
        volatility_multiplier=15.0,
        liquidity_multiplier=0.05,
        shock_days=1,
        recovery_days=100,
        correlation_override=0.99,
        estimated_probability=0.01
    ),

    StressScenario.LIQUIDITY_DROUGHT: ScenarioDefinition(
        name="Liquidity Drought",
        description="Extreme market liquidity dry-up",
        equity_shock_pct=-0.15,
        volatility_multiplier=3.0,
        liquidity_multiplier=0.1,
        shock_days=5,
        recovery_days=30,
        estimated_probability=0.10
    ),

    StressScenario.OVERNIGHT_GAP_10PCT: ScenarioDefinition(
        name="10% Overnight Gap",
        description="Major adverse overnight gap",
        equity_shock_pct=-0.10,
        volatility_multiplier=2.0,
        liquidity_multiplier=0.5,
        shock_days=1,
        recovery_days=10,
        estimated_probability=0.15
    )
}


class LiquidityAdjustedVaR:
    """
    Liquidity-Adjusted Value at Risk (LVAR).

    Accounts for:
    1. Market risk (traditional VaR)
    2. Liquidation cost (bid-ask, market impact)
    3. Time to liquidate
    """

    def __init__(self, confidence: float = 0.99):
        """Initialize LVAR calculator."""
        self.confidence = confidence

        logger.info("[LVAR] Liquidity-Adjusted VaR initialized")

    def calculate(
        self,
        positions: Dict[str, float],  # Symbol -> Value
        volatilities: Dict[str, float],  # Symbol -> Daily vol
        correlations: Optional[pd.DataFrame] = None,
        avg_spreads: Optional[Dict[str, float]] = None,  # Symbol -> Spread %
        avg_daily_volumes: Optional[Dict[str, float]] = None,  # Symbol -> Volume
        holding_period: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate Liquidity-Adjusted VaR.

        Returns:
            - Traditional VaR
            - Liquidity cost
            - Total LVAR
        """
        if not positions:
            return {"lvar": 0, "var": 0, "liquidity_cost": 0}

        symbols = list(positions.keys())
        values = np.array([positions[s] for s in symbols])
        vols = np.array([volatilities.get(s, 0.02) for s in symbols])

        total_value = values.sum()

        # 1. Traditional VaR (parametric)
        # Portfolio volatility
        if correlations is not None and len(symbols) > 1:
            # Use correlation matrix
            corr_matrix = np.ones((len(symbols), len(symbols)))
            for i, s1 in enumerate(symbols):
                for j, s2 in enumerate(symbols):
                    if s1 in correlations.index and s2 in correlations.columns:
                        corr_matrix[i, j] = correlations.loc[s1, s2]

            weights = values / total_value
            cov_matrix = np.outer(vols, vols) * corr_matrix
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        else:
            # Simple weighted average
            weights = values / total_value
            portfolio_vol = np.sqrt(np.sum((weights * vols) ** 2))

        # Scale to holding period
        portfolio_vol *= np.sqrt(holding_period)

        # Z-score for confidence level
        z_score = 2.33 if self.confidence == 0.99 else 1.65  # 99% or 95%

        traditional_var = total_value * portfolio_vol * z_score

        # 2. Liquidity cost
        liquidity_cost = 0

        if avg_spreads is not None and avg_daily_volumes is not None:
            for symbol in symbols:
                position_value = positions[symbol]
                spread = avg_spreads.get(symbol, 0.001)  # Default 0.1%
                adv = avg_daily_volumes.get(symbol, 1000000)  # Default $1M

                # Cost = spread/2 + market impact
                # Market impact ~ sqrt(position/ADV) * some coefficient
                participation_rate = min(1, position_value / adv)
                market_impact = 0.1 * np.sqrt(participation_rate)  # 10% sqrt rule

                liquidity_cost += position_value * (spread / 2 + market_impact)

        # 3. Total LVAR
        lvar = traditional_var + liquidity_cost

        return {
            "lvar": float(lvar),
            "var": float(traditional_var),
            "liquidity_cost": float(liquidity_cost),
            "portfolio_vol": float(portfolio_vol),
            "holding_period": holding_period,
            "confidence": self.confidence
        }


class StressTester:
    """
    Real-time stress testing framework.

    Simulates extreme market scenarios on current portfolio.
    """

    def __init__(self):
        """Initialize the stress tester."""
        self.scenarios = HISTORICAL_SCENARIOS.copy()
        self.lvar = LiquidityAdjustedVaR()

        logger.info(
            f"[STRESS] Stress Tester initialized with "
            f"{len(self.scenarios)} scenarios"
        )

    def add_scenario(self, scenario_id: str, definition: ScenarioDefinition):
        """Add a custom scenario."""
        self.scenarios[scenario_id] = definition

    def stress_test(
        self,
        positions: Dict[str, float],  # Symbol -> Value
        volatilities: Dict[str, float],
        sector_mapping: Optional[Dict[str, str]] = None,  # Symbol -> Sector
        margin_requirement: float = 0.25,
        scenario: StressScenario = StressScenario.CRASH_2008
    ) -> StressTestResult:
        """
        Run a stress test on the portfolio.

        Args:
            positions: Current positions {symbol: value}
            volatilities: Volatilities {symbol: daily_vol}
            sector_mapping: Sector assignments
            margin_requirement: Margin requirement ratio
            scenario: Scenario to simulate
        """
        if scenario not in self.scenarios:
            scenario = StressScenario.CRASH_2008

        scenario_def = self.scenarios[scenario]

        # Calculate portfolio value before
        portfolio_before = sum(positions.values())

        # Apply scenario shocks
        stressed_positions = {}
        position_impacts = []

        for symbol, value in positions.items():
            # Base shock
            shock = scenario_def.equity_shock_pct

            # Sector-specific shock
            if sector_mapping and symbol in sector_mapping:
                sector = sector_mapping[symbol]
                if sector in scenario_def.sector_shocks:
                    shock = scenario_def.sector_shocks[sector]

            # Volatility scaling
            base_vol = volatilities.get(symbol, 0.02)
            vol_adjusted_shock = shock * (1 + base_vol / 0.02)  # Higher vol = bigger shock

            stressed_value = value * (1 + vol_adjusted_shock)
            stressed_positions[symbol] = stressed_value

            position_impacts.append({
                "symbol": symbol,
                "value_before": value,
                "value_after": stressed_value,
                "loss": value - stressed_value,
                "loss_pct": -vol_adjusted_shock
            })

        # Portfolio value after
        portfolio_after = sum(stressed_positions.values())
        portfolio_loss = portfolio_before - portfolio_after
        portfolio_loss_pct = portfolio_loss / portfolio_before if portfolio_before > 0 else 0

        # Calculate stressed VaR
        stressed_vols = {
            s: v * scenario_def.volatility_multiplier
            for s, v in volatilities.items()
        }

        lvar_result = self.lvar.calculate(positions, volatilities)
        stressed_lvar_result = self.lvar.calculate(stressed_positions, stressed_vols)

        # Liquidity impact
        liquidation_cost = portfolio_before * (1 - scenario_def.liquidity_multiplier) * 0.02
        liquidation_time = 8 / scenario_def.liquidity_multiplier  # Hours

        # Margin check
        margin_required = portfolio_before * margin_requirement
        can_meet_margin = portfolio_after >= margin_required
        margin_call = portfolio_after < margin_required

        # Survival check
        survives = portfolio_loss_pct < 0.5 and can_meet_margin

        # Sort worst positions
        worst = sorted(position_impacts, key=lambda x: x["loss"], reverse=True)[:5]

        # Recommendations
        recommendations = self._generate_recommendations(
            portfolio_loss_pct, worst, survives, margin_call, scenario_def
        )

        return StressTestResult(
            timestamp=datetime.utcnow(),
            scenario_name=scenario_def.name,
            portfolio_value_before=portfolio_before,
            portfolio_value_after=portfolio_after,
            portfolio_loss=portfolio_loss,
            portfolio_loss_pct=portfolio_loss_pct,
            stressed_var=stressed_lvar_result["lvar"],
            normal_var=lvar_result["lvar"],
            var_ratio=stressed_lvar_result["lvar"] / lvar_result["lvar"] if lvar_result["lvar"] > 0 else 0,
            liquidation_cost=liquidation_cost,
            liquidation_time_hours=liquidation_time,
            can_meet_margin=can_meet_margin,
            worst_positions=worst,
            survives_scenario=survives,
            margin_call_triggered=margin_call,
            recommendations=recommendations
        )

    def run_all_scenarios(
        self,
        positions: Dict[str, float],
        volatilities: Dict[str, float],
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, StressTestResult]:
        """Run all stress scenarios."""
        results = {}

        for scenario in self.scenarios.keys():
            if isinstance(scenario, StressScenario):
                result = self.stress_test(
                    positions, volatilities, sector_mapping, scenario=scenario
                )
                results[scenario.value] = result

        return results

    def _generate_recommendations(
        self,
        loss_pct: float,
        worst_positions: List[Dict],
        survives: bool,
        margin_call: bool,
        scenario: ScenarioDefinition
    ) -> List[str]:
        """Generate recommendations based on stress test."""
        recs = []

        if not survives:
            recs.append("CRITICAL: Portfolio does not survive this scenario")

        if margin_call:
            recs.append("WARNING: Margin call would be triggered")

        if loss_pct > 0.30:
            recs.append("Consider reducing overall exposure by 20-30%")
        elif loss_pct > 0.20:
            recs.append("Consider adding tail hedges (puts, VIX calls)")

        if worst_positions:
            top_loser = worst_positions[0]
            if top_loser["loss_pct"] > 0.40:
                recs.append(
                    f"HIGH RISK: {top_loser['symbol']} would lose "
                    f"{top_loser['loss_pct']:.0%} - consider hedging"
                )

        # Sector concentration
        if scenario.sector_shocks:
            recs.append("Reduce concentration in crisis-sensitive sectors")

        if scenario.liquidity_multiplier < 0.3:
            recs.append("Maintain higher cash buffer for liquidity events")

        return recs

    def get_summary(
        self,
        results: Dict[str, StressTestResult]
    ) -> Dict[str, Any]:
        """Get summary of all stress tests."""
        if not results:
            return {}

        losses = [r.portfolio_loss_pct for r in results.values()]
        survival_rate = sum(1 for r in results.values() if r.survives_scenario) / len(results)

        worst_scenario = max(results.items(), key=lambda x: x[1].portfolio_loss_pct)

        return {
            "scenarios_tested": len(results),
            "survival_rate": survival_rate,
            "avg_loss_pct": np.mean(losses),
            "max_loss_pct": max(losses),
            "worst_scenario": worst_scenario[0],
            "worst_loss": worst_scenario[1].portfolio_loss_pct,
            "margin_calls": sum(1 for r in results.values() if r.margin_call_triggered)
        }


# Singleton
_tester: Optional[StressTester] = None


def get_stress_tester() -> StressTester:
    """Get or create the Stress Tester."""
    global _tester
    if _tester is None:
        _tester = StressTester()
    return _tester
