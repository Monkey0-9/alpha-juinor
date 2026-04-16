"""
risk/portfolio_stress_simulator.py

Pre-Trade Portfolio Stress Simulator (Ticket 22)
Simulates ruin scenarios before allocation.

Scenarios:
1. Market Crash (-5 sigma move)
2. Correlation Spike (All correlations -> 1.0)
3. Liquidity Freeze (ADV -> 20%)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger("STRESS_SIMULATOR")

@dataclass
class StressResult:
    passed: bool
    max_drawdown: float
    ruin_probability: float
    scenarios: Dict[str, float]
    reason: str = ""

class PortfolioStressSimulator:
    """
    Simulates extreme market conditions to prevent portfolio ruin.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        # Hard limits
        self.max_drawdown_limit = self.config.get("max_drawdown_limit", 0.20) # 20% limit
        self.max_leverage_limit = self.config.get("max_leverage_limit", 2.0)
        self.liquidity_haircut = 0.20 # 20% of normal liquidity

        # Market crash definition (-5 daily sigma)
        # Daily vol ~ 1.5% -> 5 sigma ~ 7.5% drop
        self.market_crash_return = -0.075

    def simulate(
        self,
        portfolio_weights: Dict[str, float],
        betas: Dict[str, float],
        sigmas: Dict[str, float],
        advs: Dict[str, float],
        prices: Dict[str, float],
        nav: float
    ) -> StressResult:
        """
        Run stress simulations on the PROPOSED portfolio.

        Args:
            portfolio_weights: {symbol: weight} (post-allocation)
            betas: {symbol: beta to SPY}
            sigmas: {symbol: daily volatility}
            advs: {symbol: Average Daily Volume in shares}
            prices: {symbol: current price}
            nav: Current Net Asset Value

        Returns:
            StressResult indicating pass/fail.
        """
        scenarios = {}

        # 0. Leverage Check
        gross_lev = sum(abs(w) for w in portfolio_weights.values())
        scenarios["gross_leverage"] = gross_lev

        if gross_lev > self.max_leverage_limit:
             return StressResult(
                passed=False,
                max_drawdown=0.0,
                ruin_probability=1.0,
                scenarios=scenarios,
                reason=f"Gross Leverage {gross_lev:.2f} > Limit {self.max_leverage_limit}"
            )

        # 1. Market Crash Scenario (-5 sigma)
        # PnL = Sum(w * beta * market_ret) + Sum(w * (1-beta) * alpha?)
        # Simplified: PnL ~= Beta * MarketRet

        portfolio_beta = 0.0
        for sym, w in portfolio_weights.items():
            b = betas.get(sym, 1.0) # Default to 1.0 if unknown high beta
            portfolio_beta += w * b

        crash_impact = portfolio_beta * self.market_crash_return
        scenarios["crash_impact"] = crash_impact

        if crash_impact < -self.max_drawdown_limit:
             return StressResult(
                passed=False,
                max_drawdown=crash_impact,
                ruin_probability=0.8, # High prob of ruin check
                scenarios=scenarios,
                reason=f"Market Crash Impact {crash_impact:.2%} exceeds limit {self.max_drawdown_limit:.2%}"
            )

        # 2. Correlation Spike Scenario (Corr -> 1.0)
        # If all assets are perfectly correlated, PortVol = Sum(w * sigma)
        # This is the theoretical maximum volatility.
        max_vol = 0.0
        for sym, w in portfolio_weights.items():
            s = sigmas.get(sym, 0.02) # Default 2%
            max_vol += abs(w) * s

        # Stress CVaR 95 (assuming Normal even though it's stress)
        # CVaR (Normal) ~= -1.65 * sigma
        # But in stress, let's use -2.33 (99%) or higher?
        # Let's say Stress CVaR = -2.0 * max_vol
        stress_cvar = -2.0 * max_vol
        scenarios["stress_corr_cvar"] = stress_cvar

        if stress_cvar < -self.max_drawdown_limit: # Using DD limit for CVaR too?
             # Maybe CVaR limit should be tighter, e.g. 5% daily?
             pass # Warn but don't block? Or Block?
             # "Simulate ruin" -> Block.
             # If daily potential loss is 20%, you are dead.
             if stress_cvar < -0.10: # 10% daily loss potential
                 return StressResult(
                    passed=False,
                    max_drawdown=stress_cvar,
                    ruin_probability=0.5,
                    scenarios=scenarios,
                    reason=f"Stress Correlation CVaR {stress_cvar:.2%} exceeds safety threshold -10%"
                )

        # 3. Liquidity Freeze Scenario (ADV * 0.2)
        # Can we liquidate 100% of portfolio in 1 day without > 2% cost?
        # Notional = weight * NAV
        # Share Count = Notional / Price
        # Participation = Share Count / (ADV * 0.2)
        # Impact Cost (Linear) ~= 0.1 * Participation? (Square root law better)
        # Impact (bps) ~= 100 * (Size/ADV)^0.5 ?
        # Let's check max_participation. If > 10% of Stress ADV, fail.

        for sym, w in portfolio_weights.items():
            if abs(w) < 0.001: continue

            notional = abs(w) * nav
            p = prices.get(sym, 100.0)
            shares = notional / p

            adv = advs.get(sym, 1000000)
            stress_adv = adv * self.liquidity_haircut

            participation = shares / max(stress_adv, 1)

            if participation > 0.10: # Trying to sell >10% of daily volume
                 return StressResult(
                    passed=False,
                    max_drawdown=0.0,
                    ruin_probability=0.2,
                    scenarios=scenarios,
                    reason=f"Liquidity Trap: {sym} needs {participation:.1%} of Stress ADV to liquidate"
                )

        # All passed
        return StressResult(
            passed=True,
            max_drawdown=crash_impact,
            ruin_probability=0.001,
            scenarios=scenarios,
            reason="All Stress Tests Passed"
        )
