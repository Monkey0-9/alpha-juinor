import numpy as np
from typing import Dict, List
from .greeks_calculator import RealTimeGreeksCalculator, Greeks

class OptionsRiskManager:
    """Institutional-grade options risk management"""
    
    def __init__(self):
        self.greeks_calc = RealTimeGreeksCalculator()
        self.portfolio_greeks = {
            "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0
        }
        self.limits = {
            "max_net_delta": 5000,
            "max_net_gamma": 2000,
            "max_vega_exposure": 10000
        }

    def update_portfolio_risk(self, positions: List[Dict]):
        """Calculate aggregated portfolio Greeks"""
        new_greeks = {k: 0.0 for k in self.portfolio_greeks}
        
        for pos in positions:
            g = self.greeks_calc.calculate_greeks(
                pos["S"], pos["K"], pos["T"], pos["r"], pos["sigma"], pos["type"]
            )
            qty = pos["quantity"]
            new_greeks["delta"] += g.delta * qty
            new_greeks["gamma"] += g.gamma * qty
            new_greeks["theta"] += g.theta * qty
            new_greeks["vega"] += g.vega * qty
            new_greeks["rho"] += g.rho * qty
            
        self.portfolio_greeks = new_greeks
        return self.check_violations()

    def check_violations(self) -> List[str]:
        violations = []
        if abs(self.portfolio_greeks["delta"]) > self.limits["max_net_delta"]:
            violations.append("DELTA_LIMIT_EXCEEDED")
        if abs(self.portfolio_greeks["gamma"]) > self.limits["max_net_gamma"]:
            violations.append("GAMMA_LIMIT_EXCEEDED")
        return violations
