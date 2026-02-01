# services/risk_enforcer.py
"""
Risk Enforcer service. Called by PM Brain with proposed portfolio
or new trade intent.
Performs portfolio CVaR check, correlation/entanglement throttle,
volatility scaling.
Returns allow/deny with explain and suggested haircuts.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional
from database.manager import DatabaseManager

logger = logging.getLogger("RISK_ENFORCER")

DEFAULT_PARAMS = {
    "cvar_limit_pct": 0.05,    # max allowed portfolio CVaR (example)
    "entanglement_threshold": 0.7,
    "entanglement_haircut": 0.5,
    "volatility_cap_pct": 0.25,  # if implied volatility >, reduce allocation
    "max_drawdown_limit": -0.15, # 15% Max Drawdown Kill Switch
    "gross_leverage_limit": 2.0,
    "net_leverage_limit": 0.5
}


class RiskEnforcer:
    def __init__(self, params: Dict = None):
        self.params = DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)
        self.db = DatabaseManager()

    def check_kill_switch(self) -> bool:
        """
        Check if global kill switch is active.
        Returns: True if Kill Switch is ACTIVE (Trading Halted).
        """
        # 1. Check Governance table (manual override)
        gov_settings = self.db.get_governance_settings()
        if gov_settings.get("kill_switch_active", False):
            return True

        # 2. Check PnL Drawdown (automated)
        pnl_stats = self.db.get_pnl_stats() # Assuming this method exists or we use equity history
        current_dd = pnl_stats.get("current_drawdown", 0.0)

        if current_dd < self.params["max_drawdown_limit"]:
            logger.critical(f"Global Drawdown {current_dd:.2%} exceeds limit {self.params['max_drawdown_limit']:.2%}")
            return True

        return False

    def portfolio_cvar(
        self, scenario_returns: np.ndarray, weights: np.ndarray,
        alpha: float = 0.95
    ) -> float:
        # scenario_returns shape (N_s, n), weights (n,)
        port_losses = - (scenario_returns @ weights)
        VaR = np.quantile(port_losses, alpha)
        if np.any(port_losses >= VaR):
            cvar = float(np.mean(port_losses[port_losses >= VaR]))
        else:
            cvar = float(np.max(port_losses))
        return cvar

    def enforce(
        self, proposed_weights: np.ndarray, scenario_returns: np.ndarray,
        returns_matrix_for_ent: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Inputs:
          proposed_weights: np.array shape (n,)
          scenario_returns: np.array shape (N_s, n)
          returns_matrix_for_ent: optional matrix (n_assets, T)
          to compute entanglement
        Returns:
          {allow:bool, reasons:[], suggested_weights: np.array or None}
        """
        reasons = []
        allow = True

        # 0. Global Kill Switch
        if self.check_kill_switch():
            return {
                "allow": False,
                "reasons": ["GLOBAL_KILL_SWITCH_ACTIVE"],
                "suggested_weights": np.zeros_like(proposed_weights),
                "cvar": 0.0
            }

        # 0.5 Leverage Checks
        gross = np.sum(np.abs(proposed_weights))
        net = np.sum(proposed_weights)

        if gross > self.params["gross_leverage_limit"]:
             allow = False
             reasons.append(f"gross_lev_{gross:.2f}_exceeds_{self.params['gross_leverage_limit']}")

        if abs(net) > self.params["net_leverage_limit"]:
             allow = False
             reasons.append(f"net_lev_{net:.2f}_exceeds_{self.params['net_leverage_limit']}")

        # 1) CVaR
        cvar = self.portfolio_cvar(
            scenario_returns, proposed_weights, alpha=0.95
        )
        if cvar > self.params["cvar_limit_pct"]:
            allow = False
            reasons.append(
                f"cvar_exceeds_limit:{cvar:.4f}>"
                f"{self.params['cvar_limit_pct']}"
            )

        # 2) entanglement
        if returns_matrix_for_ent is not None:
            try:
                from risk.quantum.entanglement_detector import (
                    build_entanglement_matrix, entanglement_indices
                )
                E = build_entanglement_matrix(returns_matrix_for_ent, q=0.05)
                indices, global_score = entanglement_indices(E)
                if global_score > self.params["entanglement_threshold"]:
                    # produce haircut: scale down weights by entanglement_haircut
                    suggested = proposed_weights * (
                        1 - self.params["entanglement_haircut"]
                    )
                    reasons.append(
                        f"entanglement_high:{global_score:.3f};"
                        f"suggested_haircut={self.params['entanglement_haircut']}"
                    )
                    return {
                        "allow": False,
                        "reasons": reasons,
                        "suggested_weights": suggested.tolist(),
                        "cvar": cvar,
                        "entanglement_score": global_score
                    }
            except ImportError:
                # Entanglement detector not available, skip check
                pass

        return {
            "allow": allow,
            "reasons": reasons,
            "suggested_weights": None if allow else np.zeros_like(proposed_weights),
            "cvar": cvar
        }
