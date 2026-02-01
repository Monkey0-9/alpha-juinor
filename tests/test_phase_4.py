
import unittest
import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add project root
sys.path.insert(0, os.getcwd())

from services.risk_enforcer import RiskEnforcer
from regime.controller import RegimeController, RegimeLabel, RegimeState, RegimeOverrides

class TestPhase4(unittest.TestCase):

    def setUp(self):
        # Mock DB
        self.mock_db = MagicMock()
        self.mock_db.get_governance_settings.return_value = {"kill_switch_active": False}
        self.mock_db.get_pnl_stats.return_value = {"current_drawdown": -0.05}

    def test_risk_enforcer_kill_switch_active(self):
        """Test Kill Switch activation via DB."""
        enforcer = RiskEnforcer()
        enforcer.db = self.mock_db
        self.mock_db.get_governance_settings.return_value = {"kill_switch_active": True}

        result = enforcer.enforce(
            proposed_weights=np.array([0.1, 0.1]),
            scenario_returns=np.zeros((10, 2))
        )

        self.assertFalse(result["allow"])
        self.assertIn("GLOBAL_KILL_SWITCH_ACTIVE", result["reasons"])

    def test_risk_enforcer_drawdown_limit(self):
        """Test Kill Switch activation via Drawdown."""
        enforcer = RiskEnforcer(params={"max_drawdown_limit": -0.10})
        enforcer.db = self.mock_db
        self.mock_db.get_pnl_stats.return_value = {"current_drawdown": -0.15}

        result = enforcer.enforce(
             proposed_weights=np.array([0.1, 0.1]),
             scenario_returns=np.zeros((10, 2))
        )
        self.assertFalse(result["allow"])
        self.assertIn("GLOBAL_KILL_SWITCH_ACTIVE", result["reasons"])

    def test_risk_enforcer_cvar_check(self):
        """Test CVaR limits."""
        enforcer = RiskEnforcer(params={"cvar_limit_pct": 0.05})
        enforcer.db = self.mock_db

        # Scenario where asset 0 crashes 10%
        # Weights [1.0, 0] -> Portfolio loses 10%
        scenarios = np.zeros((100, 2))
        scenarios[:10, 0] = -0.10 # 10% loss in 10% of cases

        weights = np.array([1.0, 0.0])

        result = enforcer.enforce(
            proposed_weights=weights,
            scenario_returns=scenarios
        )

        self.assertFalse(result["allow"])
        # Should mention cvar
        self.assertTrue(any("cvar" in r for r in result["reasons"]))

    def test_regime_controller_integration(self):
        """Test Regime Controller basic flow."""
        controller = RegimeController(db_manager=self.mock_db)

        # Mock dependencies in detect_regime logic if needed
        # Or just test overrides logic which is deterministic
        overrides = RegimeOverrides.for_regime(RegimeLabel.CRISIS)
        self.assertEqual(overrides.execution_tactic, "HALT")

        overrides_ro = RegimeOverrides.for_regime(RegimeLabel.RISK_ON)
        self.assertEqual(overrides_ro.execution_tactic, "NORMAL")
        self.assertTrue(overrides_ro.new_positions_allowed)

if __name__ == "__main__":
    unittest.main()
