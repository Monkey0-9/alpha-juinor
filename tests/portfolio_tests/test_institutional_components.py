"""
tests/test_institutional_components.py

Integration test for Phase 9 components.
"""
import sys
import unittest
import logging
from unittest.mock import MagicMock

sys.path.insert(0, r'c:\mini-quant-fund')
from portfolio.opportunity_cost import OpportunityCostManager
from portfolio.capital_competition import CapitalCompetitionEngine
from risk.portfolio_stress_simulator import PortfolioStressSimulator
from risk.kill_switch import GlobalKillSwitch
from risk.regime_controller import RegimeController
from ui.terminal_dashboard import TerminalDashboard
from contracts.allocation import AllocationRequest
from contracts.alpha_contract import AlphaOutput

sys.path.insert(0, r'c:\mini-quant-fund')


class TestInstitutionalComponents(unittest.TestCase):
    def test_capital_competition(self):
        engine = CapitalCompetitionEngine()
        # Create AllocationRequests (mu, sigma, confidence, liquidity etc)
        c1 = AllocationRequest(
            symbol="AAPL", mu=0.001, sigma=0.02, confidence=1.0, cvar_95=-0.02,
            liquidity=1000000, regime="NORMAL", provider="test"
        )
        c2 = AllocationRequest(
            symbol="GOOG", mu=0.0005, sigma=0.02, confidence=1.0, cvar_95=-0.02,
            liquidity=1000000, regime="NORMAL", provider="test"
        )

        ranked = engine.run_competition([c1, c2], capital_budget=1.0)
        self.assertEqual(ranked[0].symbol, "AAPL")
        # Ensure decision is set
        self.assertIn(ranked[0].decision, ["ALLOCATE", "REJECT"])

    def test_stress_simulator(self):
        sim = PortfolioStressSimulator()
        # Safe Portfolio
        res = sim.simulate({"SPY": 0.5}, {}, {}, {}, {}, 1e6)
        self.assertTrue(res.passed)

        # Crash Portfolio
        res = sim.simulate({"SPY": 3.0}, {}, {}, {}, {}, 1e6) # High leverage
        self.assertFalse(res.passed)

    def test_regime_loading(self):
        ctrl = RegimeController()
        # Should load defaults from yaml
        overrides = ctrl.get_current_overrides()
        self.assertIsInstance(overrides, dict)

    def test_kill_switch(self):
        ks = GlobalKillSwitch()
        # Mock Agent
        agent = MagicMock()
        agent.dashboard = TerminalDashboard()
        agent.dashboard.state['regime']['confidence'] = 0.1 # Fail

        is_safe, reason = ks.verify_safety(agent)
        self.assertFalse(is_safe)
        self.assertIn("Regime Confidence", reason)

    def test_alpha_contract(self):
        ao = AlphaOutput(
            mu=0.01, sigma=0.01, cvar_95=-0.02, confidence=0.8,
            provider="test", model_version="1.0", input_schema_hash="abc",
            distribution_type="NORMAL", model_disagreement=0.5
        )
        valid, errs = ao.validate()
        self.assertTrue(valid, f"Validation failed: {errs}")

    def test_opportunity_cost_manager(self):
        ocm = OpportunityCostManager(
            risk_free_rate=0.04, time_horizon=1.0
        )
        # Mock some alpha outputs
        alpha_outputs = [
            AlphaOutput(
                mu=0.01, sigma=0.01, cvar_95=-0.02, confidence=0.8,
                provider="test", model_version="1.0", input_schema_hash="abc",
                distribution_type="NORMAL", model_disagreement=0.5
            ),
            AlphaOutput(
                mu=0.005, sigma=0.005, cvar_95=-0.01, confidence=0.9,
                provider="test", model_version="1.0", input_schema_hash="def",
                distribution_type="NORMAL", model_disagreement=0.2
            )
        ]
        results = ocm.calculate_opportunity_costs(alpha_outputs)
        self.assertIsInstance(results, dict)
        self.assertIn('weighted_cost', results)
        # Validate fields
        self.assertGreater(
            results['weighted_cost'], 0, "Weighted cost should be positive"
        )



if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    unittest.main()
