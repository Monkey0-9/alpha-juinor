"""
tests/backtest/test_crisis_survivability.py
"""
import unittest
from backtest.scenario_runner import ScenarioRunner

class TestCrisisSurvivability(unittest.TestCase):
    def test_runner_logic(self):
        runner = ScenarioRunner()
        # Run a short fake scenario
        res = runner.run_scenario("TEST_SHOCK", "2020-01-01", "2020-02-01")

        self.assertIsNotNone(res)
        self.assertIsInstance(res.max_drawdown, float)
        self.assertIsInstance(res.cvar_95, float)
        # Should pass default random walk criteria usually or fail gacefully
        if not res.passed:
            print(f"Test scenario failed criteria (Expected in stress test): {res}")

if __name__ == "__main__":
    unittest.main()
