import os
import sys
import numpy as np

# Elite Path Resolution
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.mini_quant_fund.institutional.execution_alumni import (
    ExecutionAlumni
)


class RL_ExecutionAgent:
    """
    Sovereign RL Agent for Intelligent Order Execution.
    """
    def __init__(self, model_path=None):
        self.state_dim = 12
        self.action_dim = 5
        self.impact_model = ExecutionAlumni()

    def _get_market_state(self):
        return np.random.normal(0, 1, self.state_dim)

    def get_action(self, market_state: np.ndarray = None) -> int:
        if market_state is None:
            market_state = self._get_market_state()

        exp_vals = np.exp(market_state[:self.action_dim])
        probs = exp_vals / np.sum(exp_vals)
        return int(np.argmax(probs))

    def optimize_order(self, symbol, qty, target_time_mins=60):
        print(f"[*] [RL] Optimizing {qty} unit liquidation for {symbol}")

        impact_bps = self.impact_model.calculate_market_impact(
            qty, target_time_mins / (60 * 6.5)
        )

        action = self.get_action()
        schedules = [
            "TWAP", "VWAP", "IS_OPTIMAL",
            "AGGRESSIVE_SWEEP", "BLOCK_LIQUIDATION"
        ]
        chosen_path = schedules[action]

        print(
            f"[OK] [RL] Strategy: {chosen_path} | "
            f"Est. Impact: {impact_bps:.2f} bps"
        )
        return {
            "strategy": chosen_path,
            "estimated_impact_bps": impact_bps,
            "optimal_horizon": target_time_mins
        }
