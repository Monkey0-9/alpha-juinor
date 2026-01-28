"""
Integration note: Stress Gate Hookup

TO COMPLETE STRESS GATE INTEGRATION:

In execution/handler.py (or wherever orders are executed):

```python
from execution.stress_gate import StressSimulationGate

class ExecutionHandler:
    def __init__(self):
        self.stress_gate = StressSimulationGate(max_stress_loss=0.10)

    def execute_orders(self, orders, current_portfolio, alphas):
        # Convert orders to proposed trades Dict[symbol, weight]
        proposed_trades = {o['symbol']: o['quantity'] for o in orders}

        # RUN STRESS TEST (Elite-Tier)
        stress_result = self.stress_gate.stress_test_portfolio(
            proposed_trades,
            current_portfolio,
            alphas
        )

        if not stress_result["pass"]:
            logger.error(
                f"[STRESS_GATE] Portfolio FAILED stress test: "
                f"{stress_result['worst_scenario']} = {stress_result['max_drawdown']:.2%}"
            )

            # Scale down positions to pass
            scale_factor = self.stress_gate.get_scale_down_factor(
                stress_result["max_drawdown"]
            )

            for order in orders:
                order['quantity'] *= scale_factor

            logger.warning(f"[STRESS_GATE] Scaled positions by {scale_factor:.2f}")

        # Proceed with execution
        return self._execute_scaled_orders(orders)
```

This completes the stress-first mentality.
"""
