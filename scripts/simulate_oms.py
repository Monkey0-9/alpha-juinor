
import logging
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.oms import OrderManager, OrderState
from risk.pretrade_checks import PreTradeRiskManager
from execution.engine import ExecutionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OMS_SIM")

def run_simulation():
    print("=== OMS & RISK SIMULATION ===")

    # Setup
    oms = OrderManager()
    risk = PreTradeRiskManager(max_order_notional=50000.0, max_pct_adv=0.01) # Strict limits
    engine = ExecutionEngine(order_manager=oms, risk_manager=risk)

    # 1. Good Order
    print("\n--- Test 1: Valid Order ---")
    o1 = engine.submit_order("AAPL", qty=10, side="buy", order_type="market")
    print(f"Order 1 State: {o1.state} (Expected: SUBMITTED)")

    # Simulate Fill
    oms.update_order_from_broker(o1.id, "filled", filled_qty=10, avg_price=150.0)
    print(f"Order 1 State after Fill: {o1.state} (Expected: FILLED)")

    # 2. Risk Reject (Notional)
    print("\n--- Test 2: Risk Reject (Notional) ---")
    # 1000 qty * $100 price = $100,000 > $50,000 limit
    o2 = engine.submit_order("GOOGL", qty=1000, side="buy")
    print(f"Order 2 State: {o2.state} (Expected: REJECTED)")
    print(f"Order 2 Reason: {o2.reason}")

    # 3. Risk Reject (ADV)
    print("\n--- Test 3: Risk Reject (ADV) ---")
    # 50,000 qty vs 1,000,000 ADV = 5%, Limit is 1%
    o3 = engine.submit_order("MSFT", qty=50000, side="sell")
    print(f"Order 3 State: {o3.state} (Expected: REJECTED)")

    print("\n=== SIMULATION COMPLETE ===")

if __name__ == "__main__":
    run_simulation()
