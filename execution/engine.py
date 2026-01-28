
import logging
from typing import List, Dict, Optional
from execution.oms import OrderManager, OrderState, Order
from risk.pretrade_checks import PreTradeRiskManager
from risk.market_impact_models import MarketImpactModel
from data.collectors.alpaca_collector import AlpacaDataProvider # Assuming Execution Handler exists or using this for now

logger = logging.getLogger("ExecutionEngine")

class ExecutionEngine:
    """
    Orchestrates the execution of trading decisions.
    Flow:
    1. Receive Decision (Target PF) -> Generate Orders
    2. Pre-Trade Risk Check
    3. Order Creation (OMS)
    4. Routing to Broker
    5. Monitoring & Reconciling
    """

    def __init__(self, broker_client=None, risk_manager=None, order_manager=None):
        self.oms = order_manager or OrderManager()
        self.risk = risk_manager or PreTradeRiskManager()
        self.broker = broker_client
        self.impact_model = MarketImpactModel()

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = "market", limit_price: float = None, cycle_id=None) -> Optional[Order]:
        """
        Submit a new order to the engine.
        Returns the Order object if accepted (created), or None if rejected immediately.
        """
        # 1. Create Order in OMS (New State)
        order = self.oms.create_order(symbol, qty, side, order_type, cycle_id)
        if limit_price:
            order.limit_price = limit_price

        # 2. Pre-Trade Risk Checks
        # Need current market data for risk check
        # For now, fetching via broker or assuming passed in?
        # Ideally engines have access to DataRouter. Here we mock or fetch if broker avail.
        market_price = self._get_market_price(symbol)
        adv = self._get_adv(symbol) # Placeholder

        if not self.risk.check(order, market_price, adv):
            # Order rejected by risk, state is already updated to REJECTED by risk check
            return order

        # 3. Transition to Pending Submit
        order.transition(OrderState.PENDING_SUBMIT)

        # 4. Route to Broker
        try:
            # broker_id = self.broker.submit(order)
            # Mocking broker submission for now
            broker_id = f"ext-{order.id[:8]}"
            order.external_id = broker_id
            order.transition(OrderState.SUBMITTED, reason=f"Routed to Broker {broker_id}")
            logger.info(f"[EXEC] Order {order.id} submitted for {symbol} {side} {qty}")
        except Exception as e:
            logger.error(f"[EXEC] Broker submission failed: {e}")
            order.transition(OrderState.REJECTED, reason=f"Broker Error: {e}")

        return order

    def _get_market_price(self, symbol):
        # Todo: Integrate DataRouter
        return 100.0

    def _get_adv(self, symbol):
        # Todo: Integrate DataRouter/Features
        return 1000000.0

    def cancel_all_for_cycle(self, cycle_id: str):
        """Cancel all open orders for a cycle."""
        active = [o for o in self.oms.get_active_orders() if o.cycle_id == cycle_id]
        for order in active:
            # self.broker.cancel(order.external_id)
            order.transition(OrderState.CANCELED, reason="Ops Cancel")
