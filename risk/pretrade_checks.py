
import logging

logger = logging.getLogger("RiskCheck")

class PreTradeRiskManager:
    """
    Enforces pre-trade risk limits mandated by institutional policy.
    Blockers:
    - Max Notional per Order
    - Max % of ADV
    - Fat Finger (Price Deviation)
    """

    def __init__(self, max_order_notional=100000.0, max_pct_adv=0.01):
        self.MAX_ORDER_NOTIONAL = max_order_notional
        self.MAX_PCT_ADV = max_pct_adv
        self.FAT_FINGER_THRESHOLD = 0.10 # 10% deviation

    def check(self, order, market_price: float, adv: float) -> bool:
        """
        Run all checks. Returns True if PASSED, False if REJECTED.
        Mutates order.reason if rejected.
        """
        symbol = order.symbol
        qty = abs(order.qty)
        notional = qty * market_price

        # 1. Notional Check
        if notional > self.MAX_ORDER_NOTIONAL:
            reason = f"Risk Reject: Notional ${notional:,.2f} > ${self.MAX_ORDER_NOTIONAL:,.2f}"
            logger.error(f"[RISK] {symbol} {reason}")
            order.transition(order.state.REJECTED, reason=reason)
            return False

        # 2. ADV Check
        if adv > 0:
            pct_adv = qty / adv
            if pct_adv > self.MAX_PCT_ADV:
                reason = f"Risk Reject: Size {qty} is {pct_adv:.2%} of ADV (Max {self.MAX_PCT_ADV:.1%})"
                logger.error(f"[RISK] {symbol} {reason}")
                order.transition(order.state.REJECTED, reason=reason)
                return False
        else:
            # No ADV data - Reject or Warn?
            # Institutional: Reject if liquidity unknown
            reason = "Risk Reject: Unknown ADV"
            logger.error(f"[RISK] {symbol} {reason}")
            order.transition(order.state.REJECTED, reason=reason)
            return False

        # 3. Fat Finger / Price Deviation check (if limit order)
        if order.limit_price:
            dev = abs(order.limit_price - market_price) / market_price
            if dev > self.FAT_FINGER_THRESHOLD:
                reason = f"Risk Reject: Limit {order.limit_price} deviates {dev:.1%} from market {market_price}"
                logger.error(f"[RISK] {symbol} {reason}")
                order.transition(order.state.REJECTED, reason=reason)
                return False

        return True
