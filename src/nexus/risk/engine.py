from typing import List, Optional
from ..models.trade import Order, PortfolioState
from .rules import RiskRule
from ..core.context import engine_context

class RiskEngine:
    """
    Centralized risk management engine.
    Validates orders against pre-trade rules and monitors portfolio against post-trade rules.
    """
    def __init__(self):
        self.pre_trade_rules: List[RiskRule] = []
        self.post_trade_rules: List[RiskRule] = []
        self.logger = engine_context.get_logger("risk_engine")
        self.is_halted = False

    def add_pre_trade_rule(self, rule: RiskRule):
        self.pre_trade_rules.append(rule)

    def add_post_trade_rule(self, rule: RiskRule):
        self.post_trade_rules.append(rule)

    def check_pre_trade(self, order: Order, portfolio: PortfolioState) -> bool:
        """Runs all pre-trade risk checks."""
        if self.is_halted:
            self.logger.fatal("Risk check failed: SYSTEM IS HALTED")
            return False

        for rule in self.pre_trade_rules:
            if not rule.validate(order, portfolio):
                self.logger.error(f"Pre-trade risk violation: {rule.get_reason()}")
                return False
        return True

    def check_post_trade(self, portfolio: PortfolioState) -> bool:
        """Runs all post-trade risk monitoring checks."""
        for rule in self.post_trade_rules:
            if not rule.validate(None, portfolio):
                self.logger.fatal(f"CRITICAL POST-TRADE RISK VIOLATION: {rule.get_reason()}")
                self.halt_system()
                return False
        return True

    def halt_system(self):
        """Immediately halts the entire engine via a global kill-switch."""
        self.is_halted = True
        self.logger.fatal("!!! RISK ENGINE INITIATED GLOBAL SYSTEM HALT !!!")
        # In a real system, this would broadcast a signal to cancel all open orders
        engine_context.set_running(False)
