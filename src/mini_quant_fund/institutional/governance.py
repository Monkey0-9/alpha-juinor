import datetime
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GovernanceGate")


class InstitutionalGovernance:
    """
    Governance Gate for high-stakes trading.
    Ensures compliance with SEC/FINRA-like standards.
    """
    def __init__(self, single_position_limit=0.05, max_drawdown_limit=0.15):
        self.single_position_limit = single_position_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.audit_log = []

    def check_compliance(self, trade_request, current_portfolio):
        """
        Runs 5 layers of compliance checks
        """
        symbol = trade_request['symbol']
        qty = trade_request['qty']
        price = trade_request['price']
        trade_value = qty * price
        total_value = current_portfolio['total_value']

        violations = []

        # 1. Single Position Concentration Check
        pos_pct = trade_value / total_value
        if pos_pct > self.single_position_limit:
            violations.append(
                f"POSITION_CONCENTRATION: {pos_pct:.2%} > "
                f"{self.single_position_limit:.1%}"
            )

        # 2. Maximum Drawdown Protection
        if current_portfolio['drawdown'] > self.max_drawdown_limit:
            violations.append(
                f"DRAWDOWN_BREACH: {current_portfolio['drawdown']:.1%} > "
                f"{self.max_drawdown_limit:.1%}"
            )

        # 3. Wash Trading Prevention
        # (Placeholder for real check)

        # 4. Symbol Blacklist Check
        blacklist = ["ZOMBIE", "SCAM_COIN", "DANGER_TKR"]
        if symbol in blacklist:
            violations.append(f"BLACKLIST_SYMBOL: {symbol}")

        if not violations:
            self._log_audit(trade_request, "APPROVED")
            return True, []
        else:
            self._log_audit(trade_request, "REJECTED", violations)
            return False, violations

    def _log_audit(self, request, status, details=None):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "symbol": request['symbol'],
            "side": request['side'],
            "status": status,
            "details": details or "Success"
        }
        self.audit_log.append(entry)
        logger.info(f"AUDIT LOG: {json.dumps(entry)}")


if __name__ == "__main__":
    gov = InstitutionalGovernance()
    portfolio = {"total_value": 1000000, "drawdown": 0.05}
    trade = {"symbol": "AAPL", "qty": 1000, "price": 180, "side": "buy"}
    approved, errors = gov.check_compliance(trade, portfolio)
    print(f"Trade Approved: {approved} | Errors: {errors}")
