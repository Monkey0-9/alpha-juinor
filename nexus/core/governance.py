import datetime
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class GovernanceEngine:
    """
    Governance Gate for institutional-grade trading compliance.
    """
    def __init__(self, single_position_limit: float = 0.05, max_drawdown_limit: float = 0.15):
        self.single_position_limit = single_position_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.audit_log: List[Dict] = []

    def check_compliance(self, trade_request: Dict, portfolio_state: Dict) -> Tuple[bool, List[str]]:
        """
        Performs compliance checks against risk and concentration limits.
        """
        symbol = trade_request['symbol']
        qty = trade_request['qty']
        price = trade_request['price']
        trade_value = qty * price
        total_value = portfolio_state['total_value']
        drawdown = portfolio_state['drawdown']

        violations = []

        # 1. Concentration Check
        if total_value > 0:
            pos_pct = trade_value / total_value
            if pos_pct > self.single_position_limit:
                violations.append(
                    f"POSITION_CONCENTRATION: {pos_pct:.2%} > {self.single_position_limit:.1%}"
                )

        # 2. Drawdown Protection
        if drawdown > self.max_drawdown_limit:
            violations.append(
                f"DRAWDOWN_BREACH: {drawdown:.1%} > {self.max_drawdown_limit:.1%}"
            )

        # 3. Symbol Blacklist
        blacklist = ["ZOMBIE", "SCAM_COIN", "DANGER_TKR"]
        if symbol in blacklist:
            violations.append(f"BLACKLIST_SYMBOL: {symbol} is restricted")

        if not violations:
            self._log_audit(trade_request, "APPROVED")
            return True, []
        
        self._log_audit(trade_request, "REJECTED", violations)
        return False, violations

    def _log_audit(self, request: Dict, status: str, details: Optional[List[str]] = None):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "symbol": request['symbol'],
            "side": request['side'],
            "status": status,
            "details": details or "Compliance Passed"
        }
        self.audit_log.append(entry)
        msg = f"AUDIT: {status} - {request['side']} {request['symbol']} - {entry['details']}"
        logger.info(msg)
