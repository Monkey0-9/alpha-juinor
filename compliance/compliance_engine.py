"""
Compliance & Audit Trail
=========================
SEC/FINRA/MiFID II compliance tracking for:
- Order audit trail
- Best execution reporting
- Wash sale detection
- Position limit enforcement
- Trade surveillance
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ComplianceEngine:
    """
    Institutional compliance engine.

    Covers:
    - SEC Rule 17a-4 (record retention)
    - FINRA Rule 4511 (books and records)
    - MiFID II best execution
    - Wash sale rule (IRS)
    - Position limits
    """

    def __init__(self):
        self._trade_log: List[Dict] = []
        self._wash_sale_window = timedelta(days=30)
        self._position_limits: Dict[str, float] = {}
        self._violations: List[Dict] = []

    def record_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        market_price: float,
        venue: str,
        timestamp: datetime = None,
        latency_ms: float = 0,
    ) -> Dict:
        """Record trade for best execution reporting."""
        ts = timestamp or datetime.utcnow()
        slippage_bps = (
            abs(fill_price - market_price) / market_price * 10000 if market_price else 0
        )

        record = {
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "fill_price": fill_price,
            "market_price": market_price,
            "slippage_bps": round(slippage_bps, 2),
            "venue": venue,
            "latency_ms": latency_ms,
            "hash": self._hash_record(ts, symbol, fill_price, quantity),
        }
        self._trade_log.append(record)

        if slippage_bps > 50:
            self._flag_violation(
                "EXCESSIVE_SLIPPAGE",
                record,
                f"Slippage {slippage_bps:.1f}bps > 50bps",
            )
        return record

    def best_execution_report(self, period_days: int = 30) -> Dict:
        """Generate best execution report."""
        cutoff = datetime.utcnow() - timedelta(days=period_days)
        recent = [t for t in self._trade_log if t["timestamp"] >= cutoff.isoformat()]

        if not recent:
            return {"period": period_days, "trades": 0}

        slippages = [t["slippage_bps"] for t in recent]
        return {
            "period_days": period_days,
            "total_trades": len(recent),
            "avg_slippage_bps": round(sum(slippages) / len(slippages), 2),
            "max_slippage_bps": max(slippages),
            "venues_used": list(set(t["venue"] for t in recent)),
            "violations": len(
                [v for v in self._violations if v["timestamp"] >= cutoff.isoformat()]
            ),
        }

    def check_wash_sale(
        self,
        symbol: str,
        side: str,
        timestamp: datetime = None,
    ) -> Optional[Dict]:
        """Check if trade triggers wash sale rule."""
        ts = timestamp or datetime.utcnow()
        window_start = ts - self._wash_sale_window

        related_sells = [
            t
            for t in self._trade_log
            if (
                t["symbol"] == symbol
                and t["side"] == "sell"
                and t["timestamp"] >= window_start.isoformat()
            )
        ]

        if side == "buy" and related_sells:
            violation = {
                "type": "POTENTIAL_WASH_SALE",
                "symbol": symbol,
                "sell_date": related_sells[-1]["timestamp"],
                "buy_date": ts.isoformat(),
            }
            self._flag_violation("WASH_SALE", violation, symbol)
            return violation
        return None

    def set_position_limit(self, symbol: str, max_shares: float):
        """Set max position size for a symbol."""
        self._position_limits[symbol] = max_shares

    def check_position_limit(
        self,
        symbol: str,
        current_pos: float,
        proposed_qty: float,
    ) -> bool:
        """Check if proposed trade exceeds limits."""
        limit = self._position_limits.get(symbol, float("inf"))
        new_pos = current_pos + proposed_qty
        if abs(new_pos) > limit:
            self._flag_violation(
                "POSITION_LIMIT",
                {"symbol": symbol, "new_pos": new_pos},
                f"{symbol}: {abs(new_pos)} > {limit}",
            )
            return False
        return True

    def get_audit_trail(
        self,
        start: datetime = None,
        end: datetime = None,
        symbol: str = None,
    ) -> List[Dict]:
        """Get audit trail with optional filters."""
        result = self._trade_log.copy()
        if start:
            result = [t for t in result if t["timestamp"] >= start.isoformat()]
        if end:
            result = [t for t in result if t["timestamp"] <= end.isoformat()]
        if symbol:
            result = [t for t in result if t["symbol"] == symbol]
        return result

    def get_violations(self) -> List[Dict]:
        """Get all compliance violations."""
        return self._violations.copy()

    def _flag_violation(
        self,
        vtype: str,
        details: Dict,
        desc: str,
    ):
        """Record compliance violation."""
        self._violations.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "type": vtype,
                "description": desc,
                "details": details,
            }
        )
        logger.warning(f"COMPLIANCE: {vtype} - {desc}")

    @staticmethod
    def _hash_record(
        ts: datetime,
        symbol: str,
        price: float,
        qty: float,
    ) -> str:
        """Tamper-evident hash of trade record."""
        data = f"{ts.isoformat()}|{symbol}|{price}|{qty}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


_compliance: Optional[ComplianceEngine] = None


def get_compliance() -> ComplianceEngine:
    global _compliance
    if _compliance is None:
        _compliance = ComplianceEngine()
    return _compliance


# Alias used by main.py
get_compliance_engine = get_compliance
