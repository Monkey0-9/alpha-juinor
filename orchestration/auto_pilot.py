"""
Auto Pilot - Full Trading Automation Controller.

Manages all automated trading actions:
- AUTO BUY: When signal + PM Score + Risk pass
- AUTO SELL: Stop loss, target hit, signal reversal
- AUTO HOLD: No actionable signal
- AUTO REBALANCE: Drift exceeds threshold

Safety interlocks prevent dangerous actions.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AutoAction(Enum):
    """Automated action types."""
    BUY = "AUTO_BUY"
    SELL = "AUTO_SELL"
    HOLD = "AUTO_HOLD"
    REBALANCE = "AUTO_REBALANCE"
    STOP_LOSS = "AUTO_STOP_LOSS"
    TAKE_PROFIT = "AUTO_TAKE_PROFIT"
    WAIT = "AUTO_WAIT"


@dataclass
class AutoDecision:
    """Record of an automated decision."""
    timestamp: datetime
    symbol: str
    action: AutoAction
    quantity: float
    price: Optional[float]
    reason: str
    approved: bool
    executed: bool = False
    execution_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position tracking."""
    symbol: str
    quantity: float
    avg_entry: float
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: Optional[datetime] = None


class AutoPilot:
    """
    Central automation controller for the trading system.

    Features:
    - Manages buy/sell/hold decisions
    - Enforces safety limits
    - Logs all automated actions for audit
    - Graceful degradation on errors
    """

    def __init__(
        self,
        max_position_size: float = 0.10,
        max_daily_trades: int = 50,
        min_confidence: float = 0.60,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        rebalance_threshold: float = 0.05,
        enabled: bool = True
    ):
        self.max_position_size = max_position_size
        self.max_daily_trades = max_daily_trades
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.rebalance_threshold = rebalance_threshold
        self.enabled = enabled

        self.positions: Dict[str, Position] = {}
        self.daily_trades = 0
        self.last_trade_reset = datetime.utcnow()
        self.decision_log: List[AutoDecision] = []
        self.paused = False
        self.pause_until: Optional[datetime] = None

    def _reset_daily_counter(self):
        """Reset daily trade counter if new day."""
        now = datetime.utcnow()
        if (now - self.last_trade_reset).days >= 1:
            self.daily_trades = 0
            self.last_trade_reset = now
            logger.info("AutoPilot: Daily trade counter reset")

    def _check_pause(self) -> bool:
        """Check if autopilot is paused."""
        if self.pause_until and datetime.utcnow() < self.pause_until:
            return True
        self.paused = False
        self.pause_until = None
        return False

    def pause(self, minutes: int = 60):
        """Pause autopilot for specified minutes."""
        self.paused = True
        self.pause_until = datetime.utcnow() + timedelta(minutes=minutes)
        logger.warning(f"AutoPilot: Paused for {minutes} minutes")

    def resume(self):
        """Resume autopilot."""
        self.paused = False
        self.pause_until = None
        logger.info("AutoPilot: Resumed")

    def update_position(self, pos: Position):
        """Update position tracking."""
        self.positions[pos.symbol] = pos

    def remove_position(self, symbol: str):
        """Remove a position (after sell)."""
        self.positions.pop(symbol, None)

    def check_stop_loss(self, symbol: str) -> Optional[AutoDecision]:
        """Check if stop loss triggered."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        if pos.stop_loss and pos.current_price <= pos.stop_loss:
            decision = AutoDecision(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                action=AutoAction.STOP_LOSS,
                quantity=pos.quantity,
                price=pos.current_price,
                reason=f"Stop loss triggered: {pos.current_price:.2f} <= {pos.stop_loss:.2f}",
                approved=True,
                metadata={"pnl": pos.unrealized_pnl}
            )
            self._log_decision(decision)
            return decision
        return None

    def check_take_profit(self, symbol: str) -> Optional[AutoDecision]:
        """Check if take profit triggered."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        if pos.take_profit and pos.current_price >= pos.take_profit:
            decision = AutoDecision(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                action=AutoAction.TAKE_PROFIT,
                quantity=pos.quantity,
                price=pos.current_price,
                reason=f"Take profit triggered: {pos.current_price:.2f} >= {pos.take_profit:.2f}",
                approved=True,
                metadata={"pnl": pos.unrealized_pnl}
            )
            self._log_decision(decision)
            return decision
        return None

    def decide_action(
        self,
        symbol: str,
        signal_direction: int,
        signal_strength: float,
        signal_confidence: float,
        current_price: float,
        pm_score: float = 0.5,
        risk_passed: bool = True,
        portfolio_value: float = 100000
    ) -> AutoDecision:
        """
        Decide automated action based on signal and context.

        Returns: AutoDecision with approved=True if action should be taken.
        """
        self._reset_daily_counter()

        # Safety checks
        if not self.enabled:
            return self._create_wait("AutoPilot disabled")

        if self._check_pause():
            return self._create_wait("AutoPilot paused")

        if self.daily_trades >= self.max_daily_trades:
            return self._create_wait("Daily trade limit reached")

        if signal_confidence < self.min_confidence:
            return AutoDecision(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                action=AutoAction.HOLD,
                quantity=0,
                price=current_price,
                reason=f"Low confidence: {signal_confidence:.2f} < {self.min_confidence}",
                approved=False
            )

        if not risk_passed:
            return self._create_wait(f"{symbol}: Risk check failed")

        # Determine action
        in_position = symbol in self.positions

        if signal_direction > 0 and not in_position:
            # BUY Signal
            size = self._calculate_position_size(
                signal_strength, pm_score, portfolio_value, current_price
            )
            if size > 0:
                decision = AutoDecision(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    action=AutoAction.BUY,
                    quantity=size,
                    price=current_price,
                    reason=f"Long signal: str={signal_strength:.2f}, pm={pm_score:.2f}",
                    approved=True,
                    metadata={
                        "stop_loss": current_price * (1 - self.stop_loss_pct),
                        "take_profit": current_price * (1 + self.take_profit_pct)
                    }
                )
                self._log_decision(decision)
                return decision

        elif signal_direction < 0 and in_position:
            # SELL Signal (exit long)
            pos = self.positions[symbol]
            decision = AutoDecision(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                action=AutoAction.SELL,
                quantity=pos.quantity,
                price=current_price,
                reason=f"Short signal reversal: str={signal_strength:.2f}",
                approved=True,
                metadata={"pnl": pos.unrealized_pnl}
            )
            self._log_decision(decision)
            return decision

        elif signal_direction == 0:
            # HOLD or WAIT
            if in_position:
                # Check exits
                stop = self.check_stop_loss(symbol)
                if stop:
                    return stop
                tp = self.check_take_profit(symbol)
                if tp:
                    return tp
                return AutoDecision(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    action=AutoAction.HOLD,
                    quantity=0,
                    price=current_price,
                    reason="Neutral signal, holding position",
                    approved=False
                )
            else:
                return self._create_wait(f"{symbol}: No signal, waiting")

        return AutoDecision(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            action=AutoAction.HOLD,
            quantity=0,
            price=current_price,
            reason="No actionable signal",
            approved=False
        )

    def _calculate_position_size(
        self,
        strength: float,
        pm_score: float,
        portfolio_value: float,
        price: float
    ) -> float:
        """Calculate position size based on signal strength and PM score."""
        # Base allocation capped by max_position_size
        conviction = (strength + pm_score) / 2
        alloc_pct = min(conviction * self.max_position_size * 2, self.max_position_size)

        dollar_amount = portfolio_value * alloc_pct
        shares = int(dollar_amount / price) if price > 0 else 0
        return float(shares)

    def _create_wait(self, reason: str) -> AutoDecision:
        """Create a WAIT decision."""
        return AutoDecision(
            timestamp=datetime.utcnow(),
            symbol="",
            action=AutoAction.WAIT,
            quantity=0,
            price=None,
            reason=reason,
            approved=False
        )

    def _log_decision(self, decision: AutoDecision):
        """Log decision for audit trail."""
        self.decision_log.append(decision)
        if decision.approved:
            self.daily_trades += 1

        logger.info(
            f"AutoPilot: {decision.action.value} {decision.symbol} "
            f"qty={decision.quantity} @ {decision.price} | {decision.reason}"
        )

    def check_rebalance(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> List[AutoDecision]:
        """
        Check if rebalancing is needed.

        Returns list of rebalance decisions.
        """
        decisions = []

        for symbol, target in target_weights.items():
            current = current_weights.get(symbol, 0.0)
            drift = abs(target - current)

            if drift > self.rebalance_threshold:
                action = AutoAction.REBALANCE
                qty_change = target - current  # Positive = buy more

                decision = AutoDecision(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    action=action,
                    quantity=abs(qty_change),
                    price=None,
                    reason=f"Drift {drift:.1%} > threshold {self.rebalance_threshold:.1%}",
                    approved=True,
                    metadata={"target": target, "current": current}
                )
                self._log_decision(decision)
                decisions.append(decision)

        return decisions

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get recent decisions for audit."""
        recent = self.decision_log[-limit:]
        return [
            {
                "timestamp": d.timestamp.isoformat(),
                "symbol": d.symbol,
                "action": d.action.value,
                "quantity": d.quantity,
                "price": d.price,
                "reason": d.reason,
                "approved": d.approved,
                "executed": d.executed
            }
            for d in recent
        ]

    def get_status(self) -> Dict:
        """Get autopilot status."""
        return {
            "enabled": self.enabled,
            "paused": self.paused,
            "pause_until": self.pause_until.isoformat() if self.pause_until else None,
            "daily_trades": self.daily_trades,
            "max_daily_trades": self.max_daily_trades,
            "positions_count": len(self.positions),
            "total_decisions": len(self.decision_log)
        }


# Global instance
_autopilot: Optional[AutoPilot] = None


def get_autopilot() -> AutoPilot:
    """Get or create global AutoPilot instance."""
    global _autopilot
    if _autopilot is None:
        _autopilot = AutoPilot()
    return _autopilot
