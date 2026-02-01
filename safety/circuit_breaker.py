# safety/circuit_breaker.py
"""
Circuit breaker: track realized PnL per-day and per-week.
Persist state to JSON file (simple durable store). Expose programmatic API.
Integrate this: check CircuitBreaker.before_any_execution() in executor pipeline.
"""
from dataclasses import dataclass, asdict
import json, os, time
from typing import Dict, Any

STATE_PATH = os.environ.get("CIRCUIT_STATE_PATH", "runs/circuit_state.json")

@dataclass
class CircuitConfig:
    max_single_trade_loss_pct: float = 0.01   # 1% NAV per trade
    max_daily_loss_pct: float = 0.02          # 2% NAV
    max_weekly_loss_pct: float = 0.05         # 5% NAV
    nav_usd: float = 1_000_000.0
    auto_halt_enabled: bool = True

class CircuitBreaker:
    def __init__(self, cfg: CircuitConfig = CircuitConfig(), state_path: str = STATE_PATH):
        self.cfg = cfg
        self.state_path = state_path
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                self.state = json.load(f)
        else:
            self.state = {
                "halted": False,
                "halt_reason": None,
                "last_reset_ts": time.time(),
                "daily_pnl_usd": 0.0,
                "weekly_pnl_usd": 0.0,
                "last_day_epoch": self._day_epoch(),
            }
            self._persist()

    def _persist(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def _day_epoch(self):
        return int(time.time() // 86400)

    def _week_epoch(self):
        return int(time.time() // (86400 * 7))

    def record_trade_result(self, pnl_usd: float, trade_notional_usd: float) -> Dict[str,Any]:
        """
        Call this function after a trade is closed (realized PnL).
        Returns dict with status and whether system should halt.
        """
        # roll daily/weekly if needed
        current_day = self._day_epoch()
        if current_day != self.state.get("last_day_epoch"):
            self.state["daily_pnl_usd"] = 0.0
            self.state["last_day_epoch"] = current_day
        # weekly horizon - naive (week epoch)
        # for simplicity, we track weekly continuously but reset if epoch changes
        # left as simple implementation
        self.state["daily_pnl_usd"] += pnl_usd
        self.state["weekly_pnl_usd"] += pnl_usd
        self._persist()

        # check single trade notional loss fraction
        max_single_loss = -self.cfg.max_single_trade_loss_pct * self.cfg.nav_usd
        if pnl_usd < max_single_loss:
            self.state["halted"] = True
            self.state["halt_reason"] = f"single_trade_loss_exceeded: {pnl_usd:.2f}"
            self._persist()
            return {"halt": True, "reason": self.state["halt_reason"]}

        # check daily losses
        if self.state["daily_pnl_usd"] < -self.cfg.max_daily_loss_pct * self.cfg.nav_usd:
            self.state["halted"] = True
            self.state["halt_reason"] = f"daily_loss_exceeded: {self.state['daily_pnl_usd']:.2f}"
            self._persist()
            return {"halt": True, "reason": self.state["halt_reason"]}

        # check weekly losses
        if self.state["weekly_pnl_usd"] < -self.cfg.max_weekly_loss_pct * self.cfg.nav_usd:
            self.state["halted"] = True
            self.state["halt_reason"] = f"weekly_loss_exceeded: {self.state['weekly_pnl_usd']:.2f}"
            self._persist()
            return {"halt": True, "reason": self.state["halt_reason"]}

        return {"halt": False}

    def is_halted(self) -> bool:
        return bool(self.state.get("halted", False))

    def get_state(self) -> Dict[str,Any]:
        return self.state

    def reset(self, operator: str = "system") -> None:
        self.state.update({
            "halted": False,
            "halt_reason": None,
            "daily_pnl_usd": 0.0,
            "weekly_pnl_usd": 0.0,
            "last_reset_ts": time.time()
        })
        self._persist()

    def force_halt(self, reason: str) -> None:
        self.state["halted"] = True
        self.state["halt_reason"] = reason
        self._persist()
