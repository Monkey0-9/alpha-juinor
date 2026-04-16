# ops/kill_switch.py
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class KillSwitch:
    """
    Automated Safety Circuit Breaker.
    Monitors for catastrophic breaches and enforces systematic halting.
    """
    
    def __init__(self, config: Dict[str, Any], state_path: str = "output/ops/kill_state.json"):
        self.max_dd = config.get('max_total_drawdown_limit', 0.15)
        self.max_daily_loss = config.get('max_daily_drawdown_limit', 0.03)
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._is_active = False
        
    def check_breach(self, current_equity: float, total_peak_equity: float, daily_start_equity: float) -> bool:
        """
        Returns True if a breach occurred.
        """
        if total_peak_equity <= 0 or daily_start_equity <= 0:
            return False
            
        total_dd = (current_equity - total_peak_equity) / total_peak_equity
        daily_loss = (current_equity - daily_start_equity) / daily_start_equity
        
        reasons = []
        if abs(total_dd) > self.max_dd:
            reasons.append(f"TOTAL DRAWDOWN BREACH: {total_dd:.2%}")
            
        if abs(daily_loss) > self.max_daily_loss:
            reasons.append(f"DAILY LOSS BREACH: {daily_loss:.2%}")
            
        if reasons:
            self._activate(reasons)
            return True
            
        return False

    def _activate(self, reasons: list):
        self._is_active = True
        msg = f"!!! KILL SWITCH ACTIVATED !!! Reasons: {', '.join(reasons)}"
        logger.critical(msg)
        
        # Persist state so system won't restart without manual intervention
        with open(self.state_path, "w") as f:
            json.dump({
                "active": True,
                "reasons": reasons,
                "timestamp": str(pd.Timestamp.now())
            }, f)

    def is_locked(self) -> bool:
        """Check if system is in a locked/halted state."""
        if self.state_path.exists():
            with open(self.state_path, "r") as f:
                state = json.load(f)
                return state.get("active", False)
        return False

    def clear_lock(self):
        """MANUAL OVERRIDE ONLY."""
        if self.state_path.exists():
            self.state_path.unlink()
            logger.info("Kill Switch lock cleared manually.")
