# governance/do_not_trade.py
"""
Do-Not-Trade rule engine. Centralized function to decide whether system should pause new trading.
Checks: regime_confidence, recent_hit_rate, replay_mismatch, circuit breaker/halt.
Return True to allow trading or False to block.
"""
from typing import Dict, Any
from safety.kill_switch import KillSwitch
from safety.circuit_breaker import CircuitBreaker, CircuitConfig

def allow_trading(system_metrics: Dict[str,Any]) -> Dict[str,Any]:
    """
    system_metrics sample:
      {
        "regime_confidence":0.6,
        "recent_hit_rate":0.45,
        "replay_mismatch_rate":0.0,
        "manual_disable":False
      }
    """
    kill = KillSwitch()
    if kill.is_engaged():
        return {"allow": False, "reason": "kill_switch_engaged"}

    cb = CircuitBreaker(CircuitConfig(nav_usd=system_metrics.get("nav_usd",1e6)))
    if cb.is_halted():
        return {"allow": False, "reason": "circuit_breaker_halted"}

    # regime confidence
    if system_metrics.get("regime_confidence", 1.0) < 0.4:
        return {"allow": False, "reason": "low_regime_confidence"}

    if system_metrics.get("recent_hit_rate", 1.0) < 0.3:
        return {"allow": False, "reason": "recent_hit_rate_low"}

    # replay mismatch
    if system_metrics.get("replay_mismatch_rate", 0.0) > 0.0:
        return {"allow": False, "reason": "replay_mismatch"}

    if system_metrics.get("manual_disable", False):
        return {"allow": False, "reason": "manual_disable_flag"}

    return {"allow": True}
