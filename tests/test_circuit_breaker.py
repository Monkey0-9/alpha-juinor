# tests/test_circuit_breaker.py
import os, tempfile
from safety.circuit_breaker import CircuitBreaker, CircuitConfig

def test_circuit_breaker_triggers(tmp_path):
    p = tmp_path / "state.json"
    cfg = CircuitConfig(max_single_trade_loss_pct=0.01, max_daily_loss_pct=0.02, max_weekly_loss_pct=0.05, nav_usd=100000.0)
    cb = CircuitBreaker(cfg=cfg, state_path=str(p))
    # small profitable trade
    r = cb.record_trade_result(100.0, 1000.0)
    assert not r["halt"]
    # large loss single trade
    r2 = cb.record_trade_result(-2000.0, 20000.0)
    assert r2["halt"]
    assert cb.is_halted()

def test_circuit_breaker_daily_limit(tmp_path):
    p = tmp_path / "state2.json"
    cfg = CircuitConfig(max_single_trade_loss_pct=0.01, max_daily_loss_pct=0.02, nav_usd=100000.0)
    cb = CircuitBreaker(cfg=cfg, state_path=str(p))
    # accumulate losses under single trade limit but hit daily
    cb.record_trade_result(-500.0, 5000.0)  # 0.5% loss
    cb.record_trade_result(-500.0, 5000.0)  # cumulative 1%
    cb.record_trade_result(-600.0, 6000.0)  # cumulative 1.6%. daily = -1600
    r = cb.record_trade_result(-500.0, 5000.0)  # cumulative 2.1%. daily = -2100 (> 2000 limit)
    assert r["halt"]
    assert "daily_loss_exceeded" in cb.state["halt_reason"]

def test_circuit_breaker_reset(tmp_path):
    p = tmp_path / "state3.json"
    cfg = CircuitConfig(nav_usd=100000.0)
    cb = CircuitBreaker(cfg=cfg, state_path=str(p))
    cb.force_halt("test_halt")
    assert cb.is_halted()
    cb.reset()
    assert not cb.is_halted()
