
import pytest
import pandas as pd
import numpy as np
from contracts import AllocationRequest
from portfolio.capital_auction import CapitalAuctionEngine
from execution.gates import ExecutionGatekeeper
from portfolio.allocator import InstitutionalAllocator

def test_capital_auction_competition():
    """Verify that higher confidence/mu wins in the auction."""
    engine = CapitalAuctionEngine(hurdle_rate=0.01, total_cap_limit=1.0)

    req1 = AllocationRequest(
        symbol="AAPL", mu=0.05, sigma=0.02, confidence=0.8,
        liquidity=1e6, regime="NORMAL", timestamp="now"
    )
    req2 = AllocationRequest(
        symbol="TSLA", mu=0.02, sigma=0.02, confidence=0.4,
        liquidity=1e6, regime="NORMAL", timestamp="now"
    )

    weights = engine.auction_capital([req1, req2])

    assert "AAPL" in weights
    assert "TSLA" in weights
    assert weights["AAPL"] > weights["TSLA"]
    assert sum(weights.values()) <= 1.05 # Allow small float margin

def test_capital_auction_hurdle():
    """Verify that signals below the hurdle rate are rejected."""
    engine = CapitalAuctionEngine(hurdle_rate=0.10) # Very high hurdle
    req = AllocationRequest(
        symbol="AAPL", mu=0.0001, sigma=0.02, confidence=0.9,
        liquidity=1e6, regime="NORMAL", timestamp="now"
    )
    weights = engine.auction_capital([req])
    assert "AAPL" not in weights

def test_execution_gate_adv():
    """Verify ADV limit scaling."""
    gate = ExecutionGatekeeper(adv_limit_pct=0.10)
    # Order 100k, ADV 500k -> 20% of ADV -> should be scaled to 50k (10%)
    is_ok, reason, scaled_qty = gate.validate_execution(
        symbol="AAPL", qty=100000, side="BUY", price=100, adv_30d=500000, volatility=0.02
    )
    assert is_ok is False
    assert reason == "ADV_LIMIT_EXCEEDED"
    assert scaled_qty == 50000

def test_execution_gate_impact():
    """Verify high impact rejection/scaling."""
    # Set high ADV limit to ensure we hit the impact gate
    gate = ExecutionGatekeeper(adv_limit_pct=1.0, max_impact_bps=5.0)
    # Large order relative to ADV usually causes high impact in these models
    is_ok, reason, scaled_qty = gate.validate_execution(
        symbol="ILLIQ", qty=50000, side="BUY", price=100, adv_30d=100000, volatility=0.10
    )
    assert is_ok is False
    assert reason == "HIGH_IMPACT_ESTIMATE"
    assert scaled_qty < 50000

def test_allocator_batch_interface():
    """Verify that InstitutionalAllocator handles batch requests via auction engine."""
    allocator = InstitutionalAllocator(max_leverage=1.0)

    # Mocking signals DataFrame
    signals = pd.DataFrame({"AAPL": [0.9], "MSFT": [0.6]}, index=[pd.Timestamp.utcnow()])

    weights = allocator.allocate(signals)
    assert isinstance(weights, dict)
    assert "AAPL" in weights
    assert "MSFT" in weights
    assert weights["AAPL"] > weights["MSFT"]
