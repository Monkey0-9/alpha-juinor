import pytest
from mini_quant_fund.portfolio.allocator import InstitutionalAllocator, SymbolDecision

def test_allocator_accepts_metadata():
    allocator = InstitutionalAllocator()
    decisions = [SymbolDecision(symbol="AAPL", mu=0.01, sigma=0.02, confidence=0.8)]

    # This should not raise TypeError
    orders = allocator.allocate(decisions, nav=1000000, metadata={"test": True})

    assert len(orders) > 0
    assert orders[0].symbol == "AAPL"

def test_allocator_no_metadata():
    allocator = InstitutionalAllocator()
    decisions = [SymbolDecision(symbol="AAPL", mu=0.01, sigma=0.02, confidence=0.8)]

    # This should also work fine
    orders = allocator.allocate(decisions, nav=1000000)
    assert len(orders) > 0
