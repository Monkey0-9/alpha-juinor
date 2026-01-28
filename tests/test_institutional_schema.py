#!/usr/bin/env python
"""Test script for institutional architecture schema."""

import sys
sys.path.insert(0, '.')

from database.schema import (
    # Existing classes
    PriceRecord, CorporateAction, FeatureRecord, ModelOutput,
    DecisionRecord, OrderRecord, PositionRecord, AuditEntry, CycleMeta,
    get_schema_version,
    # New classes - Factor Attribution
    FactorReturn, SecurityFactorExposure, AgentPnl, PortfolioAttribution,
    # New classes - Agent Weight Learning
    AgentWeight, AgentPerformance, RegimeHistory,
    # New classes - RL Execution
    ExecutionModelVersion, ExecutionDecision, ExecutionOutcome,
    # New classes - Portfolio Optimization
    PortfolioTarget, PortfolioConstraint, OptimizationResult,
    # New classes - Slippage Feedback
    ExecutionFeedback, SlippageModelCoeff, SlippagePredictionError
)

print("=" * 60)
print("INSTITUTIONAL ARCHITECTURE SCHEMA TEST")
print("=" * 60)

# Test schema version
version = get_schema_version()
print(f"\n✓ Schema version: {version}")

# Test all dataclasses
classes_to_test = [
    # Existing
    ("PriceRecord", PriceRecord("AAPL", "2024-01-01", 100, 101, 99, 100.5, 100.5, 1000000, "yahoo")),
    ("CorporateAction", CorporateAction("AAPL", "2024-01-01", "split", {"ratio": 2})),
    ("CycleMeta", CycleMeta(cycle_id="test_123")),
    # Factor Attribution
    ("FactorReturn", FactorReturn("2024-01-01", "MOMENTUM", 0.05)),
    ("AgentPnl", AgentPnl("c1", "AAPL", "MomentumAgent", 0.02, 0.1, 0.8, 0.2, 0.03, 0.006)),
    ("PortfolioAttribution", PortfolioAttribution("2024-01-01")),
    # Agent Weight Learning
    ("AgentWeight", AgentWeight("2024-01-01", "c1", "MomentumAgent", 0.25)),
    ("RegimeHistory", RegimeHistory("2024-01-01", "BULL_QUIET")),
    # RL Execution
    ("ExecutionModelVersion", ExecutionModelVersion("v1", "SAC")),
    ("ExecutionDecision", ExecutionDecision(order_id="order_123", side="BUY", quantity=100, symbol="AAPL")),
    ("ExecutionOutcome", ExecutionOutcome("order_123")),
    # Portfolio Optimization
    ("PortfolioTarget", PortfolioTarget("c1", "AAPL", 0.05)),
    ("OptimizationResult", OptimizationResult("c1")),
    # Slippage Feedback
    ("ExecutionFeedback", ExecutionFeedback("order_123", 100.0, 100.1, 10.0)),
    ("SlippageModelCoeff", SlippageModelCoeff("AAPL", "2024-01-01")),
    ("SlippagePredictionError", SlippagePredictionError("2024-01-01", "AAPL", 0.01, 0.012)),
]

print("\n✓ All dataclasses instantiated successfully:")
for name, obj in classes_to_test:
    print(f"  - {name}")

# Test SQL creation
from database.schema import SCHEMA_SQL
sql_lines = len(SCHEMA_SQL.split('\n'))
print(f"\n✓ SQL schema: {sql_lines} lines")

# Count CREATE TABLE statements
create_count = SCHEMA_SQL.upper().count('CREATE TABLE')
print(f"✓ Tables defined: {create_count}")

# Count indexes
index_count = SCHEMA_SQL.upper().count('CREATE INDEX')
print(f"✓ Indexes defined: {index_count}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)

