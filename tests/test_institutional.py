import pytest
import numpy as np
import pandas as pd
from nexus.core.governance import GovernanceEngine
from nexus.core.execution_ai import ExecutionAgent
from nexus.core.alpha import AlphaEngine
from nexus.core.engine import NexusEngine
from nexus.math.optimization import PortfolioOptimizer, MultiFactorEngine

def test_governance_concentration():
    engine = GovernanceEngine(single_position_limit=0.1)
    portfolio = {"total_value": 100000, "drawdown": 0.02}
    
    # Within limit
    trade = {"symbol": "AAPL", "qty": 10, "price": 150, "side": "buy"} # $1500 = 1.5%
    approved, violations = engine.check_compliance(trade, portfolio)
    assert approved is True
    
    # Exceeds limit
    large_trade = {"symbol": "TSLA", "qty": 100, "price": 200, "side": "buy"} # $20000 = 20%
    approved, violations = engine.check_compliance(large_trade, portfolio)
    assert approved is False
    assert any("POSITION_CONCENTRATION" in v for v in violations)

def test_governance_drawdown():
    engine = GovernanceEngine(max_drawdown_limit=0.1)
    trade = {"symbol": "AAPL", "qty": 1, "price": 150, "side": "buy"}
    
    # Healthy
    portfolio_healthy = {"total_value": 100000, "drawdown": 0.05}
    approved, _ = engine.check_compliance(trade, portfolio_healthy)
    assert approved is True
    
    # Breach
    portfolio_stressed = {"total_value": 100000, "drawdown": 0.12}
    approved, violations = engine.check_compliance(trade, portfolio_stressed)
    assert approved is False
    assert any("DRAWDOWN_BREACH" in v for v in violations)

def test_portfolio_optimizer():
    optimizer = PortfolioOptimizer()
    symbols = ["AAPL", "MSFT", "GOOGL"]
    signals = [0.8, 0.4, 0.9]
    
    weights = optimizer.optimize_weights(symbols, signals)
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    # Strongest signal (GOOGL) should have highest weight
    assert weights["GOOGL"] > weights["AAPL"] > weights["MSFT"]


def test_portfolio_optimizer_skips_negative_signals():
    optimizer = PortfolioOptimizer()
    symbols = ["AAPL", "MSFT", "GOOGL"]
    signals = [0.5, -0.3, 0.1]

    weights = optimizer.optimize_weights(symbols, signals)
    assert weights["MSFT"] == 0.0
    assert weights["AAPL"] > 0.0
    assert weights["GOOGL"] > 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_rank_assets_penalizes_volatility():
    engine = MultiFactorEngine()
    signals = {"AAPL": 0.5, "MSFT": 0.5}
    hist_data = {
        "AAPL": pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]}),
        "MSFT": pd.DataFrame({"close": [100.0, 120.0, 80.0, 130.0, 70.0]}),
    }

    rankings = engine.rank_assets(signals, hist_data)
    assert list(rankings.keys())[0] == "AAPL"
    assert list(rankings.keys())[-1] == "MSFT"


def test_execution_agent_deterministic_routing():
    agent = ExecutionAgent()
    assert agent.get_action(np.array([0.0003, 0.8, 0.04, 0.0])) == 0
    assert agent.get_action(np.array([0.0002, 1.0, 0.015, 0.03])) == 2
    assert agent.get_action(np.array([0.0001, 1.2, 0.01, 0.005])) == 1


def test_alpha_engine_probability_is_deterministic():
    alpha_engine = AlphaEngine()
    bullish = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    bearish = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
    assert alpha_engine.monte_carlo_simulation(bullish) == 1.0
    assert alpha_engine.monte_carlo_simulation(bearish) == 0.0
    assert alpha_engine.monte_carlo_simulation(np.array([100.0, 100.0, 100.0, 100.0, 100.0])) == 0.0


def test_engine_risk_scale_uses_strategy_agreement():
    engine = NexusEngine(backend_url="http://127.0.0.1:8000")
    low_scale = engine.determine_risk_scale(
        {"regime": "SIDEWAYS", "strategy_agreement": 0.2},
        {"var": -0.01, "volatility": 0.015},
    )
    high_scale = engine.determine_risk_scale(
        {"regime": "SIDEWAYS", "strategy_agreement": 0.8},
        {"var": -0.01, "volatility": 0.015},
    )
    assert low_scale < high_scale


def test_factor_engine_ranking():
    engine = MultiFactorEngine()
    signals = {"AAPL": 0.5, "MSFT": -0.2, "TSLA": 0.8}
    # Constant data to focus on alpha signal dominance in this test
    hist_data = {
        "AAPL": pd.DataFrame({"close": [100.0]*10}),
        "MSFT": pd.DataFrame({"close": [100.0]*10}),
        "TSLA": pd.DataFrame({"close": [100.0]*10})
    }
    
    rankings = engine.rank_assets(signals, hist_data)
    assert list(rankings.keys())[0] == "TSLA" # Highest signal should be first
    assert list(rankings.keys())[-1] == "MSFT" # Lowest signal should be last
