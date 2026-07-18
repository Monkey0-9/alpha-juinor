import pandas as pd
from nexus.core.governance import GovernanceEngine
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
    # Kelly optimizer may drop low-conviction symbols below MIN_POSITION floor — expected
    assert len(weights) >= 1
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    # Strongest signal (GOOGL) should have highest weight among included symbols
    assert "GOOGL" in weights
    if "AAPL" in weights:
        assert weights["GOOGL"] >= weights["AAPL"]
    # Verify MSFT (weakest signal) doesn't dominate
    if "MSFT" in weights and "GOOGL" in weights:
        assert weights["GOOGL"] >= weights["MSFT"]

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
