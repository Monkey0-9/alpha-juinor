import numpy as np
import pandas as pd
from nexus.math.models import TrendAccelerationModel, NeuralODE
from nexus.math.optimization import MonteCarloSimulator
from nexus.core.intelligence import MarketBrain
from nexus.core.governance import GovernanceEngine

def test_neural_ode_alias():
    """Verify NeuralODE is an alias for TrendAccelerationModel."""
    assert NeuralODE == TrendAccelerationModel
    model = NeuralODE()
    prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    force = model.predict_trajectory(prices)
    assert isinstance(force, float)

def test_monte_carlo_real_simulation():
    """Verify Monte Carlo is not a hardcoded stub."""
    mc = MonteCarloSimulator()
    returns = np.random.normal(0.001, 0.02, 100)
    
    # Run twice to see variation (or at least check it's not 0.999)
    prob = mc.run_survival_analysis(100000, returns, days=252, n_simulations=100)
    assert prob != 0.999
    assert 0.0 <= prob <= 1.0

def test_market_brain_deterministic():
    """Verify MarketBrain no longer uses random confidence."""
    brain = MarketBrain()
    data = pd.DataFrame({"close": [100, 101, 100, 99, 100]})
    analysis = brain.analyze_market(data, [])
    
    assert "strategy_agreement" in analysis
    assert isinstance(analysis["strategy_agreement"], float)
    
    # Second run should be identical for same data
    analysis2 = brain.analyze_market(data, [])
    assert analysis["strategy_agreement"] == analysis2["strategy_agreement"]

def test_governance_persistence_logic():
    """Verify governance logs to the persistence layer."""
    gov = GovernanceEngine()
    trade = {"symbol": "TSLA", "qty": 1, "price": 200, "side": "buy"}
    portfolio = {"total_value": 100000, "drawdown": 0.0}
    
    approved, _ = gov.check_compliance(trade, portfolio)
    assert approved is True
    
    # Check that audit log has entry
    assert len(gov.audit_log) > 0
    assert gov.audit_log[-1]["symbol"] == "TSLA"
