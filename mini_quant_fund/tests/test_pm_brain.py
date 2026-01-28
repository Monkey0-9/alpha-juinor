import pytest
from mini_quant_fund.meta_intelligence.pm_brain import PMBrain, AgentOutput
from mini_quant_fund.data_intelligence.ingestion import DataQualityResult

def test_pm_brain_aggregation():
    brain = PMBrain(gamma=0.5)
    dq = DataQualityResult(symbol="AAPL", score=1.0, is_valid=True, errors=[])
    signals = [
        AgentOutput(symbol="AAPL", mu=0.02, sigma=0.04, confidence=0.8, debug={}),
        AgentOutput(symbol="AAPL", mu=0.02, sigma=0.04, confidence=0.8, debug={})
    ]

    res = brain.aggregate_signals("AAPL", signals, dq)
    assert res["symbol"] == "AAPL"
    assert res["mu_hat"] == 0.02
    assert res["f"] > 0 # Kelly size should be positive for positive mu

def test_pm_brain_disagreement_penalty():
    brain = PMBrain(gamma=0.5)
    dq = DataQualityResult(symbol="AAPL", score=1.0, is_valid=True, errors=[])

    # High disagreement signals
    signals = [
        AgentOutput(symbol="AAPL", mu=0.05, sigma=0.02, confidence=0.5, debug={}),
        AgentOutput(symbol="AAPL", mu=-0.05, sigma=0.02, confidence=0.5, debug={})
    ]

    res = brain.aggregate_signals("AAPL", signals, dq)
    assert res["mu_hat"] == 0.0
    assert res["sigma_agg"] > 0.02 # Should be larger due to MAD penalty

def test_pm_brain_data_quality_reduction():
    brain = PMBrain(gamma=0.5)
    dq = DataQualityResult(symbol="AAPL", score=0.5, is_valid=True, errors=["Minor gap"])
    signals = [AgentOutput(symbol="AAPL", mu=0.02, sigma=0.04, confidence=1.0, debug={})]

    res_good = brain.aggregate_signals("AAPL", signals, DataQualityResult(symbol="AAPL", score=1.0, is_valid=True, errors=[]))
    res_bad = brain.aggregate_signals("AAPL", signals, dq)

    assert res_bad["f"] < res_good["f"]
    assert "DQ_REDUCTION" in res_bad["reason_codes"]
