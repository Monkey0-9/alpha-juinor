"""
Unit tests for institutional math formulas and components.
Tests: MAD, conviction z-score, fractional Kelly, CVaR, impact model.
"""

import pytest
import numpy as np
import pandas as pd
from meta_intelligence.pm_brain import PMBrain
from execution_ai.impact_model import ImpactModel
from risk.cvar import calculate_portfolio_cvar
from maths.financial import fractional_kelly


class TestMathFormulas:
    """Test core mathematical formulas"""

    def test_expected_return_aggregation(self):
        """Test mu_hat = sum(w_k * alpha_k)"""
        # Equal weight aggregation
        mus = [0.02, 0.03, 0.01]
        weights = [1/3, 1/3, 1/3]

        mu_hat = sum(w * mu for w, mu in zip(weights, mus))
        expected = 0.02  # (0.02 + 0.03 + 0.01) / 3

        assert abs(mu_hat - expected) < 1e-6

    def test_mad_calculation(self):
        """Test Median Absolute Deviation"""
        pm_brain = PMBrain()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mad = pm_brain.compute_mad(values)

        # Median = 3.0
        # Deviations = [2.0, 1.0, 0.0, 1.0, 2.0]
        # MAD = median([2.0, 1.0, 0.0, 1.0, 2.0]) = 1.0
        assert abs(mad - 1.0) < 1e-6

    def test_conviction_zscore(self):
        """Test z = (mu - median) / MAD"""
        pm_brain = PMBrain()

        all_mus = [0.01, 0.02, 0.03, 0.04, 0.05]
        mu_hat = 0.04

        z_score = pm_brain.compute_conviction_zscore(mu_hat, all_mus)

        # Median = 0.03
        # MAD = median([0.02, 0.01, 0.0, 0.01, 0.02]) = 0.01
        # z = (0.04 - 0.03) / 0.01 = 1.0
        assert abs(z_score - 1.0) < 1e-6

    def test_risk_adjusted_score(self):
        """Test S = mu / sigma"""
        mu = 0.10
        sigma = 0.15

        sharpe = mu / sigma
        expected = 0.6667

        assert abs(sharpe - expected) < 0.001

    def test_fractional_kelly(self):
        """Test f = gamma * mu / sigma^2"""
        mu = 0.10
        sigma = 0.20
        gamma = 0.15

        kelly = fractional_kelly(mu, sigma, gamma)
        expected = gamma * mu / (sigma ** 2)  # 0.15 * 0.10 / 0.04 = 0.375

        assert abs(kelly - expected) < 0.001

    def test_cvar_calculation(self):
        """Test CVaR (95%) calculation"""
        # Generate returns with known distribution
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

        cvar = calculate_portfolio_cvar(returns, confidence_level=0.95)

        # CVaR should be positive (representing loss)
        assert cvar > 0
        # Should be reasonable (< 10% for daily returns)
        assert cvar < 0.10

    def test_impact_model(self):
        """Test Almgren-Chriss impact estimation"""
        impact_model = ImpactModel()

        result = impact_model.estimate_impact(
            symbol="AAPL",
            order_size_usd=100000,
            adv_usd=10000000,  # $10M ADV
            volatility=0.02
        )

        # Check structure
        assert 'permanent_impact' in result
        assert 'temporary_impact' in result
        assert 'total_bps' in result
        assert 'participation_rate' in result

        # Participation rate should be 1%
        assert abs(result['participation_rate'] - 0.01) < 0.001

        # Total impact should be reasonable (< 100 bps for 1% participation)
        assert result['total_bps'] < 100
        assert result['total_bps'] > 0


class TestPMBrainIntegration:
    """Test PM Brain integration"""

    def test_pm_brain_aggregation(self):
        """Test full PM brain aggregation"""
        from contracts import AgentResult

        pm_brain = PMBrain()

        # Create mock agent results
        results = [
            AgentResult("AAPL", "momentum", mu=0.02, sigma=0.015, confidence=0.8, metadata={}),
            AgentResult("AAPL", "mean_reversion", mu=0.01, sigma=0.020, confidence=0.7, metadata={}),
            AgentResult("AAPL", "volatility", mu=0.03, sigma=0.010, confidence=0.9, metadata={})
        ]

        allocation_req, decision, reasons, metadata = pm_brain.aggregate(
            symbol="AAPL",
            results=results,
            cycle_id="test",
            regime="BULL_QUIET",
            liquidity_usd=1e6
        )

        # Should get some decision
        assert decision is not None
        assert len(reasons) > 0

        # Metadata should have key stats
        assert 'pm_score' in metadata
        assert 'conviction_zscore' in metadata
        assert 'avg_confidence' in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
