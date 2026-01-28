#!/usr/bin/env python3
"""
tests/agents/test_institutional_decision_agent.py

Unit tests for the Institutional Trading Decision Agent.

Tests cover:
1. Input validation
2. Ensemble scoring
3. Position sizing
4. Risk gates
5. Entry/exit logic
6. Determinism
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.institutional_decision_agent import (
    InstitutionalDecisionAgent,
    create_sample_input,
    validate_input,
    compute_mean_reversion_component,
    compute_momentum_component,
    compute_liquidity_component,
    compute_volatility_component,
    compute_ensemble_score,
    compute_kelly_fraction,
    compute_volatility_target_size,
    compute_stop_loss,
    compute_entry_zone,
    check_regime_gate,
    check_cvar_gate,
    check_data_confidence_gate,
    check_execution_gate,
    compute_signature,
    Decision
)


class TestInputValidation:
    """Test input validation."""

    def test_valid_input(self):
        """Test that valid input passes validation."""
        inp = create_sample_input()
        is_valid, errors = validate_input(inp)
        assert is_valid
        assert len(errors) == 0

    def test_missing_required_field(self):
        """Test that missing required field fails validation."""
        inp = create_sample_input()
        del inp["price"]
        is_valid, errors = validate_input(inp)
        assert not is_valid
        assert "Missing required field: price" in errors

    def test_missing_feature(self):
        """Test that missing feature fails validation."""
        inp = create_sample_input()
        del inp["features"]["rsi_3"]
        is_valid, errors = validate_input(inp)
        assert not is_valid
        assert "Missing required feature: rsi_3" in errors

    def test_invalid_price(self):
        """Test that invalid price fails validation."""
        inp = create_sample_input()
        inp["price"] = -100
        is_valid, errors = validate_input(inp)
        assert not is_valid
        assert "Invalid price: -100" in errors


class TestEnsembleScoring:
    """Test ensemble scoring components."""

    def test_mean_reversion_oversold(self):
        """Test mean reversion with oversold conditions."""
        # boll_z=-2, rsi_3=20 -> should be positive (buy signal)
        score = compute_mean_reversion_component(boll_z=-2.0, rsi_3=20.0)
        assert score > 0
        assert score <= 1.0

    def test_mean_reversion_overbought(self):
        """Test mean reversion with overbought conditions."""
        # boll_z=2, rsi_3=80 -> should be negative (sell signal)
        score = compute_mean_reversion_component(boll_z=2.0, rsi_3=80.0)
        assert score < 0
        assert score >= -1.0

    def test_momentum_uptrend(self):
        """Test momentum with uptrend."""
        score = compute_momentum_component(ema_9=105, ema_21=100, macd_hist=0.5)
        assert score > 0

    def test_momentum_downtrend(self):
        """Test momentum with downtrend."""
        score = compute_momentum_component(ema_9=95, ema_21=100, macd_hist=-0.5)
        assert score < 0

    def test_liquidity_high(self):
        """Test liquidity with high ADV and volume."""
        score = compute_liquidity_component(adv_usd=10_000_000, volume_z=2.0)
        assert score == 0.3

    def test_liquidity_low(self):
        """Test liquidity with low ADV."""
        score = compute_liquidity_component(adv_usd=1_000_000, volume_z=0.5)
        assert score == -0.2

    def test_volatility_high_risk(self):
        """Test volatility with high ATR."""
        score = compute_volatility_component(atr_pct=0.05, threshold=0.03)
        assert score == -0.5

    def test_volatility_normal(self):
        """Test volatility with normal ATR."""
        score = compute_volatility_component(atr_pct=0.02, threshold=0.03)
        assert score == 0.1

    def test_ensemble_range(self):
        """Test ensemble score is in valid range."""
        score = compute_ensemble_score(0.8, 0.6, 0.3, 0.1)
        assert -1.0 <= score <= 1.0


class TestPositionSizing:
    """Test position sizing logic."""

    def test_kelly_positive_edge(self):
        """Test Kelly with positive edge."""
        kelly = compute_kelly_fraction(
            expected_edge_bps=50,
            estimated_variance=0.04  # 20% annual vol
        )
        assert kelly > 0
        assert kelly <= 0.25  # Max Kelly cap

    def test_kelly_zero_edge(self):
        """Test Kelly with zero edge."""
        kelly = compute_kelly_fraction(
            expected_edge_bps=0,
            estimated_variance=0.04
        )
        assert kelly == 0

    def test_kelly_negative_edge(self):
        """Test Kelly with negative edge."""
        kelly = compute_kelly_fraction(
            expected_edge_bps=-20,
            estimated_variance=0.04
        )
        assert kelly <= 0

    def test_volatility_target_sizing(self):
        """Test volatility target sizing."""
        size = compute_volatility_target_size(
            nav_usd=1_000_000,
            realized_vol_annual=0.30,
            strategy_target_vol=0.15
        )
        assert 0 <= size <= 1.0
        assert size == 0.5  # 0.15 / 0.30


class TestRiskGates:
    """Test risk gate implementations."""

    def test_regime_gate_risk_on_buy(self):
        """Test regime gate allows BUY in RISK_ON."""
        ok, reason = check_regime_gate("RISK_ON", Decision.BUY)
        assert ok
        assert reason == ""

    def test_regime_gate_risk_off_buy(self):
        """Test regime gate blocks BUY in RISK_OFF."""
        ok, reason = check_regime_gate("RISK_OFF", Decision.BUY)
        assert not ok
        assert "regime_block" in reason

    def test_regime_gate_risk_off_hold(self):
        """Test regime gate allows HOLD in RISK_OFF."""
        ok, reason = check_regime_gate("RISK_OFF", Decision.HOLD)
        assert ok

    def test_cvar_gate_within_limit(self):
        """Test CVaR gate with acceptable risk."""
        ok, reason = check_cvar_gate(
            portfolio_cvar=0.03,
            symbol_cvar_contribution=0.01,
            position_size=0.02,
            cvar_limit=0.05
        )
        assert ok

    def test_cvar_gate_exceeds_limit(self):
        """Test CVaR gate with excessive risk."""
        ok, reason = check_cvar_gate(
            portfolio_cvar=0.04,
            symbol_cvar_contribution=0.20,  # Higher contribution
            position_size=0.10,
            cvar_limit=0.05
        )
        # 0.04 + 0.20 * 0.10 = 0.06 > 0.05
        assert not ok
        assert "cvar_breach" in reason

    def test_data_confidence_gate_pass(self):
        """Test data confidence gate with good data."""
        ok, reason = check_data_confidence_gate(0.8, threshold=0.6)
        assert ok

    def test_data_confidence_gate_fail(self):
        """Test data confidence gate with poor data."""
        ok, reason = check_data_confidence_gate(0.4, threshold=0.6)
        assert not ok
        assert "low_data_confidence" in reason

    def test_execution_gate_pass(self):
        """Test execution gate with sufficient edge."""
        ok, reason = check_execution_gate(
            slippage_bps=5,
            spread_bps=3,
            expected_edge_bps=20,
            buffer_bps=5
        )
        assert ok  # 20 - 5 - 3 - 5 = 7 > 0

    def test_execution_gate_fail(self):
        """Test execution gate with insufficient edge."""
        ok, reason = check_execution_gate(
            slippage_bps=10,
            spread_bps=5,
            expected_edge_bps=15,
            buffer_bps=5
        )
        assert not ok  # 15 - 10 - 5 - 5 = -5 < 0


class TestEntryExitLogic:
    """Test entry/exit logic."""

    def test_entry_zone_calculation(self):
        """Test entry zone calculation."""
        zone = compute_entry_zone(price=100, atr_pct=0.02, entry_type="LIMIT")
        assert zone["type"] == "LIMIT"
        assert zone["low"] < 100
        assert zone["high"] > 100
        assert zone["low"] == 99.0  # 100 * (1 - 0.02 * 0.5)
        assert zone["high"] == 100.6  # 100 * (1 + 0.02 * 0.3)

    def test_stop_loss_calculation(self):
        """Test stop loss calculation."""
        stop_price, stop_pct = compute_stop_loss(price=100, atr_pct=0.02, n_atr=1.5)
        assert stop_price < 100
        assert stop_pct == 0.03  # 1.5 * 0.02
        assert stop_price == 97.0  # 100 * (1 - 0.03)

    def test_stop_loss_minimum(self):
        """Test stop loss minimum enforcement."""
        stop_price, stop_pct = compute_stop_loss(price=100, atr_pct=0.001, n_atr=1.5)
        assert stop_pct >= 0.005  # Minimum 0.5%


class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_input_same_output(self):
        """Test that same input produces same output."""
        agent = InstitutionalDecisionAgent()
        inp = create_sample_input()

        proposal1 = agent.compute_proposal(inp)
        proposal2 = agent.compute_proposal(inp)

        assert proposal1.decision == proposal2.decision
        assert proposal1.ensemble_score == proposal2.ensemble_score
        assert proposal1.signature == proposal2.signature

    def test_signature_deterministic(self):
        """Test signature is deterministic."""
        sig1 = compute_signature("run1", 123, "BUY", 0.75)
        sig2 = compute_signature("run1", 123, "BUY", 0.75)
        assert sig1 == sig2

    def test_signature_changes_with_input(self):
        """Test signature changes with different input."""
        sig1 = compute_signature("run1", 123, "BUY", 0.75)
        sig2 = compute_signature("run1", 123, "SELL", 0.75)
        assert sig1 != sig2


class TestAgentDecisions:
    """Test full agent decision flow."""

    def test_hold_decision(self):
        """Test HOLD decision with neutral signals."""
        agent = InstitutionalDecisionAgent()
        inp = create_sample_input()
        inp["features"]["boll_z"] = 0
        inp["features"]["rsi_3"] = 50

        proposal = agent.compute_proposal(inp)
        assert proposal.decision in ["HOLD", "REJECT"]

    def test_buy_decision(self):
        """Test BUY decision with strong oversold signals."""
        agent = InstitutionalDecisionAgent(buy_threshold=0.5)
        inp = create_sample_input()
        inp["features"]["boll_z"] = -3.0
        inp["features"]["rsi_3"] = 10.0
        inp["features"]["ema_9"] = 190.0
        inp["features"]["ema_21"] = 180.0
        inp["features"]["macd_hist"] = 0.5
        inp["features"]["volume_z"] = 2.0
        inp["models"]["expected_edge_bps"] = 50

        proposal = agent.compute_proposal(inp)
        assert proposal.decision == "BUY"
        assert proposal.suggested_qty > 0
        assert proposal.suggested_notional_pct > 0

    def test_reject_on_risk_off(self):
        """Test REJECT in RISK_OFF regime when BUY signal is strong."""
        agent = InstitutionalDecisionAgent(buy_threshold=0.5)
        inp = create_sample_input()
        inp["market"]["regime"] = "RISK_OFF"
        # Create strong BUY signal to trigger regime gate
        inp["features"]["boll_z"] = -3.5
        inp["features"]["rsi_3"] = 5.0
        inp["features"]["ema_9"] = 195.0
        inp["features"]["ema_21"] = 180.0
        inp["features"]["macd_hist"] = 1.0
        inp["features"]["volume_z"] = 3.0
        inp["models"]["expected_edge_bps"] = 80

        proposal = agent.compute_proposal(inp)
        # Either REJECT or regime_ok is False
        assert proposal.decision == "REJECT" or not proposal.risk_checks.get("regime_ok", True)

    def test_reject_on_low_data_confidence(self):
        """Test REJECT with low data confidence."""
        agent = InstitutionalDecisionAgent()
        inp = create_sample_input()
        inp["features"]["data_confidence"] = 0.3

        proposal = agent.compute_proposal(inp)
        assert proposal.decision == "REJECT"
        assert not proposal.risk_checks["data_confidence_ok"]

    def test_output_schema(self):
        """Test output contains all required fields."""
        agent = InstitutionalDecisionAgent()
        inp = create_sample_input()
        proposal = agent.compute_proposal(inp)

        # Check required fields
        assert hasattr(proposal, "run_id")
        assert hasattr(proposal, "seed")
        assert hasattr(proposal, "decision")
        assert hasattr(proposal, "confidence")
        assert hasattr(proposal, "ensemble_score")
        assert hasattr(proposal, "suggested_notional_pct")
        assert hasattr(proposal, "entry_zone")
        assert hasattr(proposal, "exit_logic")
        assert hasattr(proposal, "risk_checks")
        assert hasattr(proposal, "explain")
        assert hasattr(proposal, "signature")

    def test_json_serialization(self):
        """Test proposal serializes to valid JSON."""
        agent = InstitutionalDecisionAgent()
        inp = create_sample_input()
        proposal = agent.compute_proposal(inp)

        json_str = proposal.to_json()
        parsed = json.loads(json_str)

        assert parsed["decision"] == proposal.decision
        assert parsed["run_id"] == proposal.run_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
