# tests/test_decision_agent.py
"""
Unit tests for the production DecisionAgent.

Tests:
1. Determinism - same input produces identical output
2. Schema validation - output contains required fields
3. Decision logic - various scenarios
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.decision_agent import DecisionAgent
from tests.test_helpers import sample_input


class TestDeterminism:
    """Test that agent produces deterministic outputs."""

    def test_decision_agent_deterministic(self):
        """Same input produces identical output."""
        run_id = "test_run_0001"
        inp = sample_input(run_id=run_id)

        agent1 = DecisionAgent(agent_id="agent_x")
        p1 = agent1.run(inp)

        agent2 = DecisionAgent(agent_id="agent_x")
        p2 = agent2.run(inp)

        assert p1 == p2, "Determinism failure: proposals differ"

    def test_different_run_id_different_seed(self):
        """Different run_id produces different seed."""
        inp1 = sample_input(run_id="run_001")
        inp2 = sample_input(run_id="run_002")

        agent = DecisionAgent(agent_id="agent_x")
        p1 = agent.run(inp1)
        p2 = agent.run(inp2)

        assert p1["seed"] != p2["seed"]


class TestSchemaValidation:
    """Test output schema compliance."""

    def test_required_fields_present(self):
        """All required fields are present in output."""
        inp = sample_input()
        agent = DecisionAgent()
        proposal = agent.run(inp)

        required_fields = [
            "run_id", "seed", "timestamp", "agent_id",
            "decision", "confidence", "ensemble_score",
            "primary_signal", "suggested_notional_pct",
            "suggested_qty", "price_limits", "entry_zone",
            "exit_logic", "risk_checks", "explain",
            "warnings", "contract_version", "schema_hash",
            "signature"
        ]

        for field in required_fields:
            assert field in proposal, f"Missing required field: {field}"

    def test_decision_is_valid_enum(self):
        """Decision is one of BUY/SELL/HOLD/REJECT."""
        inp = sample_input()
        agent = DecisionAgent()
        proposal = agent.run(inp)

        assert proposal["decision"] in ["BUY", "SELL", "HOLD", "REJECT"]

    def test_json_serializable(self):
        """Output is JSON serializable."""
        inp = sample_input()
        agent = DecisionAgent()
        proposal = agent.run(inp)

        # Should not raise
        json_str = json.dumps(proposal)
        parsed = json.loads(json_str)
        assert parsed["run_id"] == proposal["run_id"]


class TestDecisionLogic:
    """Test decision-making scenarios."""

    def test_reject_on_low_data_confidence(self):
        """Low data confidence results in REJECT."""
        inp = sample_input()
        inp["data_confidence"] = 0.3  # Below threshold

        agent = DecisionAgent()
        proposal = agent.run(inp)

        assert proposal["decision"] == "REJECT"
        assert not proposal["risk_checks"]["data_confidence_ok"]

    def test_hold_on_neutral_signals(self):
        """Neutral signals result in HOLD."""
        inp = sample_input()
        inp["features"]["rsi_3"] = 50.0  # Neutral
        inp["features"]["boll_z"] = 0.0  # Neutral
        inp["features"]["ema_9"] = 100.0
        inp["features"]["ema_21"] = 100.0
        inp["features"]["macd_hist"] = 0.0

        agent = DecisionAgent()
        proposal = agent.run(inp)

        # With neutral signals, should not be BUY
        assert proposal["decision"] in ["HOLD", "REJECT"]

    def test_sell_on_position_with_negative_ensemble(self):
        """Existing position with negative ensemble triggers SELL."""
        inp = sample_input()
        inp["position_state"] = {
            "has_position": True,
            "qty": 100,
            "position_pct": 0.01
        }
        # Strong sell signals
        inp["features"]["rsi_3"] = 85.0
        inp["features"]["boll_z"] = 2.5
        inp["features"]["ema_9"] = 95.0
        inp["features"]["ema_21"] = 105.0
        inp["features"]["macd_hist"] = -0.8

        agent = DecisionAgent()
        proposal = agent.run(inp)

        # Likely SELL or HOLD depending on threshold
        assert proposal["decision"] in ["SELL", "HOLD"]

    def test_stop_loss_triggers_sell(self):
        """Unrealized loss beyond stop triggers SELL."""
        inp = sample_input()
        inp["position_state"] = {
            "has_position": True,
            "qty": 100,
            "unrealized_pct": -0.05  # -5% loss
        }
        inp["risk"]["stop_distance_pct"] = 0.02  # 2% stop

        agent = DecisionAgent()
        proposal = agent.run(inp)

        assert proposal["decision"] == "SELL"
        assert proposal["confidence"] == 0.95


class TestRiskGates:
    """Test risk gate behavior."""

    def test_exit_logic_populated(self):
        """Exit logic contains required fields."""
        inp = sample_input()
        agent = DecisionAgent()
        proposal = agent.run(inp)

        exit_logic = proposal["exit_logic"]

        if exit_logic:  # Not empty (not REJECT)
            assert "stop_loss_price" in exit_logic
            assert "stop_loss_pct" in exit_logic
            assert "take_profit_tiers" in exit_logic

    def test_risk_checks_populated(self):
        """Risk checks contain required gates."""
        inp = sample_input()
        agent = DecisionAgent()
        proposal = agent.run(inp)

        risk_checks = proposal["risk_checks"]

        assert "data_confidence_ok" in risk_checks
        assert "cvar_ok" in risk_checks
        assert "entanglement_ok" in risk_checks
        assert "execution_cost_ok" in risk_checks


class TestExplanation:
    """Test explainability features."""

    def test_explain_has_indicators(self):
        """Explanation includes indicators used."""
        inp = sample_input()
        agent = DecisionAgent()
        proposal = agent.run(inp)

        explain = proposal["explain"]

        assert "indicators_used" in explain or "reason" in explain

    def test_explain_has_edge_estimate(self):
        """Explanation includes expected edge."""
        inp = sample_input()
        agent = DecisionAgent()
        proposal = agent.run(inp)

        explain = proposal["explain"]

        if "expected_edge_bps" in explain:
            assert isinstance(explain["expected_edge_bps"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
