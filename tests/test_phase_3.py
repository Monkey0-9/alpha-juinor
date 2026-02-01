
import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Add project root
sys.path.insert(0, os.getcwd())

from contracts.alpha_model import AlphaOutput, AlphaDecision
from portfolio.pm_brain import PMBrain
from contracts.allocation import RejectedAsset

class TestPhase3(unittest.TestCase):

    def setUp(self):
        self.brain = PMBrain()
        # Mock Quantum modules to avoid external dependencies
        self.brain.regime_space.get_compatibility = MagicMock(return_value=1.0)

    def test_allocation_logic_simple(self):
        """Test simple allocation."""
        alphas = {
            "AAPL": AlphaOutput(
                mu=0.01, sigma=0.02, cvar_95=-0.03, confidence=0.8,
                provider="test", model_version="v1", input_schema_hash="abc", decision=AlphaDecision.BUY
            ),
            "SPY": AlphaOutput(
                mu=0.005, sigma=0.015, cvar_95=-0.03, confidence=0.9,
                provider="test", model_version="v1", input_schema_hash="abc", decision=AlphaDecision.BUY
            )
        }

        current_positions = {"AAPL": 0.0, "SPY": 0.0}

        weights, rejected = self.brain.allocate(
            alphas=alphas,
            current_positions=current_positions,
            cov=None,
            liquidity={"AAPL": 0.01, "SPY": 0.01}
        )

        self.assertGreater(weights.get("AAPL", 0), 0)
        self.assertGreater(weights.get("SPY", 0), 0)
        self.assertAlmostEqual(sum(weights.values()), 1.0, delta=0.01)

    def test_rejection_logic(self):
        """Test rejection of bad alphas."""
        alphas = {
            "BAD_ASSET": AlphaOutput(
                mu=-0.01, sigma=0.02, cvar_95=-0.03, confidence=0.1, # Low confidence
                provider="test", model_version="v1", input_schema_hash="abc", decision=AlphaDecision.BUY
            )
        }

        current_positions = {}

        weights, rejected = self.brain.allocate(
            alphas=alphas,
            current_positions=current_positions
        )

        self.assertEqual(len(weights), 0)
        # Note: Depending on enforcement, it might be filtered out before optimization
        # or filtered out by optimization.
        # Our enforcement logic rejects confidence < 0.2
        self.assertEqual(len(weights), 0)

if __name__ == "__main__":
    unittest.main()
