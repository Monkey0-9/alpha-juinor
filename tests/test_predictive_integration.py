"""
Test Predictive Integration
Verifies that the Autonomous Brain correctly leverages the Machine Learning model.
"""
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd

from intelligence.autonomous_brain import AutonomousTradingBrain


class TestPredictiveIntegration(unittest.TestCase):
    def setUp(self):
        self.brain = AutonomousTradingBrain()
        # Mocking modules to isolate Brain logic
        self.brain._regime = MagicMock()
        self.brain._regime.analyze.return_value = MagicMock(
            primary_regime=MagicMock(value="BULLISH"),
            secondary_regime=MagicMock(value="VOLATILE"),
            avoid_strategies=[],
            recommended_strategies=[],
            risk_multiplier=1.0
        )
        self.brain._smart_money = MagicMock()
        self.brain._smart_money.scan_market.return_value = []

    def test_model_loading(self):
        """Verify the brain attempts to load the predictive model."""
        # It should have initialized _predictive_model (even if None)
        self.assertTrue(hasattr(self.brain, '_predictive_model'))

    @patch('intelligence.autonomous_brain.PredictiveModel')
    def test_high_confidence_boost(self, MockModel):
        """Test that high ML probability boosts the score."""
        # Setup Brain with Mocked Model
        brain = AutonomousTradingBrain()
        brain._predictive_model = MockModel.return_value
        brain._predictive_model.get_forecast.return_value = {
            'probability': 0.85,
            'confidence': 'high_bullish'
        }

        # Create a dummy signal
        signal = {
            'symbol': 'TEST',
            'confidence': 0.6,
            'action': 'BUY',
            'strategy': 'TestStrategy',
            'reasoning': 'Base Reason'
        }

        # Mock Market Data
        market_data = pd.DataFrame({'TEST': {'Close': 100}})
        # Note: Brain expects specific structure, let's use a standard Dict or DF
        # rank_opportunities expects market_data index/columns to match
        # Let's mock the market_data access inside _rank_opportunities
        # Using a dictionary for market_data mimics how it's often used if not MultiIndex
        market_data = {'TEST': pd.DataFrame({'close': [100]})}

        smart_money = {}
        regime = {'avoid': [], 'recommended': []}

        ranked = brain._rank_opportunities([signal], smart_money, regime, market_data)

        # 0.6 * 1.25 = 0.75
        self.assertGreater(ranked[0]['score'], 0.6)
        self.assertIn("ML BOOST", ranked[0]['reasoning'])

    @patch('intelligence.autonomous_brain.PredictiveModel')
    def test_low_confidence_penalty(self, MockModel):
        """Test that low ML probability penalizes the score."""
        brain = AutonomousTradingBrain()
        brain._predictive_model = MockModel.return_value
        brain._predictive_model.get_forecast.return_value = {
            'probability': 0.20,
            'confidence': 'high_bearish'
        }

        signal = {
            'symbol': 'TEST',
            'confidence': 0.6,
            'action': 'BUY',
            'strategy': 'TestStrategy'
        }
        market_data = {'TEST': pd.DataFrame({'close': [100]})}

        ranked = brain._rank_opportunities([signal], {}, {'avoid':[]}, market_data)

        # 0.6 * 0.7 = 0.42
        self.assertLess(ranked[0]['score'], 0.6)
        self.assertIn("ML PENALTY", ranked[0]['reasoning'])

if __name__ == '__main__':
    unittest.main()
