"""
Integration tests for CompositeAlphaStrategy.
"""

import pytest
import pandas as pd
import numpy as np
from strategies.composite_alpha import CompositeAlphaStrategy
from risk.engine import RiskManager
from alpha_families.fundamental_alpha import FundamentalAlpha
from alpha_families.statistical_alpha import StatisticalAlpha


class TestCompositeAlphaStrategy:
    """Test CompositeAlphaStrategy functionality."""

    @pytest.fixture
    def risk_manager(self):
        """Create configured RiskManager."""
        return RiskManager(
            max_leverage=2.0,
            target_vol_limit=0.15,
            var_limit=0.05,
            initial_capital=1000000.0
        )

    @pytest.fixture
    def composite_strategy(self, risk_manager):
        """Create CompositeAlphaStrategy instance."""
        return CompositeAlphaStrategy(risk_manager)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)

        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, 200)))
        volumes = np.random.lognormal(10, 0.5, 200)

        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.004, 200)),
            'High': prices * (1 + np.random.normal(0.004, 0.009, 200)),
            'Low': prices * (1 - np.random.normal(0.004, 0.009, 200)),
            'Close': prices,
            'Volume': volumes
        }, index=dates)

        return data

    def test_initialization(self, composite_strategy):
        """Test strategy initialization."""
        assert composite_strategy.risk_manager is not None
        assert len(composite_strategy.alpha_families) > 0
        assert len(composite_strategy.alpha_weights) > 0
        assert composite_strategy.max_alpha_weight == 0.25
        assert composite_strategy.min_alpha_weight == 0.02

    def test_alpha_family_loading(self, composite_strategy):
        """Test that alpha families are properly loaded."""
        alpha_names = [type(alpha).__name__ for alpha in composite_strategy.alpha_families]

        # Should include both old and new alpha families
        expected_alphas = [
            'MomentumTS', 'MeanReversionAlpha', 'VolatilityCarry', 'TrendStrength',
            'FundamentalAlpha', 'StatisticalAlpha', 'AlternativeAlpha', 'MLAlpha'
        ]

        for expected in expected_alphas:
            assert expected in alpha_names

    def test_regime_based_weighting(self, composite_strategy, sample_market_data):
        """Test regime-based alpha weighting."""
        # Set up different regimes
        regimes = ['BULL_QUIET', 'BEAR_CRISIS', 'HIGH_VOL']

        for regime in regimes:
            regime_context = {'regime_tag': regime}

            weights = composite_strategy._calculate_regime_weights(regime_context)

            assert isinstance(weights, dict)
            assert len(weights) == len(composite_strategy.alpha_families)

            # Weights should sum to approximately 1
            total_weight = sum(weights.values())
            assert abs(total_weight - 1.0) < 0.01

            # Individual weights should be within bounds
            for weight in weights.values():
                assert composite_strategy.min_alpha_weight <= weight <= composite_strategy.max_alpha_weight

    def test_signal_combination(self, composite_strategy, sample_market_data):
        """Test signal combination from multiple alphas."""
        # Generate signals from individual alphas
        alpha_signals = {}
        regime_context = {'regime_tag': 'BULL_QUIET'}

        for alpha in composite_strategy.alpha_families:
            try:
                signal = alpha.generate_signal(sample_market_data, regime_context)
                alpha_signals[type(alpha).__name__] = signal
            except Exception as e:
                # Some alphas might fail with synthetic data
                alpha_signals[type(alpha).__name__] = {'signal': 0.0, 'confidence': 0.0}

        # Test signal combination
        combined_result = composite_strategy._combine_signals(alpha_signals, regime_context)

        assert isinstance(combined_result, dict)
        assert 'combined_signal' in combined_result
        assert 'confidence' in combined_result
        assert 'alpha_contributions' in combined_result

        assert -1 <= combined_result['combined_signal'] <= 1
        assert 0 <= combined_result['confidence'] <= 1

    def test_risk_adjusted_signals(self, composite_strategy, sample_market_data):
        """Test risk-adjusted signal generation."""
        regime_context = {'regime_tag': 'BULL_QUIET'}

        # Test with different risk levels
        risk_levels = [0.05, 0.10, 0.20]  # Drawdown levels

        for risk_level in risk_levels:
            # Simulate risk state
            composite_strategy.risk_manager._max_equity = 1000000.0
            current_equity = 1000000.0 * (1 - risk_level)
            composite_strategy.risk_manager._max_equity = current_equity

            result = composite_strategy.generate_signal(sample_market_data, regime_context)

            assert isinstance(result, dict)
            assert 'signal' in result
            assert 'confidence' in result

            # Higher risk should lead to more conservative signals
            if risk_level > 0.15:  # High risk threshold
                assert abs(result['signal']) <= 0.5  # Should be scaled down

    def test_adaptive_alpha_weights(self, composite_strategy, sample_market_data):
        """Test adaptive alpha weight adjustment based on performance."""
        initial_weights = composite_strategy.alpha_weights.copy()

        # Simulate some performance history
        # (In practice, this would be based on actual backtest results)
        performance_scores = {
            'MomentumTS': 0.8,
            'FundamentalAlpha': 0.6,
            'StatisticalAlpha': 0.9,
            'MeanReversionAlpha': 0.4
        }

        # Update weights based on performance
        composite_strategy._update_alpha_weights(performance_scores)

        # Weights should have changed
        new_weights = composite_strategy.alpha_weights

        # High performers should have higher weights
        assert new_weights.get('StatisticalAlpha', 0) > new_weights.get('MeanReversionAlpha', 0)

    def test_signal_generation_full_pipeline(self, composite_strategy, sample_market_data):
        """Test full signal generation pipeline."""
        regime_context = {'regime_tag': 'NORMAL'}

        result = composite_strategy.generate_signal(sample_market_data, regime_context)

        assert isinstance(result, dict)
        assert all(key in result for key in ['signal', 'confidence', 'metadata'])

        # Check metadata structure
        metadata = result['metadata']
        assert 'alpha_contributions' in metadata
        assert 'regime_adjusted' in metadata
        assert 'risk_adjusted' in metadata

        # Alpha contributions should be a dictionary
        contributions = metadata['alpha_contributions']
        assert isinstance(contributions, dict)
        assert len(contributions) > 0

    def test_error_handling(self, composite_strategy):
        """Test error handling with invalid data."""
        # Test with empty dataframe
        empty_data = pd.DataFrame()

        result = composite_strategy.generate_signal(empty_data)

        assert result['signal'] == 0.0
        assert result['confidence'] == 0.0
        assert 'error' in result['metadata']

    def test_regime_transition_handling(self, composite_strategy, sample_market_data):
        """Test handling of regime transitions."""
        # Test multiple regime transitions
        regimes = ['BULL_QUIET', 'BEAR_CRISIS', 'HIGH_VOL', 'LOW_VOL']

        signals_by_regime = {}

        for regime in regimes:
            regime_context = {'regime_tag': regime}
            result = composite_strategy.generate_signal(sample_market_data, regime_context)
            signals_by_regime[regime] = result['signal']

        # Different regimes should potentially produce different signals
        # (though with synthetic data, this might not always be true)
        assert all(isinstance(signal, (int, float)) for signal in signals_by_regime.values())


class TestCompositeAlphaIntegration:
    """Integration tests combining composite alpha with risk management."""

    @pytest.fixture
    def integrated_setup(self):
        """Create integrated setup with composite alpha and risk management."""
        rm = RiskManager(initial_capital=1000000.0)
        strategy = CompositeAlphaStrategy(rm)
        return rm, strategy

    def test_signal_to_position_conversion(self, integrated_setup, sample_market_data):
        """Test conversion of composite signals to portfolio positions."""
        rm, strategy = integrated_setup

        # Generate composite signal
        signal_result = strategy.generate_signal(sample_market_data)

        # Convert to position weights
        target_weights = {'MARKET': signal_result['signal']}

        # Apply risk management
        risk_result = rm.check_pre_trade(
            target_weights=target_weights,
            baskets_returns=pd.DataFrame({'MARKET': sample_market_data['Close'].pct_change()}),
            timestamp=pd.Timestamp('2023-06-01'),
            current_equity=950000.0
        )

        assert isinstance(risk_result, dict)
        assert 'scale_factor' in risk_result

        # Final position should be risk-adjusted
        final_weight = target_weights['MARKET'] * risk_result['scale_factor']
        assert abs(final_weight) <= abs(target_weights['MARKET'])

    def test_portfolio_rebalancing_workflow(self, integrated_setup, sample_market_data):
        """Test full portfolio rebalancing workflow."""
        rm, strategy = integrated_setup

        # Step 1: Generate signals
        signals = strategy.generate_signal(sample_market_data)

        # Step 2: Create target portfolio
        target_weights = {
            'TECH': signals['signal'] * 0.4,
            'FINANCE': signals['signal'] * 0.3,
            'HEALTHCARE': signals['signal'] * 0.3
        }

        # Step 3: Risk check
        risk_check = rm.check_pre_trade(
            target_weights=target_weights,
            baskets_returns=sample_market_data['Close'].pct_change().to_frame('MARKET'),
            timestamp=pd.Timestamp.now(),
            current_equity=980000.0
        )

        # Step 4: Apply risk scaling
        if risk_check['decision'] == 'SCALE':
            scale_factor = risk_check['scale_factor']
            adjusted_weights = {k: v * scale_factor for k, v in target_weights.items()}
        else:
            adjusted_weights = target_weights

        # Verify final weights are reasonable
        total_exposure = sum(abs(w) for w in adjusted_weights.values())
        assert total_exposure <= rm.max_leverage

    def test_performance_tracking(self, integrated_setup, sample_market_data):
        """Test performance tracking and adaptation."""
        rm, strategy = integrated_setup

        # Simulate multiple signal generations
        n_periods = 5
        performance_history = []

        for i in range(n_periods):
            # Generate signal
            signal = strategy.generate_signal(sample_market_data.iloc[i*20:(i+1)*20])

            # Simulate market outcome (simplified)
            market_return = sample_market_data['Close'].pct_change().iloc[i*20:(i+1)*20].mean()
            signal_return = signal['signal'] * market_return

            performance_history.append({
                'signal': signal['signal'],
                'market_return': market_return,
                'signal_return': signal_return,
                'confidence': signal['confidence']
            })

        # Strategy should track performance
        assert len(performance_history) == n_periods
        assert all(isinstance(p['signal'], (int, float)) for p in performance_history)


if __name__ == "__main__":
    pytest.main([__file__])
