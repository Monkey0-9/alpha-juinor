"""
Quick Validation Tests for All New Modules
==========================================

Simpler tests to verify all modules import and basic functionality works.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        # Phase 1
        from ml import graph_neural_network
        print("✓ graph_neural_network")

        from ml import enhanced_portfolio_rl
        print("✓ enhanced_portfolio_rl")

        from brokers import ib_broker
        print("✓ ib_broker")

        from alternative_data.integrations import satellite_adapter
        print("✓ satellite_adapter")

        from alternative_data.integrations import credit_card_adapter
        print("✓ credit_card_adapter")

        from alternative_data.integrations import geolocation_adapter
        print("✓ geolocation_adapter")

        # Phase 2
        from hft import low_latency_engine
        print("✓ low_latency_engine")

        from risk import advanced_risk_models
        print("✓ advanced_risk_models")

        from infrastructure import cloud_native
        print("✓ cloud_native")

        from compliance import regulatory_automation
        print("✓ regulatory_automation")

        # Phase 3
        from quantum import quantum_finance
        print("✓ quantum_finance")

        from strategies.market_making import advanced_mm
        print("✓ advanced_mm")

        from data import data_lake
        print("✓ data_lake")

        print("\n✅ All imports successful!")
        return True

    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\nTesting basic functionality...")

    try:
        # Test alternative data
        from alternative_data.integrations.satellite_adapter import SatelliteAdapter
        adapter = SatelliteAdapter()
        signal = adapter.get_parking_lot_traffic("WMT", "store_123")
        assert signal is not None
        print("✓ Satellite adapter works")

        # Test IB broker
        from brokers.ib_broker import IBBrokerAdapter
        broker = IBBrokerAdapter()
        broker.connect()
        assert broker.connected
        print("✓ IB broker works")

        # Test HFT engine
        from hft.low_latency_engine import LowLatencyMarketDataHandler
        handler = LowLatencyMarketDataHandler()
        assert handler is not None
        print("✓ HFT engine works")

        # Test cloud native
        from infrastructure.cloud_native import MicroservicesArchitecture
        arch = MicroservicesArchitecture()
        assert arch is not None
        print("✓ Cloud-native works")

        # Test quantum finance
        import numpy as np

        from quantum.quantum_finance import QuantumPortfolioOptimizer
        optimizer = QuantumPortfolioOptimizer(num_assets=10)
        assert optimizer is not None
        print("✓ Quantum finance works")

        # Test market making
        from strategies.market_making.advanced_mm import InventoryManagementModel
        model = InventoryManagementModel()
        bid, ask = model.get_quotes(100, 0, 0.2)
        assert bid < ask
        print("✓ Market making works")

        print("\n✅ All basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_derivatives():
    """Test derivatives module."""
    print("\nTesting derivatives...")

    try:
        from derivatives.exotic_options import BarrierOption, price_barrier_option
        from derivatives.volatility_surface import BlackScholesModel, SABRModel
        from strategies.delta_hedging import DeltaHedgingStrategy

        # Test Black-Scholes
        price = BlackScholesModel.price(100, 100, 1.0, 0.05, 0.02, 0.2, "call")
        assert price > 0
        print("✓ Black-Scholes pricing")

        # Test SABR
        sabr = SABRModel(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        vol = sabr.implied_volatility(100, 100, 1.0)
        assert vol > 0
        print("✓ SABR model")

        # Test barrier option
        option = BarrierOption(
            S=100, K=100, H=90, T=1.0, r=0.05, q=0.02,
            sigma=0.2, option_type="call", barrier_type="down-and-out"
        )
        price = price_barrier_option(option, n_simulations=1000)
        assert price >= 0
        print("✓ Barrier option pricing")

        # Test delta hedging
        strategy = DeltaHedgingStrategy("AAPL")
        assert strategy is not None
        print("✓ Delta hedging")

        print("\n✅ Derivatives tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Derivatives test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATION TESTS FOR ALL PHASE 1-3 MODULES")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Derivatives", test_derivatives()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed")
        sys.exit(1)
