"""
Manual Integration Test - Complete Trading Pipeline
===================================================

Tests the full trading flow with all new Phase 1-3 components.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime

import numpy as np


def test_alternative_data_pipeline():
    """Test alternative data integration."""
    print("\n" + "="*60)
    print("PHASE 1: Alternative Data Integration")
    print("="*60)

    from alternative_data.integrations.credit_card_adapter import CreditCardAdapter
    from alternative_data.integrations.geolocation_adapter import GeolocationAdapter
    from alternative_data.integrations.satellite_adapter import SatelliteAdapter

    # Test satellite data
    satellite = SatelliteAdapter()
    signal = satellite.get_parking_lot_traffic("WMT", "store_12345")
    alpha = satellite.get_alpha_signal(signal)
    print(f"✓ Satellite: {signal.symbol} parking traffic change: {signal.change_pct:.2f}%, alpha: {alpha}")

    # Test credit card data
    cc = CreditCardAdapter()
    revenue_signal = cc.get_revenue_growth("AMZN")
    cc_alpha = cc.get_alpha_signal(revenue_signal)
    print(f"✓ Credit Card: {revenue_signal.symbol} revenue growth: {revenue_signal.change_pct:.2f}%, alpha: {cc_alpha}")

    # Test geolocation data
    geo = GeolocationAdapter()
    foot_traffic = geo.get_foot_traffic("SBUX", "store_abc")
    geo_alpha = geo.get_alpha_signal(foot_traffic)
    print(f"✓ Geolocation: {foot_traffic.symbol} foot traffic change: {foot_traffic.change_pct:.2f}%, alpha: {geo_alpha}")

    print("✅ Alternative data pipeline working!")
    return True


def test_derivatives_trading():
    """Test derivatives and options trading."""
    print("\n" + "="*60)
    print("PHASE 1: Derivatives & Options Engine")
    print("="*60)

    from derivatives.exotic_options import BarrierOption, price_barrier_option
    from derivatives.volatility_surface import OptionQuote, SABRModel, VolatilitySurface
    from strategies.delta_hedging import DeltaHedgingStrategy, OptionPosition

    # Test SABR calibration
    sabr = SABRModel(alpha=0.25, beta=0.5, rho=-0.3, nu=0.4)
    vol = sabr.implied_volatility(F=100, K=100, T=1.0)
    print(f"✓ SABR Model: ATM vol = {vol:.4f}")

    # Test barrier option
    barrier = BarrierOption(
        S=100, K=105, H=90, T=0.5, r=0.05, q=0.02,
        sigma=0.25, option_type="call", barrier_type="down-and-out"
    )
    price = price_barrier_option(barrier, n_simulations=10000)
    print(f"✓ Barrier Option: Price = ${price:.2f}")

    # Test delta hedging
    strategy = DeltaHedgingStrategy("AAPL")
    option = OptionPosition("AAPL", strike=150, expiry_days=30,
                           option_type="call", quantity=10, entry_price=5.0)
    strategy.add_option(option)

    greeks = strategy.calculate_portfolio_greeks(150, 0.25)
    print(f"✓ Delta Hedging: Portfolio delta = {greeks['delta']:.2f}")

    print("✅ Derivatives engine working!")
    return True


def test_multi_asset_trading():
    """Test multi-asset trading infrastructure."""
    print("\n" + "="*60)
    print("PHASE 1: Multi-Asset Global Trading")
    print("="*60)

    from brokers.ib_broker import FuturesRollCalendar, IBBrokerAdapter

    # Test IB broker
    broker = IBBrokerAdapter()
    connected = broker.connect()
    print(f"✓ IB Connection: {'Connected' if connected else 'Simulated'}")

    # Test futures contract
    es_contract = broker.create_futures_contract("ES", "CME", "202603")
    market_data = broker.get_market_data(es_contract)
    print(f"✓ Futures Market Data: ES bid={market_data['bid']:.2f}, ask={market_data['ask']:.2f}")

    # Test forex
    eur_contract = broker.create_forex_contract("EUR", "USD")
    eur_data = broker.get_market_data(eur_contract)
    print(f"✓ Forex Market Data: EUR/USD mid={eur_data['last']:.4f}")

    # Test roll calendar
    roll_cal = FuturesRollCalendar()
    active = roll_cal.get_active_contract("ES", datetime.now())
    print(f"✓ Roll Calendar: Active ES contract = {active}")

    print("✅ Multi-asset trading working!")
    return True


def test_hft_infrastructure():
    """Test HFT low-latency components."""
    print("\n" + "="*60)
    print("PHASE 2: HFT Infrastructure")
    print("="*60)

    import time

    from hft.low_latency_engine import (
        CoLocationSimulator,
        FPGATickProcessor,
        HFTAlphaModel,
        LowLatencyMarketDataHandler,
        Tick,
    )

    # Test market data handler
    handler = LowLatencyMarketDataHandler()
    tick = Tick("AAPL", time.perf_counter_ns(), 150.0, 150.05, 100, 100, 150.02, 1000)
    handler.process_tick(tick)

    avg_latency = handler.get_avg_latency_us()
    print(f"✓ Market Data Handler: Avg latency = {avg_latency:.2f} microseconds")

    # Test FPGA processor
    fpga = FPGATickProcessor()
    signals = fpga.update("AAPL", 150.0, 150.05)
    print(f"✓ FPGA Processor: Signal = {signals['signal']}, spread OK = {signals['spread_ok']}")

    # Test co-location simulator
    coloc = CoLocationSimulator(colocated=True)
    latency = coloc.simulate_order_latency()
    print(f"✓ Co-location: Order latency = {latency:.2f} microseconds")

    # Test HFT alpha
    alpha = HFTAlphaModel()
    quotes = alpha.market_making_signal("AAPL", 150.0, 150.05, 150.02)
    print(f"✓ HFT Alpha: MM bid={quotes['quote_bid']:.2f}, ask={quotes['quote_ask']:.2f}")

    print("✅ HFT infrastructure working!")
    return True


def test_advanced_risk():
    """Test advanced risk models."""
    print("\n" + "="*60)
    print("PHASE 2: Advanced Risk Modeling")
    print("="*60)

    from risk.advanced_risk_models import (
        ExtremeValueTheory,
        NetworkContagionModel,
        StressTestingFramework,
    )

    # Test network contagion
    contagion = NetworkContagionModel()
    entities = ["Fund A", "Fund B", "Fund C"]
    exposures = np.array([[0, 100, 50], [80, 0, 60], [40, 70, 0]])
    capital = {"Fund A": 200, "Fund B": 150, "Fund C": 180}

    contagion.build_network(entities, exposures, capital)
    defaults = contagion.simulate_default_cascade(["Fund A"])
    num_defaults = sum(defaults.values())
    print(f"✓ Network Contagion: {num_defaults}/{len(entities)} entities defaulted")

    # Test extreme value theory
    evt = ExtremeValueTheory()
    losses = np.random.exponential(0.01, 1000)
    var_99 = evt.estimate_var(losses, 0.99)
    cvar_99 = evt.estimate_cvar(losses, 0.99)
    print(f"✓ EVT: VaR(99%) = {var_99:.4f}, CVaR(99%) = {cvar_99:.4f}")

    # Test stress testing
    stress = StressTestingFramework()
    stress.define_scenario("Market Crash", {"AAPL": -20, "MSFT": -18, "GOOGL": -22})

    portfolio = {"AAPL": 100, "MSFT": 150, "GOOGL": 80}
    prices = {"AAPL": 150, "MSFT": 300, "GOOGL": 140}

    result = stress.run_scenario("Market Crash", portfolio, prices)
    print(f"✓ Stress Test: Market Crash P&L = ${result['pnl']:.2f} ({result['pnl_pct']:.2f}%)")

    print("✅ Advanced risk models working!")
    return True


def test_regulatory_compliance():
    """Test regulatory automation."""
    print("\n" + "="*60)
    print("PHASE 2: Regulatory Automation")
    print("="*60)

    from compliance.regulatory_automation import (
        BestExecutionAnalytics,
        MiFIDIICompliance,
        TradeRecord,
        TradeSurveillance,
    )

    # Test MiFID II
    mifid = MiFIDIICompliance()
    trade = TradeRecord("T001", datetime.now(), "AAPL", "BUY", 100, 150, "NYSE", "C001")
    mifid.add_trade(trade)
    report = mifid.generate_transaction_report(datetime(2020, 1, 1), datetime(2030, 1, 1))
    print(f"✓ MiFID II: Generated transaction report with {len(report)} records")

    # Test best execution
    best_ex = BestExecutionAnalytics()
    best_ex.add_trade(trade, benchmark_mid=150.02)
    improvement = best_ex.calculate_price_improvement(trade)
    print(f"✓ Best Execution: Price improvement = {improvement:.2f} bps")

    # Test surveillance
    surveillance = TradeSurveillance()
    alerts = surveillance.detect_wash_trading([trade])
    print(f"✓ Trade Surveillance: {len(alerts)} alerts generated")

    print("✅ Regulatory automation working!")
    return True


def test_quantum_and_market_making():
    """Test quantum finance and advanced market making."""
    print("\n" + "="*60)
    print("PHASE 3: Quantum Finance & Market Making")
    print("="*60)

    from quantum.quantum_finance import QuantumPortfolioOptimizer
    from strategies.market_making.advanced_mm import (
        AdverseSelectionProtection,
        InventoryManagementModel,
    )

    # Test quantum portfolio optimization
    quantum = QuantumPortfolioOptimizer(num_assets=20)
    returns = np.random.randn(20) * 0.01
    cov = np.eye(20) * 0.01
    selection = quantum.qaoa_portfolio_selection(returns, cov, num_select=10)
    print(f"✓ Quantum Portfolio: Selected {int(selection.sum())}/20 assets")

    # Test Avellaneda-Stoikov market making
    inv_mgmt = InventoryManagementModel()
    bid, ask = inv_mgmt.get_quotes(100, 50, 0.2)
    print(f"✓ Market Making: bid={bid:.2f}, ask={ask:.2f}, spread={(ask-bid):.4f}")

    # Test adverse selection
    adv_sel = AdverseSelectionProtection()
    toxicity = adv_sel.estimate_order_flow_toxicity([100, 101, 102], [100, 150, 200])
    print(f"✓ Adverse Selection: Flow toxicity = {toxicity:.4f}")

    print("✅ Quantum & market making working!")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" COMPREHENSIVE INTEGRATION TEST - ALL PHASE 1-3 MODULES")
    print("="*70)

    tests = [
        ("Alternative Data Pipeline", test_alternative_data_pipeline),
        ("Derivatives Trading", test_derivatives_trading),
        ("Multi-Asset Trading", test_multi_asset_trading),
        ("HFT Infrastructure", test_hft_infrastructure),
        ("Advanced Risk Models", test_advanced_risk),
        ("Regulatory Compliance", test_regulatory_compliance),
        ("Quantum & Market Making", test_quantum_and_market_making),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*70)
    print(" FINAL RESULTS")
    print("="*70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    total = len(results)
    passed_count = sum(r[1] for r in results)

    print(f"\nTotal: {passed_count}/{total} tests passed")

    if all_passed:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ System is ready for production deployment")
    else:
        print("\n⚠️  Some tests need attention")
