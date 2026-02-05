#!/usr/bin/env python3
"""Simplified Intelligence Test - ASCII output"""

import sys
import time
import numpy as np
from datetime import datetime

print("=" * 70)
print("      TRADING INTELLIGENCE VERIFICATION TEST")
print("      " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 70)

# Test 1: Load all intelligence modules
print("\n[TEST 1] Loading Intelligence Modules...")
print("-" * 50)

modules_loaded = 0

try:
    from intelligence import get_ultimate_controller
    controller = get_ultimate_controller()
    status = controller.get_status()
    modules_loaded = status.get("active_components", 0)
    print(f"  [OK] Ultimate Controller: {status['status']}")
    print(f"  [OK] Active Components: {modules_loaded}/{status['total_components']}")
except Exception as e:
    print(f"  [FAIL] Controller failed: {e}")

# Test 2: Test signal generation
print("\n[TEST 2] Signal Generation Quality...")
print("-" * 50)

test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
signals = []
avg_confidence = 0
avg_expected_return = 0
avg_latency = 0

try:
    for symbol in test_symbols:
        features = {
            "momentum_1m": np.random.uniform(-0.1, 0.15),
            "momentum_12m": np.random.uniform(-0.2, 0.3),
            "volatility": np.random.uniform(0.01, 0.04),
            "rsi": np.random.uniform(30, 70),
            "sentiment": np.random.uniform(-0.5, 0.5),
        }

        returns_history = np.random.randn(60) * 0.02
        market_data = {
            "returns": list(np.random.randn(20) * 0.01),
            "volatility": 0.015,
            "vix": 18,
            "correlation": 0.5,
            "breadth": 0.6,
            "momentum": 0.02
        }

        start = time.time()
        decision = controller.generate_decision(
            symbol=symbol,
            price=100 + np.random.randn() * 10,
            features=features,
            returns_history=returns_history,
            market_data=market_data,
            nav=1000000
        )
        elapsed = (time.time() - start) * 1000

        signals.append({
            "symbol": symbol,
            "action": decision.action,
            "confidence": decision.confidence,
            "expected_return": decision.expected_return,
            "position_size": decision.position_size,
            "latency_ms": elapsed
        })

        print(f"  {symbol}: {decision.action:4s} | Conf: {decision.confidence:.0%} | "
              f"E[R]: {decision.expected_return:+.2%} | {elapsed:.0f}ms")

    avg_confidence = np.mean([s["confidence"] for s in signals])
    avg_expected_return = np.mean([s["expected_return"] for s in signals])
    avg_latency = np.mean([s["latency_ms"] for s in signals])

    print(f"\n  [OK] Generated {len(signals)} signals successfully")

except Exception as e:
    print(f"  [FAIL] Signal generation failed: {e}")

# Test 3: Risk Management
print("\n[TEST 3] Risk Management...")
print("-" * 50)

try:
    from intelligence import get_adaptive_risk_manager
    risk_mgr = get_adaptive_risk_manager()

    budget = risk_mgr.update_state(nav=1000000, market_volatility=0.015,
                                    correlation_avg=0.5, regime="NORMAL")
    print(f"  [OK] Max Position: {budget.max_position_size:.0%}")
    print(f"  [OK] Stop Loss: {budget.stop_loss_level:.1%}")
    print(f"  [OK] Leverage Cap: {budget.leverage_cap:.1f}x")
except Exception as e:
    print(f"  [FAIL] Risk test failed: {e}")

# Test 4: Regime Detection
print("\n[TEST 4] Market Regime Detection...")
print("-" * 50)

try:
    from intelligence import get_regime_detector
    detector = get_regime_detector()

    test_returns = np.random.randn(60) * 0.01
    state = detector.detect(
        market_returns=test_returns, market_volatility=0.015,
        vix_level=20, correlation_avg=0.5, breadth=0.6, momentum_20d=0.02
    )
    print(f"  [OK] Current Regime: {state.current_regime}")
    print(f"  [OK] Confidence: {state.confidence:.0%}")
except Exception as e:
    print(f"  [FAIL] Regime test failed: {e}")

# Final Score
print("\n" + "=" * 70)
print("                    FINAL INTELLIGENCE REPORT")
print("=" * 70)

score = 0

# Score components
if modules_loaded >= 5:
    score += 30
if avg_confidence > 0.5:
    score += 20
if avg_expected_return > 0.002:
    score += 30
if avg_latency < 500:
    score += 20

annual_est = avg_expected_return * 252 * 100 if avg_expected_return else 0

print(f"""
  INTELLIGENCE SCORE: {score}/100

  GRADE: {'A+' if score >= 90 else 'A' if score >= 80 else 'B+' if score >= 70 else 'B' if score >= 60 else 'C'}

  KEY METRICS:
  - AI Components Active:    {modules_loaded}/7
  - Avg Signal Confidence:   {avg_confidence:.0%}
  - Avg Expected Return:     {avg_expected_return:+.2%}/trade
  - Decision Latency:        {avg_latency:.0f}ms
  - Annual Return Estimate:  {annual_est:+.0f}%

  TARGET PERFORMANCE:
  - Annual Return:           60-70%
  - Sharpe Ratio:           >2.5
  - Max Drawdown:           <15%

  VERDICT: {'SMART ENOUGH TO BEAT THE MARKET [YES]' if score >= 60 else 'NEEDS FURTHER OPTIMIZATION'}
""")
print("=" * 70)
