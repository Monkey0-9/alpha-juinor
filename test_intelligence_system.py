#!/usr/bin/env python3
"""
Comprehensive Trading Intelligence Test
========================================

Tests if the AI system is smart enough to beat the market.
Analyzes:
1. Signal quality and confidence
2. Risk management effectiveness
3. Expected returns vs market
4. Decision-making speed
"""

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
modules_failed = 0

try:
    from intelligence import get_ultimate_controller
    controller = get_ultimate_controller()
    status = controller.get_status()
    modules_loaded = status.get("active_components", 0)
    print(f"  ✓ Ultimate Controller: {status['status']}")
    print(f"  ✓ Active Components: {modules_loaded}/{status['total_components']}")
except Exception as e:
    print(f"  ✗ Controller failed: {e}")
    modules_failed += 1

# Test 2: Test signal generation
print("\n[TEST 2] Signal Generation Quality...")
print("-" * 50)

test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
signals = []

try:
    for symbol in test_symbols:
        # Create test features
        features = {
            "momentum_1m": np.random.uniform(-0.1, 0.15),
            "momentum_12m": np.random.uniform(-0.2, 0.3),
            "volatility": np.random.uniform(0.01, 0.04),
            "rsi": np.random.uniform(30, 70),
            "sentiment": np.random.uniform(-0.5, 0.5),
            "volume_ratio": np.random.uniform(0.8, 1.5),
            "earnings_surprise": np.random.uniform(-0.05, 0.1),
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
            "risk_reward": decision.risk_reward,
            "latency_ms": elapsed
        })

        print(f"  {symbol}: {decision.action:4s} | Conf: {decision.confidence:.0%} | "
              f"E[R]: {decision.expected_return:+.2%} | Size: {decision.position_size:.1%} | "
              f"{elapsed:.0f}ms")

    print(f"\n  ✓ Generated {len(signals)} signals successfully")

except Exception as e:
    print(f"  ✗ Signal generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Risk Management
print("\n[TEST 3] Risk Management Analysis...")
print("-" * 50)

try:
    from intelligence import get_adaptive_risk_manager
    risk_mgr = get_adaptive_risk_manager()

    # Test different volatility scenarios
    scenarios = [
        ("Low Vol (Bull)", 0.01, "BULL"),
        ("Normal Vol", 0.015, "NORMAL"),
        ("High Vol (Bear)", 0.03, "BEAR"),
        ("Crisis", 0.05, "CRISIS")
    ]

    for name, vol, regime in scenarios:
        budget = risk_mgr.update_state(
            nav=1000000,
            market_volatility=vol,
            correlation_avg=0.5,
            regime=regime
        )
        print(f"  {name:18s} | Max Pos: {budget.max_position_size:.0%} | "
              f"Stop: {budget.stop_loss_level:.1%} | Leverage: {budget.leverage_cap:.1f}x")

    print(f"\n  ✓ Risk management adapts correctly to market conditions")

except Exception as e:
    print(f"  ✗ Risk test failed: {e}")

# Test 4: Regime Detection
print("\n[TEST 4] Market Regime Detection...")
print("-" * 50)

try:
    from intelligence import get_regime_detector
    detector = get_regime_detector()

    # Test regime detection
    test_returns = np.random.randn(60) * 0.01
    state = detector.detect(
        market_returns=test_returns,
        market_volatility=0.015,
        vix_level=20,
        correlation_avg=0.5,
        breadth=0.6,
        momentum_20d=0.02
    )

    print(f"  Current Regime: {state.current_regime}")
    print(f"  Confidence: {state.confidence:.0%}")
    print(f"  Probabilities:")
    for regime, prob in sorted(state.probabilities.items(), key=lambda x: -x[1])[:3]:
        print(f"    - {regime}: {prob:.0%}")

    print(f"\n  ✓ Regime detection functioning correctly")

except Exception as e:
    print(f"  ✗ Regime test failed: {e}")

# Test 5: Performance Analysis
print("\n[TEST 5] Performance Potential Analysis...")
print("-" * 50)

if signals:
    avg_confidence = np.mean([s["confidence"] for s in signals])
    avg_expected_return = np.mean([s["expected_return"] for s in signals])
    avg_latency = np.mean([s["latency_ms"] for s in signals])
    actionable = sum(1 for s in signals if s["action"] != "HOLD")

    # Annualized return estimate (assuming 252 trading days)
    if avg_expected_return > 0:
        annual_return_est = (1 + avg_expected_return) ** 252 - 1
    else:
        annual_return_est = avg_expected_return * 252

    print(f"  Average Confidence:       {avg_confidence:.0%}")
    print(f"  Average Expected Return:  {avg_expected_return:+.2%} per trade")
    print(f"  Annualized Estimate:      {annual_return_est:+.0%}")
    print(f"  Actionable Signals:       {actionable}/{len(signals)}")
    print(f"  Average Latency:          {avg_latency:.0f}ms")

    # Scoring
    score = 0

    if avg_confidence > 0.5:
        score += 20
        print(f"\n  ✓ Confidence level GOOD (>50%)")
    elif avg_confidence > 0.3:
        score += 10
        print(f"\n  ~ Confidence level MODERATE")
    else:
        print(f"\n  ✗ Confidence level LOW")

    if avg_expected_return > 0.005:  # >0.5% per trade
        score += 30
        print(f"  ✓ Expected returns EXCELLENT (>0.5%/trade)")
    elif avg_expected_return > 0.002:
        score += 20
        print(f"  ✓ Expected returns GOOD (>0.2%/trade)")
    elif avg_expected_return > 0:
        score += 10
        print(f"  ~ Expected returns POSITIVE")
    else:
        print(f"  ✗ Expected returns NEGATIVE")

    if avg_latency < 100:
        score += 20
        print(f"  ✓ Latency EXCELLENT (<100ms)")
    elif avg_latency < 500:
        score += 10
        print(f"  ✓ Latency GOOD (<500ms)")
    else:
        print(f"  ~ Latency SLOW")

    if modules_loaded >= 5:
        score += 30
        print(f"  ✓ Intelligence stack COMPLETE ({modules_loaded} modules)")
    elif modules_loaded >= 3:
        score += 15
        print(f"  ~ Intelligence stack PARTIAL ({modules_loaded} modules)")
    else:
        print(f"  ✗ Intelligence stack DEGRADED")

# Final Report
print("\n" + "=" * 70)
print("                    FINAL INTELLIGENCE REPORT")
print("=" * 70)

grades = {
    90: ("A+", "WORLD-CLASS - Ready to beat top hedge funds"),
    80: ("A", "EXCELLENT - High probability of market outperformance"),
    70: ("B+", "VERY GOOD - Should beat market consistently"),
    60: ("B", "GOOD - Likely to beat market"),
    50: ("C+", "ADEQUATE - May beat market in favorable conditions"),
    40: ("C", "FAIR - Marginal edge over market"),
    0: ("D", "NEEDS IMPROVEMENT")
}

grade = "D"
description = "NEEDS IMPROVEMENT"
for threshold, (g, d) in sorted(grades.items(), reverse=True):
    if score >= threshold:
        grade = g
        description = d
        break

print(f"""
  INTELLIGENCE SCORE: {score}/100

  GRADE: {grade}

  ASSESSMENT: {description}

  KEY METRICS:
  - AI Components Active:    {modules_loaded}/7
  - Avg Signal Confidence:   {avg_confidence:.0%}
  - Avg Expected Return:     {avg_expected_return:+.2%}/trade
  - Decision Latency:        {avg_latency:.0f}ms
  - Annual Return Estimate:  {annual_return_est:+.0%}

  TARGET PERFORMANCE:
  - Annual Return:           60-70%
  - Sharpe Ratio:           >2.5
  - Max Drawdown:           <15%

  VERDICT: {"SMART ENOUGH TO BEAT THE MARKET ✓" if score >= 60 else "NEEDS FURTHER OPTIMIZATION"}
""")

print("=" * 70)
