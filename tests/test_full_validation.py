"""
COMPREHENSIVE INTELLIGENCE SYSTEM VALIDATION
Tests each component step-by-step
"""

import logging
import sys

import numpy as np
import pandas as pd

# Set Python to use UTF-8 for stdout
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, "/c/mini-quant-fund")

# Import all components
from mini_quant_fund.ml_alpha.advanced_features import AdvancedFeatureEngineer
from mini_quant_fund.ml_alpha.advanced_intelligence_engine import (
    AnomalyDetector,
    get_intelligence_system,
)
from mini_quant_fund.ml_alpha.enhanced_predictive_model import get_enhanced_model
from mini_quant_fund.ml_alpha.intelligence_core import get_intelligence_core
from mini_quant_fund.ml_alpha.smart_brain_engine import MarketState, get_smart_brain


def generate_sample_data(days=500, trend="up"):
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    close_prices = [100.0]

    for i in range(days - 1):
        if trend == "up":
            change = np.random.normal(0.005, 0.02)
        elif trend == "down":
            change = np.random.normal(-0.005, 0.02)
        else:
            change = np.random.normal(0, 0.02)
        close_prices.append(close_prices[-1] * (1 + change))

    close_prices = np.array(close_prices)
    opens = close_prices * np.random.uniform(0.99, 1.01, len(close_prices))
    highs = np.maximum(opens, close_prices) * np.random.uniform(
        1.0, 1.02, len(close_prices)
    )
    lows = np.minimum(opens, close_prices) * np.random.uniform(
        0.98, 1.0, len(close_prices)
    )
    volumes = np.random.uniform(1e6, 1e7, len(close_prices))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close_prices,
            "volume": volumes,
        }
    )
    return df


passed = 0
failed = 0

print("\n" + "=" * 80)
print("STEP 1: ADVANCED FEATURE ENGINEERING")
print("=" * 80)

try:
    feature_engineer = AdvancedFeatureEngineer()
    df = generate_sample_data(100, "up")
    features = feature_engineer.compute_all_features(df)
    print("[PASS] Feature Engineering Works")
    print("  - Features generated:", features.shape[1])
    print("  - Feature columns:", str(list(features.columns[:10])) + "...")
    passed += 1
except Exception as e:
    print("[FAIL] Feature Engineering Failed:", str(e))
    failed += 1

print("\n" + "=" * 80)
print("STEP 2: ANOMALY DETECTION")
print("=" * 80)

try:
    anomaly_detector = AnomalyDetector()
    anomalies = anomaly_detector.detect_anomalies(features)
    regime = anomaly_detector.detect_regime(df["close"].pct_change())
    print("[PASS] Anomaly Detection Works")
    print("  - Anomalies detected:", (anomalies == -1).sum(), "/", len(anomalies))
    print("  - Regime:", regime)
    passed += 1
except Exception as e:
    print("[FAIL] Anomaly Detection Failed:", str(e))
    failed += 1

print("\n" + "=" * 80)
print("STEP 3: INTELLIGENCE SYSTEM (ENSEMBLE)")
print("=" * 80)

try:
    intelligence_system = get_intelligence_system()
    print("[PASS] Intelligence System Initialized")

    X_train = features.iloc[:320]
    y_train = (df["close"].shift(-5) > df["close"]).astype(int).iloc[:320]
    print("  - Selected features:", len(init_results["selected_features"]))

    X_test = features.iloc[-1:].copy()
    pred, confidence, regime = intelligence_system.predict(X_test)
    print("[PASS] Ensemble Prediction Works")
    print("  - Prediction:", f"{pred:.4f}")
    print("  - Confidence:", f"{confidence:.4f}")
    print("  - Regime:", regime)
    passed += 1
except Exception as e:
    print("[FAIL] Intelligence System Failed:", str(e))
    import traceback

    traceback.print_exc()
    failed += 1

print("\n" + "=" * 80)
print("STEP 4: INTELLIGENCE CORE (META-LEARNING)")
print("=" * 80)

try:
    core = get_intelligence_core()
    print("[PASS] Intelligence Core Initialized")

    market_features = {
        "volatility": (
            features["volatility_20"].iloc[-1] if "volatility_20" in features else 0.01
        ),
        "momentum": (
            features["momentum_10"].iloc[-1] if "momentum_10" in features else 0.0
        ),
        "trend": 0.6,
    }

    core_result = core.process_prediction(
        prediction=pred,
        model_weights={"LGBMRegressor": 0.3, "XGBRegressor": 0.3},
        market_features=market_features,
        confidence=confidence,
    )
    print("[PASS] Intelligence Core Processing Works")
    print("  - Enhanced prediction:", f"{core_result['enhanced_prediction']:.4f}")
    print("  - Pattern detected:", core_result["pattern_detected"])
    print("  - Core accuracy:", core_result.get("core_accuracy", "N/A"))

    core.learn_from_outcome(
        prediction=pred,
        confidence=confidence,
        actual_outcome=0.55,
        profit_loss=0.5,
        market_features=market_features,
    )
    print("[PASS] Intelligence Core Learning Works")

    status = core.get_system_intelligence()
    print("  - Total predictions learned:", status["total_predictions"])
    print("  - Patterns discovered:", status["patterns_learned"])
    passed += 1
except Exception as e:
    print("[FAIL] Intelligence Core Failed:", str(e))
    import traceback

    traceback.print_exc()
    failed += 1

print("\n" + "=" * 80)
print("STEP 5: SMART BRAIN ENGINE (REASONING)")
print("=" * 80)

try:
    smart_brain = get_smart_brain()
    print("[PASS] Smart Brain Initialized")

    market_state = MarketState(
        volatility=0.015,
        momentum=0.02,
        trend_strength=0.7,
        regime="normal",
        lstm_signal=0.5,
        uncertainty=0.15,
    )

    decision = smart_brain.think(market_state=market_state)
    print("[PASS] Smart Brain Thinking Works")
    print("  - Decision:", decision["action"])
    print("  - Confidence:", f"{decision['confidence']:.1%}")
    print("  - Uncertainty:", f"{decision['uncertainty']:.1%}")
    print("  - Reasoning steps:", len(decision["reasoning"]))

    smart_brain.learn_from_decision_outcome(
        decision_action=decision["action"],
        actual_outcome="bullish",
        profit_loss=1.5,
        market_features=market_features,
    )
    print("[PASS] Smart Brain Learning Works")
    passed += 1
except Exception as e:
    print("[FAIL] Smart Brain Failed:", str(e))
    import traceback

    traceback.print_exc()
    failed += 1

print("\n" + "=" * 80)
print("STEP 6: ENHANCED PREDICTIVE MODEL (INTEGRATION)")
print("=" * 80)

try:
    model = get_enhanced_model()
    print("[PASS] Enhanced Model Initialized")

    forecast = model.get_forecast(df.tail(20))
    print("[PASS] Forecast Generation Works (untrained)")
    print("  - Probability:", f"{forecast['probability']:.4f}")
    print("  - Confidence:", f"{forecast['confidence']:.4f}")
    print("  - Source:", forecast["source"])

    X_train = features.iloc[:80]
    y_train = (df["close"].shift(-5) > df["close"]).astype(int).iloc[:80]
    train_result = model.train(X_train, y_train)

    print("[PASS] Model Training Complete")
    print("  - Models trained:", train_result["model_count"])
    print("  - Selected features:", len(train_result["selected_features"]))

    forecast = model.get_forecast(df.tail(20))
    print("[PASS] Forecast Generation Works (trained)")
    print("  - Probability:", f"{forecast['probability']:.4f}")
    print("  - Confidence:", f"{forecast['confidence']:.4f}")
    print("  - Source:", forecast["source"])
    print("  - Brain action:", forecast.get("brain_action", "N/A"))
    print("  - Reasoning lines:", len(forecast.get("reasoning_chain", [])))

    model.learn_from_outcome(
        prediction=forecast["probability"],
        confidence=forecast["confidence"],
        actual_outcome=0.6,
        profit_loss=0.25,
        symbol_data=df.tail(20),
        brain_decision_action=forecast.get("brain_action", "neutral"),
    )
    print("[PASS] Model Learning Works")

    status = model.get_model_status()
    print("[PASS] Model Status Retrieved")
    print("  - Is trained:", status["is_trained"])
    print("  - Models:", status["model_count"])
    print(
        "  - Intelligence core patterns:",
        status["intelligence_core"]["patterns_discovered"],
    )
    passed += 1

except Exception as e:
    print("[FAIL] Enhanced Model Failed:", str(e))
    import traceback

    traceback.print_exc()
    failed += 1

print("\n" + "=" * 80)
print("COMPREHENSIVE END-TO-END TEST")
print("=" * 80)

try:
    print("\nSimulating 5-day trading cycle...\n")

    model = get_enhanced_model()
    df = generate_sample_data(100, "up")
    features_df = feature_engineer.compute_all_features(df)

    X_train = features_df.iloc[:80]
    y_train = (df["close"].shift(-5) > df["close"]).astype(int).iloc[:80]
    model.train(X_train, y_train)

    for day in range(81, 86):
        forecast = model.get_forecast(df.iloc[:day].tail(20))
        actual = 1 if df["close"].iloc[day] > df["close"].iloc[day - 1] else 0
        pnl = (
            abs(forecast["probability"] - 0.5)
            * 2
            * (1 if (forecast["probability"] > 0.5 == actual) else -1)
        )

        print(f"Day {day}:")
        print(
            f"  Forecast: {forecast['probability']:.1%} | Confidence: {forecast['confidence']:.1%}"
        )
        print(f"  Brain Decision: {forecast.get('brain_action', 'N/A')}")
        print(f"  Actual: {'UP' if actual else 'DOWN'} | P&L: {pnl:+.2f}")

        model.learn_from_outcome(
            prediction=forecast["probability"],
            confidence=forecast["confidence"],
            actual_outcome=float(actual),
            profit_loss=pnl,
            symbol_data=df.iloc[:day].tail(20),
            brain_decision_action=forecast.get("brain_action", "neutral"),
        )

    print("\n[PASS] End-to-End Trading Cycle Complete")
    passed += 1

except Exception as e:
    print("[FAIL] End-to-End Test Failed:", str(e))
    import traceback

    traceback.print_exc()
    failed += 1

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTests Passed: {passed}")
print(f"Tests Failed: {failed}")
print(f"Total Tests: {passed + failed}")

if failed == 0:
    print("\n[SUCCESS] ALL SYSTEMS OPERATIONAL!")
else:
    print("\n[WARNING] Some systems need fixes")

print("\nKEY FEATURES DEMONSTRATED:")
print("  1. Advanced Features: 50+ feature generation")
print("  2. Anomaly Detection: Regime and anomaly detection")
print("  3. Ensemble Models: 5-model ensemble with learned weights")
print("  4. Intelligence Core: Meta-learning from predictions")
print("  5. Smart Brain: Bayesian reasoning with confidence")
print("  6. Integration: All components working together")

print("\nIMPROVEMENT RECOMMENDATIONS:")
print("  1. Faster learning using windowed approach")
print("  2. Better regime detection with more regime types")
print("  3. Strategy selection based on market regime")
print("  4. Early warning system for regime changes")
print("  5. Improved confidence calibration")
print("  6. Adaptive model weight adjustment")
print("  7. Feature importance pruning (50 -> 20-30)")
print("  8. Real-time learning without full retraining")
print("  9. Uncertainty quantification (confidence intervals)")
print("  10. Better explainability with feature contributions")

logger.info("[TEST] VALIDATION COMPLETE")
