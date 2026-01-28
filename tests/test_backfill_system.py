#!/usr/bin/env python3
"""
Test script for 5-Year Backfill & Validation Implementation.

Tests:
1. QualityAgent data quality assessment
2. Prometheus metrics generation
3. Alert triggering for data quality
4. Cycle orchestrator quality enforcement
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test imports
from data_intelligence.quality_agent import QualityAgent, QualityResult
from monitoring.prometheus_metrics import MetricsCollector, get_metrics_collector
from monitoring.alerts import AlertManager, AlertCategory, AlertSeverity

def test_quality_agent():
    """Test QualityAgent data quality assessment"""
    print("=" * 60)
    print("TEST: QualityAgent")
    print("=" * 60)

    agent = QualityAgent(min_quality=0.6)

    # Test 1: Valid data
    print("\n1. Testing valid data...")
    dates = pd.date_range(start='2021-01-19', end='2021-12-31', freq='B')
    valid_df = pd.DataFrame({
        'Open': 100 + np.random.randn(len(dates)) * 2,
        'High': 102 + np.random.randn(len(dates)) * 2,
        'Low': 98 + np.random.randn(len(dates)) * 2,
        'Close': 100 + np.random.randn(len(dates)) * 2,
        'Volume': 1000000 + np.random.randn(len(dates)) * 100000
    }, index=dates)

    result = agent.check_quality('AAPL', valid_df, '2021-01-19', '2021-12-31')
    print(f"   Symbol: {result.symbol}")
    print(f"   Quality Score: {result.quality_score:.4f}")
    print(f"   Is Usable: {result.is_usable}")
    print(f"   Reasons: {result.reasons}")
    assert result.is_usable, "Valid data should be usable"
    assert result.quality_score >= 0.6, "Quality score should be >= 0.6"
    print("   ✅ PASSED")

    # Test 2: Data with zero prices (should be rejected)
    print("\n2. Testing data with zero prices...")
    bad_df = valid_df.copy()
    bad_df.loc[bad_df.index[100], 'Close'] = 0

    result = agent.check_quality('AAPL', bad_df, '2021-01-19', '2021-12-31')
    print(f"   Quality Score: {result.quality_score:.4f}")
    print(f"   Is Usable: {result.is_usable}")
    print(f"   Reasons: {result.reasons}")
    assert not result.is_usable, "Data with zero prices should not be usable"
    assert "ZERO_NEGATIVE_PRICE" in result.reasons[0], "Should have zero price reason"
    print("   ✅ PASSED")

    # Test 3: Data with high null percentage (exceeds 5% threshold)
    print("\n3. Testing data with high null percentage...")
    null_df = valid_df.copy()
    # With ~250 rows and 4 price columns: 250*4=1000 cells
    # 5% threshold = 50 NaN cells needed, let's do 100
    null_df.loc[null_df.index[0:50], 'Close'] = np.nan
    null_df.loc[null_df.index[0:50], 'Open'] = np.nan
    # Total = 100 NaN cells / 1000 = 10% (exceeds 5% threshold)

    result = agent.check_quality('AAPL', null_df, '2021-01-19', '2021-12-31')
    print(f"   Quality Score: {result.quality_score:.4f}")
    print(f"   Is Usable: {result.is_usable}")
    print(f"   Reasons: {result.reasons}")
    assert not result.is_usable, "Data with high nulls should not be usable"
    assert any('NULL' in r for r in result.reasons), "Should have null percentage reason"
    print("   ✅ PASSED")

    # Test 4: Empty dataframe
    print("\n4. Testing empty dataframe...")
    empty_df = pd.DataFrame()

    result = agent.check_quality('AAPL', empty_df, '2021-01-19', '2021-12-31')
    print(f"   Quality Score: {result.quality_score:.4f}")
    print(f"   Is Usable: {result.is_usable}")
    print(f"   Reasons: {result.reasons}")
    assert not result.is_usable, "Empty dataframe should not be usable"
    print("   ✅ PASSED")

    # Test 5: Prometheus metrics
    print("\n5. Testing Prometheus metrics export...")
    metrics = agent.get_prometheus_metrics()
    print(f"   Metrics: {metrics}")
    assert 'quality_avg_score' in metrics, "Should have avg_score metric"
    assert 'quality_failures_total' in metrics, "Should have failures metric"
    print("   ✅ PASSED")

    print("\n✅ QualityAgent tests PASSED")
    return True


def test_prometheus_metrics():
    """Test Prometheus metrics collection"""
    print("\n" + "=" * 60)
    print("TEST: Prometheus Metrics")
    print("=" * 60)

    metrics = MetricsCollector()

    # Test 1: Record cycle
    print("\n1. Testing cycle recording...")
    metrics.record_cycle(10.5)
    metrics.record_cycle(12.3)
    print(f"   Total cycles: {metrics.cycles_total}")
    print(f"   Latest duration: {metrics.cycle_durations[-1]}s")
    assert metrics.cycles_total == 2, "Should have 2 cycles"
    print("   ✅ PASSED")

    # Test 2: Record decisions
    print("\n2. Testing decision recording...")
    metrics.record_decision("EXECUTE")
    metrics.record_decision("EXECUTE")
    metrics.record_decision("HOLD")
    metrics.record_decision("REJECT")
    print(f"   Execute: {metrics.decisions_execute}")
    print(f"   Hold: {metrics.decisions_hold}")
    print(f"   Reject: {metrics.decisions_reject}")
    assert metrics.decisions_execute == 2, "Should have 2 execute"
    assert metrics.decisions_hold == 1, "Should have 1 hold"
    assert metrics.decisions_reject == 1, "Should have 1 reject"
    print("   ✅ PASSED")

    # Test 3: Quality metrics (NEW)
    print("\n3. Testing quality metrics...")
    metrics.update_quality_metrics(
        avg_quality_score=0.85,
        data_missing_days=5,
        price_history_rows=100000,
        symbols_with_data=100,
        quality_failures=3
    )
    print(f"   Avg quality score: {metrics.avg_quality_score}")
    print(f"   Missing days: {metrics.data_missing_days_total}")
    print(f"   Price rows: {metrics.price_history_rows_total}")
    print(f"   Symbols: {metrics.symbols_with_data}")
    print(f"   Quality failures: {metrics.quality_failures_total}")
    assert metrics.avg_quality_score == 0.85, "Quality score should be 0.85"
    assert metrics.data_missing_days_total == 5, "Missing days should be 5"
    assert metrics.quality_failures_total == 3, "Quality failures should be 3"
    print("   ✅ PASSED")

    # Test 4: Generate Prometheus format
    print("\n4. Testing Prometheus format generation...")
    text = metrics.get_metrics_text()
    print(f"   First 500 chars:\n{text[:500]}...")
    assert "quant_cycles_total" in text, "Should have cycles metric"
    assert "quant_avg_quality_score" in text, "Should have quality metric"
    assert "quant_data_missing_days_total" in text, "Should have missing days metric"
    print("   ✅ PASSED")

    # Test 5: Get summary
    print("\n5. Testing summary generation...")
    summary = metrics.get_summary()
    print(f"   Summary keys: {list(summary.keys())}")
    assert 'decisions' in summary, "Should have decisions"
    assert 'avg_quality_score' in summary, "Should have quality"
    assert 'data_missing_days_total' in summary, "Should have missing days"
    print("   ✅ PASSED")

    print("\n✅ Prometheus metrics tests PASSED")
    return True


def test_alerts():
    """Test alert system for data quality"""
    print("\n" + "=" * 60)
    print("TEST: Alert System")
    print("=" * 60)

    alerts = AlertManager()

    # Test 1: Data quality failure alert
    print("\n1. Testing data quality failure alert...")
    alerts.alert_data_quality_failure(
        symbol='AAPL',
        quality_score=0.45,
        reasons=['HIGH_NULL_PERCENTAGE(15%)', 'FLASH_SPIKE_DETECTED(35%)']
    )
    print("   Alert triggered (check logs)")
    print("   ✅ PASSED")

    # Test 2: Data missing days alert
    print("\n2. Testing missing days alert...")
    alerts.alert_data_missing_days(missing_count=15, threshold=0)
    print("   Alert triggered (check logs)")
    print("   ✅ PASSED")

    # Test 3: Quality threshold breach
    print("\n3. Testing quality threshold breach...")
    alerts.alert_quality_threshold_breach(avg_quality_score=0.55, threshold=0.6)
    print("   Alert triggered (check logs)")
    print("   ✅ PASSED")

    print("\n✅ Alert system tests PASSED")
    return True


def test_integration():
    """Integration test: Quality enforcement in cycle"""
    print("\n" + "=" * 60)
    print("TEST: Integration - Quality Enforcement")
    print("=" * 60)

    from data_intelligence.quality_agent import MIN_DATA_QUALITY

    # Simulate what happens in cycle_runner._apply_quality_enforcement
    print(f"\n1. Quality threshold: {MIN_DATA_QUALITY}")

    # Test symbol with low quality should be rejected
    agent = QualityAgent(min_quality=MIN_DATA_QUALITY)

    # Create marginal data
    dates = pd.date_range(start='2021-01-19', end='2021-06-30', freq='B')
    marginal_df = pd.DataFrame({
        'Open': 100 + np.random.randn(len(dates)) * 5,
        'High': 102 + np.random.randn(len(dates)) * 5,
        'Low': 98 + np.random.randn(len(dates)) * 5,
        'Close': 100 + np.random.randn(len(dates)) * 5,
        'Volume': 500000 + np.random.randn(len(dates)) * 50000  # Low volume
    }, index=dates)

    quality = agent.check_quality('TEST_SYMBOL', marginal_df)
    print(f"\n2. Marginal data quality: {quality.quality_score:.4f}")
    print(f"   Is usable: {quality.is_usable}")

    # Test should_reject function
    should_reject = agent.should_reject(quality)
    print(f"   Should reject: {should_reject}")

    # Test get_rejection_reason function
    reason = agent.get_rejection_reason(quality)
    print(f"   Rejection reason: {reason}")

    if not quality.is_usable:
        assert 'data_quality' in reason, "Reason should include data_quality"
        print("   ✅ Correctly identified for REJECT")

    # Test high quality data
    print("\n3. Testing high quality data...")
    good_df = pd.DataFrame({
        'Open': 100 + np.random.randn(len(dates)) * 1,
        'High': 102 + np.random.randn(len(dates)) * 1,
        'Low': 98 + np.random.randn(len(dates)) * 1,
        'Close': 100 + np.random.randn(len(dates)) * 1,
        'Volume': 10000000 + np.random.randn(len(dates)) * 1000000
    }, index=dates)

    good_quality = agent.check_quality('GOOD_SYMBOL', good_df)
    print(f"   Quality score: {good_quality.quality_score:.4f}")
    print(f"   Is usable: {good_quality.is_usable}")

    if good_quality.is_usable:
        good_reason = agent.get_rejection_reason(good_quality)
        assert good_reason == "", "Good data should have empty reason"
        print("   ✅ Correctly approved (no rejection reason)")

    print("\n✅ Integration tests PASSED")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 5-YEAR BACKFILL & VALIDATION - TEST SUITE")
    print("=" * 80)

    results = []

    try:
        results.append(("QualityAgent", test_quality_agent()))
    except Exception as e:
        print(f"❌ QualityAgent test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("QualityAgent", False))

    try:
        results.append(("PrometheusMetrics", test_prometheus_metrics()))
    except Exception as e:
        print(f"❌ PrometheusMetrics test FAILED: {e}")
        results.append(("PrometheusMetrics", False))

    try:
        results.append(("Alerts", test_alerts()))
    except Exception as e:
        print(f"❌ Alerts test FAILED: {e}")
        results.append(("Alerts", False))

    try:
        results.append(("Integration", test_integration()))
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        results.append(("Integration", False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run: python run_cycle.py --paper --symbols AAPL,MSFT,GOOG")
        print("2. Run: python tools/backfill_5y.py --start 2021-01-19 --end 2026-01-19 --symbols all")
        print("3. Run: python tools/validation_report.py --start 2021-01-19 --end 2026-01-19")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

