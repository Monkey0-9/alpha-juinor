"""
Quick test to verify alpha models work correctly.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from alpha_families.fundamental_alpha import FundamentalAlpha
from alpha_families.statistical_alpha import StatisticalAlpha
from alpha_families.alternative_alpha import AlternativeAlpha
from alpha_families.ml_alpha import MLAlpha

# Create sample data
dates = pd.date_range('2023-01-01', periods=200, freq='D')
np.random.seed(42)
prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 200)))
volumes = np.random.lognormal(10, 0.5, 200)

data = pd.DataFrame({
    'Open': prices * (1 + np.random.normal(0, 0.005, 200)),
    'High': prices * (1 + np.random.normal(0.005, 0.01, 200)),
    'Low': prices * (1 - np.random.normal(0.005, 0.01, 200)),
    'Close': prices,
    'Volume': volumes
}, index=dates)

print("Testing Alpha Models...")
print("=" * 60)

# Test Fundamental Alpha
print("\n1. Testing FundamentalAlpha...")
fundamental = FundamentalAlpha()
result = fundamental.generate_signal(data)
print(f"   Signal: {result['signal']:.4f}, Confidence: {result['confidence']:.4f}")
assert 'signal' in result
assert 'confidence' in result
assert -1 <= result['signal'] <= 1
print("   ✓ FundamentalAlpha passed")

# Test Statistical Alpha
print("\n2. Testing StatisticalAlpha...")
statistical = StatisticalAlpha()
result = statistical.generate_signal(data)
print(f"   Signal: {result['signal']:.4f}, Confidence: {result['confidence']:.4f}")
assert 'signal' in result
assert 'confidence' in result
assert -1 <= result['signal'] <= 1
print("   ✓ StatisticalAlpha passed")

# Test Alternative Alpha
print("\n3. Testing AlternativeAlpha...")
alternative = AlternativeAlpha()
result = alternative.generate_signal(data)
print(f"   Signal: {result['signal']:.4f}, Confidence: {result['confidence']:.4f}")
assert 'signal' in result
assert 'confidence' in result
assert -1 <= result['signal'] <= 1
print("   ✓ AlternativeAlpha passed")

# Test ML Alpha
print("\n4. Testing MLAlpha...")
ml_alpha = MLAlpha()
result = ml_alpha.generate_signal(data)
print(f"   Signal: {result['signal']:.4f}, Confidence: {result['confidence']:.4f}")
assert 'signal' in result
assert 'confidence' in result
assert -1 <= result['signal'] <= 1
print("   ✓ MLAlpha passed")

print("\n" + "=" * 60)
print("✓ All alpha models working correctly!")
print("=" * 60)
