"""
Debug script to understand data issue
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/c/mini-quant-fund")

from mini_quant_fund.ml_alpha.advanced_features import AdvancedFeatureEngineer

# Generate sample data - NEED MORE DATA FOR 200-PERIOD MOVING AVERAGE
np.random.seed(42)
close_prices = [100.0]

for i in range(499):  # 500 samples instead of 100
    change = np.random.normal(0.005, 0.02)
    close_prices.append(close_prices[-1] * (1 + change))

close_prices = np.array(close_prices)
opens = close_prices * np.random.uniform(0.99, 1.01, len(close_prices))
highs = np.maximum(opens, close_prices) * np.random.uniform(
    1.0, 1.02, len(close_prices)
)
lows = np.minimum(opens, close_prices) * np.random.uniform(0.98, 1.0, len(close_prices))
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

print(f"Original data shape: {df.shape}")

# Compute features
engineer = AdvancedFeatureEngineer()
features = engineer.compute_all_features(df)

print(f"Features shape: {features.shape}")
print(f"Features columns: {list(features.columns)}")
print(f"Features head:\n{features.head()}")
print(f"Features tail:\n{features.tail()}")
print(f"Features null count:\n{features.isnull().sum().sum()} nulls total")

# Create target
y = (df["close"].shift(-5) > df["close"]).astype(int)
print(f"\nTarget shape: {y.shape}")
print(f"Target null count: {y.isnull().sum()} nulls")

# Select first 320 (out of 500) for training
X_train = features.iloc[:320]
y_train = y.iloc[:320]

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Check correlations
print(f"\nCorrelations with target:")
correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
print(correlations.head(10))
print(f"Features with correlation > 0.05: {(correlations > 0.05).sum()}")
