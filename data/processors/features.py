"""
data/processors/features.py
Institutional Feature Engineering Pipeline with Contract Enforcement
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

from features.contract import load_feature_contract, validate_contract_compliance
from utils.errors import FeatureValidationError

logger = logging.getLogger(__name__)


def compute_features_for_symbol(
    prices: pd.DataFrame,
    contract_name: str = "ml_v1"
) -> pd.DataFrame:
    """
    Compute features for a symbol with strict contract enforcement.

    This is the SINGLE SOURCE OF TRUTH for feature computation.
    All features MUST match the contract exactly (order + dtype).

    Args:
        prices: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
        contract_name: Feature contract to use (default: "ml_v1")

    Returns:
        DataFrame with exactly contract.n_features columns in contract order,
        dtype float32

    Raises:
        FeatureValidationError: If unable to compute required features

    Example:
        >>> df = pd.DataFrame({'Open': [...], 'High': [...], ...})
        >>> features = compute_features_for_symbol(df)
        >>> assert list(features.columns) == load_feature_contract()["features"]
        >>> assert features.dtypes.unique() == [np.float32]
    """
    # Load contract
    contract = load_feature_contract(contract_name)
    required_features = contract["features"]

    # Minimum data requirement
    if len(prices) < 252:  # Need at least 1 year for 252-day features
        raise FeatureValidationError(
            f"Insufficient data: need at least 252 rows, got {len(prices)}"
        )

    # Ensure we have required OHLCV columns
    required_cols = ["Close", "High", "Low", "Open", "Volume"]
    missing_cols = [c for c in required_cols if c not in prices.columns]
    if missing_cols:
        raise FeatureValidationError(f"Missing required price columns: {missing_cols}")

    df = prices.copy()
    features = pd.DataFrame(index=df.index)

    # =========================================================================
    # FEATURE COMPUTATION - ML v1 Contract (28 features)
    # =========================================================================

    # 1. Returns Features (ret_1d, ret_5d, ret_10d, ret_20d)
    features["ret_1d"] = df["Close"].pct_change(1, fill_method=None)
    features["ret_5d"] = df["Close"].pct_change(5, fill_method=None)
    features["ret_10d"] = df["Close"].pct_change(10, fill_method=None)
    features["ret_20d"] = df["Close"].pct_change(20, fill_method=None)

    # 2. Volatility Features (vol_5d, vol_20d, vol_60d)
    ret_series = df["Close"].pct_change(fill_method=None)
    features["vol_5d"] = ret_series.rolling(5).std() * np.sqrt(252)
    features["vol_20d"] = ret_series.rolling(20).std() * np.sqrt(252)
    features["vol_60d"] = ret_series.rolling(60).std() * np.sqrt(252)

    # 3. RSI_14
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    # 4. MACD & Signal
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    features["macd"] = ema_12 - ema_26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()

    # 5. ATR_14
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features["atr_14"] = true_range.rolling(14).mean()

    # 6. Momentum (momentum_10, momentum_20)
    features["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    features["momentum_20"] = df["Close"] / df["Close"].shift(20) - 1

    # 7. Z-Score Features (zscore_5, zscore_20)
    mean_5 = df["Close"].rolling(5).mean()
    std_5 = df["Close"].rolling(5).std()
    features["zscore_5"] = (df["Close"] - mean_5) / std_5.replace(0, np.nan)

    mean_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    features["zscore_20"] = (df["Close"] - mean_20) / std_20.replace(0, np.nan)

    # 8. Cross-Rank Sector (placeholder - needs sector data, use neutral for now)
    features["cross_rank_sector"] = 0.5

    # 9. Beta_60 (rolling beta vs market proxy)
    # Beta = Cov(stock, market) / Var(market)
    # Since we don't have SPY in this function, we estimate from volatility ratio
    # Fallback: estimate from volatility ratio relative to typical market vol (~15% annual)
    market_vol_annual = 0.15
    stock_vol = features["vol_60d"].fillna(market_vol_annual).replace(0, market_vol_annual)
    features["beta_60"] = (stock_vol / market_vol_annual).clip(0.5, 2.5)

    # 10. Turnover_20d
    # Turnover = Volume / Average Volume
    avg_volume_20 = df["Volume"].rolling(20).mean()
    features["turnover_20d"] = df["Volume"] / avg_volume_20.replace(0, np.nan)

    # 11. Skew_60 & Kurt_60
    ret_60 = df["Close"].pct_change(fill_method=None)
    features["skew_60"] = ret_60.rolling(60).skew()
    features["kurt_60"] = ret_60.rolling(60).kurt()

    # 12. Adjclose_ratio
    # Ratio of Close to its 20-day SMA
    sma_20 = df["Close"].rolling(20).mean()
    features["adjclose_ratio"] = df["Close"] / sma_20.replace(0, np.nan)

    # 13. EMA Features (ema_12, ema_26)
    features["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    features["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # 14. SMA Features (sma_50, sma_200)
    features["sma_50"] = df["Close"].rolling(50).mean()
    features["sma_200"] = df["Close"].rolling(200).mean()

    # 15. Volatility Ratio
    # Ratio of short-term to long-term volatility
    vol_ratio = features["vol_5d"] / features["vol_20d"].replace(0, np.nan)
    features["volatility_ratio"] = vol_ratio

    # 16. Regime Flag
    # Simple regime: 1 if price > SMA_200, else 0
    features["regime_flag"] = (df["Close"] > features["sma_200"]).astype(float)

    # 17. Liquidity Score
    # Simple liquidity proxy: normalized volume
    volume_percentile = df["Volume"].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    features["liq_score"] = volume_percentile

    # =========================================================================
    # VALIDATION & ENFORCEMENT
    # =========================================================================

    # Replace inf with NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    # Fill NaNs with 0.0 (institutional default for robustness)
    features = features.fillna(0.0)

    # Check all required features are present
    computed_features = set(features.columns)
    required_set = set(required_features)
    missing = list(required_set - computed_features)

    if missing:
        raise FeatureValidationError(
            f"Missing required features per contract '{contract_name}': {missing}"
        )

    # Enforce exact column order per contract
    features = features[required_features]

    # Enforce dtype float32
    features = features.astype(np.float32)

    # Validate compliance
    validation_result = validate_contract_compliance(
        list(features.columns),
        contract_name=contract_name,
        strict_order=True
    )

    if not validation_result["compliant"]:
        raise FeatureValidationError(
            f"Feature contract validation failed:\n"
            f"  Missing: {validation_result['missing']}\n"
            f"  Extra: {validation_result['extra']}\n"
            f"  Order mismatch: {validation_result['order_mismatch']}"
        )

    logger.debug(
        f"Computed {len(features.columns)} features matching contract '{contract_name}', "
        f"{len(features)} rows, dtype={features.dtypes.unique()}"
    )

    return features


# Legacy FeatureEngineer class for backwards compatibility
class FeatureEngineer:
    """
    DEPRECATED: Use compute_features_for_symbol() instead.

    Kept for backwards compatibility with existing code.
    """

    def __init__(self, use_technical: bool = True, use_lags: bool = True):
        logger.warning(
            "FeatureEngineer class is deprecated. Use compute_features_for_symbol() instead."
        )
        self.use_technical = use_technical
        self.use_lags = use_lags

    def compute_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Legacy method - delegates to compute_features_for_symbol."""
        try:
            return compute_features_for_symbol(prices, contract_name="ml_v1")
        except FeatureValidationError as e:
            logger.error(f"Feature computation failed: {e}")
            return pd.DataFrame()

    def compute_target(self, prices: pd.DataFrame, forward_window: int = 1) -> pd.Series:
        """
        Compute forward returns as target variable.
        """
        target = prices["Close"].pct_change(periods=forward_window, fill_method=None).shift(-forward_window)
        target = target.replace([np.inf, -np.inf], np.nan).dropna()
        return target
