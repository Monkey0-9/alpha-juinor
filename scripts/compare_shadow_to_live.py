"""
scripts/compare_shadow_to_live.py
P1-4: Compare shadow predictions to live trading for drift analysis
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_shadow_predictions(db_path: str, days: int = 7):
    """Load shadow predictions from audit table."""
    conn = sqlite3.connect(db_path)

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    query = """
        SELECT timestamp, symbol, model_version, prediction, confidence, features_hash
        FROM shadow_predictions
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
    """

    df = pd.read_sql_query(query, conn, params=(cutoff,))
    conn.close()

    return df


def load_live_decisions(db_path: str, days: int = 7):
    """Load live trading decisions."""
    conn = sqlite3.connect(db_path)

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    query = """
        SELECT timestamp, symbol, final_decision, mu_hat, conviction
        FROM decisions
        WHERE timestamp >= ? AND final_decision != 'HOLD'
        ORDER BY timestamp DESC
    """

    df = pd.read_sql_query(query, conn, params=(cutoff,))
    conn.close()

    return df


def compute_correlation(shadow_df: pd.DataFrame, live_df: pd.DataFrame):
    """
    Compute correlation between shadow predictions and live decisions.

    Returns dict with:
        - correlation: Pearson correlation coefficient
        - agreement_rate: % of times sign matches
        - mean_abs_diff: Average absolute difference
    """
    # Merge on symbol and closest timestamp
    shadow_df['timestamp'] = pd.to_datetime(shadow_df['timestamp'])
    live_df['timestamp'] = pd.to_datetime(live_df['timestamp'])

    # Group by symbol
    results_by_symbol = {}

    for symbol in shadow_df['symbol'].unique():
        shadow_sym = shadow_df[shadow_df['symbol'] == symbol].copy()
        live_sym = live_df[live_df['symbol'] == symbol].copy()

        if len(shadow_sym) < 2 or len(live_sym) < 2:
            continue

        # Merge asof (nearest timestamp)
        merged = pd.merge_asof(
            shadow_sym.sort_values('timestamp'),
            live_sym.sort_values('timestamp'),
            on='timestamp',
            by='symbol',
            tolerance=pd.Timedelta('1hour'),
            direction='nearest'
        )

        if len(merged) < 2:
            continue

        # Compute metrics
        # Convert decision to signal: BUY=1, SELL=-1
        merged['live_signal'] = merged['final_decision'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0}).fillna(0)

        corr = merged['prediction'].corr(merged['live_signal'])
        agreement = (np.sign(merged['prediction']) == np.sign(merged['live_signal'])).mean()
        mean_diff = (merged['prediction'] - merged['live_signal']).abs().mean()

        results_by_symbol[symbol] = {
            'correlation': corr if not np.isnan(corr) else 0.0,
            'agreement_rate': agreement,
            'mean_abs_diff': mean_diff,
            'n_samples': len(merged)
        }

    return results_by_symbol


def detect_drift(shadow_df: pd.DataFrame, window_days: int = 3):
    """
    Detect model drift by comparing recent vs historical predictions.

    Returns dict with drift metrics.
    """
    shadow_df['timestamp'] = pd.to_datetime(shadow_df['timestamp'])

    recent_cutoff = datetime.utcnow() - timedelta(days=window_days)

    recent = shadow_df[shadow_df['timestamp'] >= recent_cutoff]
    historical = shadow_df[shadow_df['timestamp'] < recent_cutoff]

    if len(recent) < 10 or len(historical) < 10:
        return {"drift_detected": False, "reason": "insufficient_data"}

    # Compare distributions
    recent_mean = recent['prediction'].mean()
    hist_mean = historical['prediction'].mean()
    recent_std = recent['prediction'].std()
    hist_std = historical['prediction'].std()

    mean_shift = abs(recent_mean - hist_mean) / (hist_std + 1e-6)
    std_change = abs(recent_std - hist_std) / (hist_std + 1e-6)

    drift_detected = mean_shift > 0.5 or std_change > 0.3

    return {
        "drift_detected": drift_detected,
        "mean_shift": mean_shift,
        "std_change": std_change,
        "recent_mean": recent_mean,
        "historical_mean": hist_mean,
        "recent_std": recent_std,
        "historical_std": hist_std
    }


def main():
    """Generate shadow vs live comparison report."""
    db_path = "runtime/institutional_trading.db"

    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return

    logger.info("Loading shadow predictions...")
    shadow_df = load_shadow_predictions(db_path, days=7)

    logger.info("Loading live decisions...")
    live_df = load_live_decisions(db_path, days=7)

    logger.info(f"Shadow predictions: {len(shadow_df)}")
    logger.info(f"Live decisions: {len(live_df)}")

    if len(shadow_df) == 0:
        logger.warning("No shadow predictions found. Is ml_mode='shadow' enabled?")
        return

    # Correlation analysis
    logger.info("\n" + "="*60)
    logger.info("CORRELATION ANALYSIS")
    logger.info("="*60)

    correlation_results = compute_correlation(shadow_df, live_df)

    for symbol, metrics in correlation_results.items():
        logger.info(
            f"{symbol}: corr={metrics['correlation']:.3f}, "
            f"agreement={metrics['agreement_rate']:.1%}, "
            f"diff={metrics['mean_abs_diff']:.3f}, "
            f"n={metrics['n_samples']}"
        )

    # Drift analysis
    logger.info("\n" + "="*60)
    logger.info("DRIFT ANALYSIS")
    logger.info("="*60)

    drift_results = detect_drift(shadow_df, window_days=3)

    if drift_results["drift_detected"]:
        logger.warning("⚠️  MODEL DRIFT DETECTED!")
        logger.warning(f"   Mean shift: {drift_results['mean_shift']:.3f} σ")
        logger.warning(f"   Std change: {drift_results['std_change']:.1%}")
    else:
        logger.info("✓ No significant drift detected")

    logger.info(f"  Recent mean: {drift_results.get('recent_mean', 0):.3f}")
    logger.info(f"  Historical mean: {drift_results.get('historical_mean', 0):.3f}")

    logger.info("\n" + "="*60)
    logger.info("Report complete. Review logs for details.")
    logger.info("="*60)


if __name__ == "__main__":
    main()
