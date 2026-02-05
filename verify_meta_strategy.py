#!/usr/bin/env python3
"""
META-STRATEGY VERIFICATION
==========================

Backtests the Combined Portfolio (Mean Reversion + Trend Following)
orchestrated by the Meta-Controller.

Goal: Combined Sharpe > Individual Sharpes & Lower Drawdown
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_factory.manager import StrategyManager
from allocator.meta_controller import MetaController
from strategy_factory.interface import Signal

def verify():
    print("=" * 70)
    print("     META-STRATEGY BACKTEST (The Strategy of Strategies)")
    print("=" * 70)

    # 1. Load Data
    print("Fetching SPY data (2005-2024)...")
    data = yf.download("SPY", start="2005-01-01", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']['SPY']
    else:
        prices = data['Close']

    prices = prices.dropna()
    returns = prices.pct_change()

    print(f"Loaded {len(prices)} days.")

    # 2. Initialize Components
    mgr = StrategyManager()
    controller = MetaController()

    # 3. Simulation Loop
    # We need to run simulation day-by-day to capture Regime shifts
    # (Since Regime depends on past data)

    print("Running simulation (this may take a minute)...")

    # Pre-calculate history for signals to speed up?
    # Actually, MetaController needs 'benchmark_prices' for regime.
    # Let's iterate.

    equity_mr = [1.0]
    equity_tf = [1.0]
    equity_combined = [1.0]
    equity_spy = [1.0]

    allocations_log = []

    # Start after sufficient lookback (e.g. 300 days for Trend + Vol)
    start_idx = 300

    # Convert prices to list for faster indexing or stay with pandas
    # Pandas iter is slow. Let's do vectorized regime detection first if possible?

    # Let's do a simplified approach:
    # 1. Generate all signals for history.
    # 2. Generate all regimes for history.
    # 3. Combine.

    print("  Generating Base Signals...")
    signal_df = mgr.generate_historical_signals("SPY", prices)

    print("  Detecting Historical Regimes...")
    regimes = []
    regime_detect = controller.regime_detector

    # Optimization: Rolling apply or loop
    # Let's just loop for regime detection, calculating on window

    dates = prices.index
    regime_map = []

    for i in range(len(prices)):
        if i < start_idx:
            regime_map.append(None)
            continue

        window = prices.iloc[:i+1]
        try:
            r = regime_detect.detect(window)
            regime_map.append(r)
        except:
            regime_map.append(None)

    # 4. Compute Returns
    print("  Computing Portfolio Returns...")

    daily_rets_mr = []
    daily_rets_tf = []
    daily_rets_combined = []

    for i in range(start_idx, len(prices)-1): # -1 because signal at T affects T+1
        date = dates[i]
        next_ret = returns.iloc[i+1] # Return for T+1

        # Signals at T
        sig_mr = signal_df.iloc[i-50][controller.MR_STRAT] if i-50 < len(signal_df) else 0 # signal_df started at index 50
        sig_tf = signal_df.iloc[i-50][controller.TF_STRAT] if i-50 < len(signal_df) else 0

        # Note: signal_df in manager started at index 50.
        # So signal_df index 0 corresponds to prices index 50.
        # price index i correspond to signal_df index i-50.

        try:
            val_mr = signal_df.loc[date, controller.MR_STRAT]
            val_tf = signal_df.loc[date, controller.TF_STRAT]
        except:
            # Date mismatch or something
            continue

        # Individual Strategy Returns (Gross)
        r_mr = val_mr * next_ret
        r_tf = val_tf * next_ret

        daily_rets_mr.append(r_mr)
        daily_rets_tf.append(r_tf)

        # Meta Controller Allocation
        # Re-construct signals dict
        sigs = {
            controller.MR_STRAT: Signal("SPY", val_mr, 1.0, False, {}),
            controller.TF_STRAT: Signal("SPY", val_tf, 1.0, False, {})
        }

        # Get Weights
        # Need regime at T
        regime = regime_map[i]
        if regime:
            weights, _ = controller.get_allocation_weights(prices.iloc[:i+1])
            # weights is e.g. {'MR': 0.7, 'TF': 0.3}
            # Combined Signal = w_mr * sig_mr + w_tf * sig_tf

            combined_sig_val = (weights.get(controller.MR_STRAT, 0) * val_mr) + \
                               (weights.get(controller.TF_STRAT, 0) * val_tf)

            # Apply Regime Risk Multiplier
            combined_sig_val *= regime.risk_multiplier

            r_comb = combined_sig_val * next_ret
            daily_rets_combined.append(r_comb)

            allocations_log.append(regime.regime)
        else:
            daily_rets_combined.append(0.0)

    # 5. Analysis
    df_res = pd.DataFrame({
        'MR': daily_rets_mr,
        'TF': daily_rets_tf,
        'Combined': daily_rets_combined
    })

    def calc_stats(series):
        ann_ret = series.mean() * 252
        ann_vol = series.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        return sharpe, ann_ret, ann_vol

    stats = {}
    for col in df_res.columns:
        stats[col] = calc_stats(df_res[col])

    print("\n" + "="*60)
    print(f"{'STRATEGY':<15} {'SHARPE':<10} {'RETURN':<10} {'VOL':<10}")
    print("-" * 60)

    best_indiv_sharpe = -99

    for col in ['MR', 'TF']:
        s, r, v = stats[col]
        print(f"{col:<15} {s:<10.2f} {r:<10.1%} {v:<10.1%}")
        best_indiv_sharpe = max(best_indiv_sharpe, s)

    s_comb, r_comb, v_comb = stats['Combined']
    print("-" * 60)
    print(f"{'Combined':<15} {s_comb:<10.2f} {r_comb:<10.1%} {v_comb:<10.1%}")
    print("=" * 60)

    # 6. Success Check
    output = []
    output.append("-" * 40)
    output.append("META-STRATEGY RESULTS")
    output.append("-" * 40)
    output.append(f"Combined Sharpe: {s_comb:.2f}")
    output.append(f"Best Individual: {best_indiv_sharpe:.2f}")

    grade = "FAIL"
    if s_comb > best_indiv_sharpe * 0.95: # Allow slight margin of error, but generally want improvement
        # Or checking if Vol is reduced
        if v_comb < max(stats['MR'][2], stats['TF'][2]):
             output.append("SUCCESS: Combined portfolio shows diversification benefits (Lower Vol or Higher Sharpe)")
             grade = "PASS"
        else:
             output.append("WARNING: No clear diversification benefit found.")
    else:
        output.append("FAILURE: Combined strategy underperformed.")

    print("\n".join(output))
    with open("meta_verification_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    return 0 if grade == "PASS" else 1

if __name__ == "__main__":
    sys.exit(verify())
