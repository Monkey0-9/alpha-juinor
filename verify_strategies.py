#!/usr/bin/env python3
"""
MULTI-STRATEGY VERIFICATION
===========================

Verifies the correlation between Mean Reversion and Trend Following strategies.
Target: Correlation < 0.3
"""

import pandas as pd
import numpy as np
import yfinance as yf
from strategy_factory.manager import StrategyManager

def verify():
    # 1. Load Data
    data = yf.download("SPY", start="2005-01-01", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']['SPY']
    else:
        prices = data['Close']

    prices = prices.dropna()
    returns = prices.pct_change()

    # 2. Generate Signals
    mgr = StrategyManager()
    signal_df = mgr.generate_historical_signals("SPY", prices)

    # 3. Calculate Strategy Returns
    aligned_returns = returns.reindex(signal_df.index)

    strat_returns = pd.DataFrame()
    for col in signal_df.columns:
        pos = signal_df[col].shift(1)
        ret = pos * aligned_returns
        strat_returns[col] = ret

    strat_returns = strat_returns.dropna()

    # 4. Correlation Analysis
    corr_matrix = strat_returns.corr()

    output = []
    output.append("-" * 40)
    output.append("CORRELATION MATRIX (Daily Returns)")
    output.append("-" * 40)
    output.append(str(corr_matrix))

    mr_name = "MeanRestoration_RSI"
    tf_name = "TrendFollowing_12M"

    grade = "ERROR"
    if mr_name in corr_matrix and tf_name in corr_matrix:
        corr = corr_matrix.loc[mr_name, tf_name]
        output.append(f"\nCorrelation between Strategies: {corr:.4f}")

        if corr < 0.3:
            output.append("SUCCESS: Correlation is below 0.3")
            grade = "PASS"
        else:
            output.append("FAILURE: Correlation is too high")
            grade = "FAIL"
    else:
        output.append("Error: Could not find expected strategy names")
        corr = 0

    # 5. Performance Quick Check
    output.append("\n" + "-" * 40)
    output.append("PERFORMANCE QUICK CHECK (Gross)")
    output.append("-" * 40)

    for col in strat_returns.columns:
        sr = strat_returns[col]
        sharpe = sr.mean() / sr.std() * np.sqrt(252)
        output.append(f"{col:<25} Sharpe: {sharpe:.2f}")

    print("\n".join(output))
    with open("verification_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    return 0 if grade == "PASS" else 1

if __name__ == "__main__":
    import sys
    sys.exit(verify())
