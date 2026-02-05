#!/usr/bin/env python3
"""
STANDALONE INSTITUTIONAL DUE DILIGENCE
======================================

Fetches data directly from yfinance and runs validation.
No dependencies on existing data infrastructure.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Constants
TRADING_DAYS = 252
TOTAL_FRICTION = 0.0007

# Universe for backtesting
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM",
    "TLT", "IEF", "LQD", "HYG", "AGG",
    "GLD", "SLV",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP",
]


def fetch_data():
    """Fetch data directly from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed")
        print("Run: pip install yfinance")
        return None

    print(f"Fetching data for {len(UNIVERSE)} symbols...")

    try:
        data = yf.download(
            UNIVERSE,
            start="2010-01-01",
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False,
            group_by='ticker'
        )
    except Exception as e:
        print(f"ERROR: Failed to download data: {e}")
        return None

    # Extract close prices
    closes = {}
    for sym in UNIVERSE:
        try:
            if sym in data.columns.get_level_values(0):
                close = data[sym]['Close']
                if len(close.dropna()) > 252:
                    closes[sym] = close.dropna()
        except Exception:
            continue

    if not closes:
        print("ERROR: No valid data downloaded")
        return None

    df = pd.DataFrame(closes)
    print(f"Loaded {len(df.columns)} symbols, {len(df)} days")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def calculate_metrics(returns):
    """Calculate performance metrics."""
    if returns.empty or returns.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "vol": 0, "win_rate": 0}

    n_years = len(returns) / TRADING_DAYS
    total_ret = (1 + returns).prod() - 1
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else 0

    cum = (1 + returns).cumprod()
    max_dd = ((cum / cum.cummax()) - 1).min()
    win_rate = (returns > 0).mean()

    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "max_dd": float(max_dd),
        "vol": float(ann_vol),
        "win_rate": float(win_rate)
    }


def run_backtest(prices, start_idx, end_idx):
    """Run simple momentum backtest."""
    period_prices = prices.iloc[start_idx:end_idx]
    returns = period_prices.pct_change().dropna()

    # Simple momentum signal
    signals = returns.rolling(63).mean().apply(np.sign)
    signals = signals / len(signals.columns)  # Equal weight

    # Portfolio returns with costs
    port_ret = (signals.shift(1) * returns).sum(axis=1)
    costs = signals.diff().abs().sum(axis=1) * TOTAL_FRICTION
    net_ret = port_ret - costs

    return calculate_metrics(net_ret), net_ret


def main():
    print("=" * 70)
    print("     INSTITUTIONAL DUE DILIGENCE VALIDATION")
    print("     " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    prices = fetch_data()
    if prices is None:
        return 1

    # Split: 70% IS, 30% OOS
    split = int(len(prices) * 0.7)

    print("\n" + "=" * 70)
    print("REPORT 1: IN-SAMPLE vs OUT-OF-SAMPLE BACKTEST")
    print("=" * 70)

    is_metrics, _ = run_backtest(prices, 0, split)
    oos_metrics, _ = run_backtest(prices, split, len(prices))

    sharpe_decay = 0
    if is_metrics["sharpe"] > 0:
        sharpe_decay = (is_metrics["sharpe"] - oos_metrics["sharpe"]) / is_metrics["sharpe"]

    print(f"\n{'Metric':<20} {'In-Sample':>12} {'Out-Sample':>12}")
    print("-" * 44)
    print(f"{'CAGR':<20} {is_metrics['cagr']:>11.1%} {oos_metrics['cagr']:>11.1%}")
    print(f"{'Sharpe':<20} {is_metrics['sharpe']:>12.2f} {oos_metrics['sharpe']:>12.2f}")
    print(f"{'Max Drawdown':<20} {is_metrics['max_dd']:>11.1%} {oos_metrics['max_dd']:>11.1%}")
    print(f"{'Volatility':<20} {is_metrics['vol']:>11.1%} {oos_metrics['vol']:>11.1%}")
    print(f"{'Win Rate':<20} {is_metrics['win_rate']:>11.1%} {oos_metrics['win_rate']:>11.1%}")
    print("-" * 44)
    print(f"{'Sharpe Decay':<20} {sharpe_decay:>23.0%}")

    # Red flags
    red_flags = []
    if sharpe_decay > 0.5:
        red_flags.append("CRITICAL: High Sharpe decay suggests overfitting")
    if oos_metrics["max_dd"] < -0.30:
        red_flags.append("WARNING: OOS Max DD exceeds -30%")
    if oos_metrics["sharpe"] < 1.0:
        red_flags.append("CAUTION: OOS Sharpe below 1.0")

    if red_flags:
        print("\nRED FLAGS:")
        for f in red_flags:
            print(f"  ! {f}")

    # Scoring
    score = 0
    oos = oos_metrics
    if oos["sharpe"] >= 2.0:
        score += 30
    elif oos["sharpe"] >= 1.5:
        score += 20
    elif oos["sharpe"] >= 1.0:
        score += 10

    if oos["max_dd"] > -0.15:
        score += 20
    elif oos["max_dd"] > -0.25:
        score += 10

    if sharpe_decay < 0.3:
        score += 20
    elif sharpe_decay < 0.5:
        score += 10

    if oos["max_dd"] > -0.50:
        score += 20

    score += 10  # Capacity points (assumed ok for small scale)

    if score >= 80:
        grade = "A"
        verdict = "INSTITUTIONAL GRADE"
    elif score >= 60:
        grade = "B"
        verdict = "ACCEPTABLE WITH CAUTION"
    elif score >= 40:
        grade = "C"
        verdict = "NEEDS IMPROVEMENT"
    else:
        grade = "D"
        verdict = "NOT READY"

    print("\n" + "=" * 70)
    print("                  FINAL ASSESSMENT")
    print("=" * 70)
    print(f"""
  SCORE: {score}/100
  GRADE: {grade}

  VERDICT: {verdict}

  KEY METRICS (Out-of-Sample):
  - Sharpe Ratio:      {oos['sharpe']:.2f}
  - CAGR:              {oos['cagr']:.1%}
  - Max Drawdown:      {oos['max_dd']:.1%}
  - Sharpe Decay:      {sharpe_decay:.0%}

  HONEST ASSESSMENT:
""")
    if score >= 70:
        print("  - Ready for institutional capital")
    else:
        print("  - NOT ready for institutional capital")

    if oos['sharpe'] > 1.5:
        print("  - Evidence of alpha beyond luck")
    else:
        print("  - Alpha may be statistical noise")

    if sharpe_decay < 0.3:
        print("  - Low overfitting risk")
    else:
        print("  - Potential overfitting detected")

    print("=" * 70)

    # Save report
    report = {
        "generated": datetime.now().isoformat(),
        "score": score,
        "grade": grade,
        "in_sample": is_metrics,
        "out_of_sample": oos_metrics,
        "sharpe_decay": sharpe_decay,
        "red_flags": red_flags
    }

    os.makedirs("output", exist_ok=True)
    fn = f"output/due_diligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {fn}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
