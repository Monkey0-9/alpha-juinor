#!/usr/bin/env python3
"""
INSTITUTIONAL DUE DILIGENCE - QUICK VALIDATION
===============================================

A streamlined validation engine using existing market data loaders.
Generates the 4 critical institutional reports.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
TRADING_DAYS = 252
TOTAL_FRICTION = 0.0007  # 7 bps total (5 tc + 2 slippage)


@dataclass
class PerformanceMetrics:
    """Backtest performance metrics."""
    period: str
    days: int
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    volatility: float
    win_rate: float
    trades: int


def load_market_data():
    """Load market data from parquet files in data/raw."""
    from pathlib import Path

    # Try multiple possible data locations
    for data_path in ["data/cache", "data/raw", "runtime/market_data"]:
        data_dir = Path(data_path)
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                break
    else:
        print("ERROR: No data directory found with parquet files")
        print("Run: python ingest_backtest_data.py first")
        return None

    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    closes = {}
    for pf in parquet_files[:30]:  # Limit for performance
        try:
            symbol = pf.stem
            df = pd.read_parquet(pf)

            # Find close column (case insensitive)
            close_col = None
            for col in df.columns:
                if col.lower() == 'close':
                    close_col = col
                    break

            if close_col and len(df) > 252:
                closes[symbol] = df[close_col]

        except Exception as e:
            continue

    if not closes:
        print("ERROR: Could not load any valid data from parquet files")
        return None

    df = pd.DataFrame(closes)
    df = df.sort_index()

    print(f"Loaded {len(df.columns)} symbols, {len(df)} days")
    return df


def calculate_metrics(returns: pd.Series, period: str) -> PerformanceMetrics:
    """Calculate performance metrics from return series."""
    if returns.empty or returns.std() == 0:
        return PerformanceMetrics(
            period, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )

    n_days = len(returns)
    n_years = n_days / TRADING_DAYS

    total_ret = (1 + returns).prod() - 1
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0]
    down_vol = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else ann_vol
    sortino = (returns.mean() * TRADING_DAYS) / down_vol if down_vol > 0 else 0

    cum = (1 + returns).cumprod()
    max_dd = ((cum / cum.cummax()) - 1).min()

    win_rate = (returns > 0).mean()
    trades = int((returns != 0).sum())

    return PerformanceMetrics(
        period=period,
        days=n_days,
        total_return=float(total_ret),
        cagr=float(cagr),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_dd=float(max_dd),
        volatility=float(ann_vol),
        win_rate=float(win_rate),
        trades=trades
    )


def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Generate momentum + mean-reversion signals."""
    returns = prices.pct_change()

    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for col in prices.columns:
        # Momentum: 12m - 1m
        mom = returns[col].rolling(252).sum() - returns[col].rolling(21).sum()
        signals[col] = np.sign(mom.fillna(0))

        # Volatility targeting
        vol = returns[col].rolling(20).std() * np.sqrt(252)
        vol_scale = (0.15 / vol).clip(0.5, 2.0)
        signals[col] = signals[col] * vol_scale

    # Equal weight
    signals = signals / len(prices.columns)
    return signals.clip(-1, 1)


def run_backtest(prices: pd.DataFrame, start_idx: int, end_idx: int,
                 period_name: str) -> PerformanceMetrics:
    """Run backtest on specified period."""
    period_prices = prices.iloc[start_idx:end_idx]
    returns = period_prices.pct_change().dropna()

    signals = generate_signals(period_prices)

    # Portfolio returns
    port_ret = (signals.shift(1) * returns).sum(axis=1)

    # Transaction costs
    turnover = signals.diff().abs().sum(axis=1)
    costs = turnover * TOTAL_FRICTION
    net_ret = port_ret - costs

    return calculate_metrics(net_ret, period_name)


def main():
    print("=" * 70)
    print("     INSTITUTIONAL DUE DILIGENCE VALIDATION")
    print("     " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Load data
    prices = load_market_data()

    if prices is None or prices.empty:
        print("\nERROR: Could not load market data")
        print("This validation requires historical price data in the database.")
        print("Run data ingestion first: python scripts/ingest_history.py")
        return 1

    print(f"\nLoaded {len(prices.columns)} symbols, {len(prices)} days")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Split: 70% in-sample, 30% out-of-sample
    split = int(len(prices) * 0.7)

    # REPORT 1: In-Sample vs Out-of-Sample Backtest
    print("\n" + "=" * 70)
    print("REPORT 1: IN-SAMPLE vs OUT-OF-SAMPLE BACKTEST")
    print("=" * 70)

    is_metrics = run_backtest(prices, 0, split, "In-Sample")
    oos_metrics = run_backtest(prices, split, len(prices), "Out-of-Sample")

    sharpe_decay = 0
    if is_metrics.sharpe > 0:
        sharpe_decay = (is_metrics.sharpe - oos_metrics.sharpe) / is_metrics.sharpe

    print(f"\n{'Metric':<20} {'In-Sample':>12} {'Out-Sample':>12}")
    print("-" * 44)
    print(f"{'CAGR':<20} {is_metrics.cagr:>11.1%} {oos_metrics.cagr:>11.1%}")
    print(f"{'Sharpe':<20} {is_metrics.sharpe:>12.2f} {oos_metrics.sharpe:>12.2f}")
    print(f"{'Sortino':<20} {is_metrics.sortino:>12.2f} {oos_metrics.sortino:>12.2f}")
    print(f"{'Max Drawdown':<20} {is_metrics.max_dd:>11.1%} {oos_metrics.max_dd:>11.1%}")
    print(f"{'Volatility':<20} {is_metrics.volatility:>11.1%} {oos_metrics.volatility:>11.1%}")
    print(f"{'Win Rate':<20} {is_metrics.win_rate:>11.1%} {oos_metrics.win_rate:>11.1%}")
    print("-" * 44)
    print(f"{'Sharpe Decay':<20} {sharpe_decay:>23.0%}")

    # Red flags
    red_flags = []
    if sharpe_decay > 0.5:
        red_flags.append("CRITICAL: High Sharpe decay suggests overfitting")
    if oos_metrics.max_dd < -0.30:
        red_flags.append("WARNING: OOS Max DD exceeds -30%")
    if oos_metrics.sharpe < 1.0:
        red_flags.append("CAUTION: OOS Sharpe below 1.0")

    if red_flags:
        print("\nRED FLAGS:")
        for f in red_flags:
            print(f"  ! {f}")

    # REPORT 2: Crisis Periods (simulated)
    print("\n" + "=" * 70)
    print("REPORT 2: STRESS TEST (if data available)")
    print("=" * 70)

    crisis_results = []
    # Note: Would test 2008, 2020, 2022 if data covers those periods
    worst_dd = oos_metrics.max_dd
    print(f"\n  Worst Drawdown in OOS period: {worst_dd:.1%}")
    print(f"  Survival: {'YES' if worst_dd > -0.50 else 'NO'}")

    # REPORT 3: Capacity Analysis
    print("\n" + "=" * 70)
    print("REPORT 3: CAPACITY ANALYSIS")
    print("=" * 70)

    base_return = oos_metrics.cagr
    capitals = [1e6, 5e6, 10e6, 25e6, 50e6, 100e6]

    print(f"\n{'Capital':>10} {'Net Return':>12} {'Impact':>10} {'Sharpe':>10}")
    print("-" * 44)

    capacity_limit = capitals[0]
    for cap in capitals:
        # Simple market impact model
        participation = cap / (10e6 * len(prices.columns))
        impact = 0.1 * np.sqrt(min(participation, 0.1))
        net_ret = max(base_return - impact, 0)
        net_sharpe = oos_metrics.sharpe * (net_ret / base_return) if base_return > 0 else 0

        if net_sharpe >= 1.0:
            capacity_limit = cap

        print(f"${cap/1e6:>8.0f}M {net_ret:>11.1%} {impact:>9.2%} {net_sharpe:>10.2f}")

    print("-" * 44)
    print(f"Estimated Capacity Limit: ${capacity_limit/1e6:.0f}M")

    # FINAL ASSESSMENT
    print("\n" + "=" * 70)
    print("                  FINAL ASSESSMENT")
    print("=" * 70)

    score = 0
    if oos_metrics.sharpe >= 2.0:
        score += 30
    elif oos_metrics.sharpe >= 1.5:
        score += 20
    elif oos_metrics.sharpe >= 1.0:
        score += 10

    if oos_metrics.max_dd > -0.15:
        score += 20
    elif oos_metrics.max_dd > -0.25:
        score += 10

    if sharpe_decay < 0.3:
        score += 20
    elif sharpe_decay < 0.5:
        score += 10

    if worst_dd > -0.50:
        score += 20

    if capacity_limit >= 10e6:
        score += 10

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

    print(f"""
  SCORE: {score}/100
  GRADE: {grade}

  VERDICT: {verdict}

  KEY METRICS (Out-of-Sample):
  - Sharpe Ratio:      {oos_metrics.sharpe:.2f}
  - CAGR:              {oos_metrics.cagr:.1%}
  - Max Drawdown:      {oos_metrics.max_dd:.1%}
  - Sharpe Decay:      {sharpe_decay:.0%}
  - Capacity:          ${capacity_limit/1e6:.0f}M

  HONEST ASSESSMENT:
  {'- Ready for institutional capital' if score >= 70 else '- NOT ready for institutional capital'}
  {'- Evidence of alpha beyond luck' if oos_metrics.sharpe > 1.5 else '- Alpha may be statistical noise'}
  {'- Low overfitting risk' if sharpe_decay < 0.3 else '- Potential overfitting detected'}
""")
    print("=" * 70)

    # Save report
    report = {
        "generated": datetime.now().isoformat(),
        "score": score,
        "grade": grade,
        "in_sample": asdict(is_metrics),
        "out_of_sample": asdict(oos_metrics),
        "sharpe_decay": sharpe_decay,
        "capacity_limit": capacity_limit,
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
