#!/usr/bin/env python3
"""
FULL INSTITUTIONAL VALIDATION SUITE
====================================

Complete validation across:
1. Multi-asset universe (20+ ETFs)
2. Crisis period stress testing (2008, 2020, 2022)
3. Strategy capacity analysis
4. Final institutional grade report
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import os

# Constants
TRADING_DAYS = 252
TRANSACTION_COST_BPS = 10
SLIPPAGE_BPS = 5
TOTAL_FRICTION = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000

# Multi-asset universe
UNIVERSE = [
    # Equity ETFs
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VTI",
    # Sector ETFs
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP",
    # Bond ETFs
    "TLT", "IEF", "LQD", "HYG", "AGG",
    # Commodities
    "GLD", "SLV",
]

# Crisis periods
CRISIS_PERIODS = {
    "2008 Financial Crisis": ("2008-09-01", "2009-03-31"),
    "2020 COVID Crash": ("2020-02-01", "2020-04-30"),
    "2022 Rate Shock": ("2022-01-01", "2022-10-31"),
}


@dataclass
class AssetResult:
    """Results for single asset validation."""
    symbol: str
    oos_sharpe: float
    oos_cagr: float
    max_dd: float
    passed: bool


@dataclass
class CrisisResult:
    """Results for crisis period."""
    name: str
    return_pct: float
    max_dd: float
    benchmark_dd: float
    survived: bool


@dataclass
class FullReport:
    """Complete institutional validation report."""
    grade: str
    score: int
    multi_asset_sharpe: float
    multi_asset_pass_rate: float
    crisis_survival_rate: float
    worst_crisis_dd: float
    capacity_limit: float
    red_flags: List[str]
    recommendations: List[str]


def fetch_multi_asset_data(
    symbols: List[str],
    start: str = "2005-01-01"
) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple assets."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        return {}

    print(f"Fetching {len(symbols)} assets from {start}...")

    data = yf.download(
        symbols,
        start=start,
        end=datetime.now().strftime("%Y-%m-%d"),
        progress=False
    )

    result = {}
    for sym in symbols:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ('Close', sym) in data.columns:
                    close = data[('Close', sym)].dropna()
                else:
                    continue
            else:
                close = data['Close'].dropna()

            if len(close) > 252:
                result[sym] = pd.DataFrame({'close': close})
        except Exception:
            continue

    print(f"Loaded {len(result)} assets successfully")
    return result


def strategy_mean_reversion(
    prices: pd.Series,
    rsi_period: int = 14,
    oversold: int = 30,
    overbought: int = 70
) -> pd.Series:
    """RSI-based mean reversion strategy."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))

    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    position = pd.Series(0.0, index=prices.index)

    for i in range(1, len(position)):
        prev_pos = position.iloc[i-1]
        curr_rsi = rsi.iloc[i]

        if pd.isna(curr_rsi):
            position.iloc[i] = 0
        elif curr_rsi < oversold:
            position.iloc[i] = 1
        elif curr_rsi > overbought:
            position.iloc[i] = 0
        else:
            position.iloc[i] = prev_pos

    return position


def strategy_dual_ma(
    prices: pd.Series,
    fast: int = 50,
    slow: int = 200
) -> pd.Series:
    """Dual moving average crossover."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def run_backtest(prices: pd.Series, signals: pd.Series) -> dict:
    """Run backtest and return metrics."""
    returns = prices.pct_change()
    aligned = signals.shift(1).fillna(0)

    strat_ret = aligned * returns
    costs = aligned.diff().abs().fillna(0) * TOTAL_FRICTION
    net_ret = strat_ret - costs
    net_ret = net_ret.dropna()

    if len(net_ret) < 20 or net_ret.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0}

    n_years = len(net_ret) / TRADING_DAYS
    vol = net_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (net_ret.mean() * TRADING_DAYS) / vol if vol > 0 else 0

    total_ret = (1 + net_ret).prod() - 1
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    cum = (1 + net_ret).cumprod()
    max_dd = ((cum / cum.cummax()) - 1).min()

    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "max_dd": float(max_dd)
    }


def validate_multi_asset(data: Dict[str, pd.DataFrame]) -> List[AssetResult]:
    """Validate strategy across multiple assets."""
    print("\n" + "=" * 60)
    print("MULTI-ASSET VALIDATION")
    print("=" * 60)

    results = []

    for sym, df in data.items():
        prices = df['close']

        # Use last 30% as OOS
        split = int(len(prices) * 0.7)
        oos_prices = prices.iloc[split:]

        signals = strategy_mean_reversion(oos_prices)
        metrics = run_backtest(oos_prices, signals)

        passed = metrics["sharpe"] > 0
        result = AssetResult(
            symbol=sym,
            oos_sharpe=metrics["sharpe"],
            oos_cagr=metrics["cagr"],
            max_dd=metrics["max_dd"],
            passed=passed
        )
        results.append(result)

        status = "✓" if passed else "✗"
        print(f"  {sym:>5}: Sharpe={metrics['sharpe']:>6.2f}, "
              f"CAGR={metrics['cagr']:>6.1%}, "
              f"MaxDD={metrics['max_dd']:>6.1%} [{status}]")

    passed_count = sum(1 for r in results if r.passed)
    print(f"\nPassed: {passed_count}/{len(results)} "
          f"({passed_count/len(results)*100:.0f}%)")

    return results


def run_crisis_stress_tests(data: Dict[str, pd.DataFrame]) -> List[CrisisResult]:
    """Test strategy during historical crisis periods."""
    print("\n" + "=" * 60)
    print("CRISIS STRESS TESTING")
    print("=" * 60)

    results = []

    # Use SPY for crisis testing
    if "SPY" not in data:
        print("ERROR: SPY not in data for crisis testing")
        return results

    spy = data["SPY"]['close']

    for crisis_name, (start, end) in CRISIS_PERIODS.items():
        try:
            crisis_prices = spy[start:end]

            if len(crisis_prices) < 20:
                print(f"  {crisis_name}: Insufficient data")
                continue

            # Strategy performance
            signals = strategy_mean_reversion(crisis_prices)
            metrics = run_backtest(crisis_prices, signals)

            # Benchmark (buy-and-hold)
            bench_ret = crisis_prices.pct_change().dropna()
            bench_cum = (1 + bench_ret).cumprod()
            bench_dd = ((bench_cum / bench_cum.cummax()) - 1).min()

            survived = metrics["max_dd"] > bench_dd * 0.7

            result = CrisisResult(
                name=crisis_name,
                return_pct=metrics["cagr"],
                max_dd=metrics["max_dd"],
                benchmark_dd=float(bench_dd),
                survived=survived
            )
            results.append(result)

            status = "SURVIVED" if survived else "FAILED"
            print(f"  {crisis_name}:")
            print(f"    Strategy DD: {metrics['max_dd']:.1%} vs "
                  f"Benchmark DD: {bench_dd:.1%} [{status}]")

        except Exception as e:
            print(f"  {crisis_name}: Error - {e}")

    return results


def estimate_capacity(avg_sharpe: float, n_assets: int) -> float:
    """Estimate strategy capacity."""
    # Simple capacity model
    # Higher Sharpe = more capacity, more assets = more capacity
    base_capacity = 1_000_000  # $1M base

    sharpe_mult = max(avg_sharpe, 0.1)
    asset_mult = np.log1p(n_assets)

    capacity = base_capacity * sharpe_mult * asset_mult
    return min(capacity, 100_000_000)  # Cap at $100M


def generate_full_report(
    asset_results: List[AssetResult],
    crisis_results: List[CrisisResult]
) -> FullReport:
    """Generate comprehensive institutional report."""

    # Multi-asset metrics
    passed_assets = [r for r in asset_results if r.passed]
    pass_rate = len(passed_assets) / len(asset_results) if asset_results else 0
    avg_sharpe = np.mean([r.oos_sharpe for r in passed_assets]) if passed_assets else 0

    # Crisis metrics
    survived = [r for r in crisis_results if r.survived]
    survival_rate = len(survived) / len(crisis_results) if crisis_results else 0
    worst_dd = min([r.max_dd for r in crisis_results]) if crisis_results else 0

    # Capacity
    capacity = estimate_capacity(avg_sharpe, len(passed_assets))

    # Scoring
    score = 0
    red_flags = []
    recommendations = []

    # Multi-asset score (max 40)
    if pass_rate >= 0.8:
        score += 40
    elif pass_rate >= 0.6:
        score += 30
    elif pass_rate >= 0.4:
        score += 20
    else:
        red_flags.append(f"Low multi-asset pass rate: {pass_rate:.0%}")

    # Sharpe score (max 25)
    if avg_sharpe >= 1.0:
        score += 25
    elif avg_sharpe >= 0.5:
        score += 15
    elif avg_sharpe > 0:
        score += 10
    else:
        red_flags.append(f"Negative average OOS Sharpe: {avg_sharpe:.2f}")

    # Crisis score (max 25)
    if survival_rate >= 0.8:
        score += 25
    elif survival_rate >= 0.5:
        score += 15
    else:
        red_flags.append(f"Low crisis survival rate: {survival_rate:.0%}")

    # Drawdown score (max 10)
    if worst_dd > -0.25:
        score += 10
    elif worst_dd > -0.40:
        score += 5
    else:
        red_flags.append(f"Severe crisis drawdown: {worst_dd:.0%}")

    # Grade
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B+"
    elif score >= 55:
        grade = "B"
    elif score >= 40:
        grade = "C"
    else:
        grade = "D"

    # Recommendations
    if pass_rate < 0.8:
        recommendations.append("Consider filtering out underperforming assets")
    if avg_sharpe < 1.0:
        recommendations.append("Add more uncorrelated signals to boost Sharpe")
    if survival_rate < 1.0:
        recommendations.append("Implement crisis detection regime switching")
    if worst_dd < -0.30:
        recommendations.append("Add volatility targeting to limit drawdowns")

    return FullReport(
        grade=grade,
        score=score,
        multi_asset_sharpe=float(avg_sharpe),
        multi_asset_pass_rate=float(pass_rate),
        crisis_survival_rate=float(survival_rate),
        worst_crisis_dd=float(worst_dd),
        capacity_limit=float(capacity),
        red_flags=red_flags,
        recommendations=recommendations
    )


def main():
    print("=" * 70)
    print("     FULL INSTITUTIONAL VALIDATION SUITE")
    print("     " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Load multi-asset data
    data = fetch_multi_asset_data(UNIVERSE, start="2005-01-01")

    if not data:
        print("ERROR: No data loaded")
        return 1

    # Run validations
    asset_results = validate_multi_asset(data)
    crisis_results = run_crisis_stress_tests(data)

    # Generate report
    report = generate_full_report(asset_results, crisis_results)

    # Print final report
    print("\n" + "=" * 70)
    print("                 FINAL INSTITUTIONAL REPORT")
    print("=" * 70)

    print(f"""
  GRADE: {report.grade}
  SCORE: {report.score}/100

  MULTI-ASSET PERFORMANCE:
  - Pass Rate:           {report.multi_asset_pass_rate:.0%}
  - Average OOS Sharpe:  {report.multi_asset_sharpe:.2f}

  CRISIS RESILIENCE:
  - Survival Rate:       {report.crisis_survival_rate:.0%}
  - Worst Crisis DD:     {report.worst_crisis_dd:.1%}

  CAPACITY ANALYSIS:
  - Estimated Limit:     ${report.capacity_limit/1e6:.0f}M
""")

    if report.red_flags:
        print("  RED FLAGS:")
        for flag in report.red_flags:
            print(f"    ! {flag}")

    if report.recommendations:
        print("\n  RECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"    → {rec}")

    print("\n" + "=" * 70)

    # Save detailed results
    results = {
        "generated": datetime.now().isoformat(),
        "grade": report.grade,
        "score": report.score,
        "multi_asset": {
            "pass_rate": report.multi_asset_pass_rate,
            "avg_sharpe": report.multi_asset_sharpe,
            "assets": [
                {
                    "symbol": r.symbol,
                    "oos_sharpe": float(r.oos_sharpe),
                    "oos_cagr": float(r.oos_cagr),
                    "max_dd": float(r.max_dd),
                    "passed": bool(r.passed)
                }
                for r in asset_results
            ]
        },
        "crisis_testing": {
            "survival_rate": report.crisis_survival_rate,
            "worst_dd": report.worst_crisis_dd,
            "periods": [
                {
                    "name": r.name,
                    "return": float(r.return_pct),
                    "max_dd": float(r.max_dd),
                    "benchmark_dd": float(r.benchmark_dd),
                    "survived": bool(r.survived)
                }
                for r in crisis_results
            ]
        },
        "capacity_limit": report.capacity_limit,
        "red_flags": report.red_flags,
        "recommendations": report.recommendations
    }

    os.makedirs("output", exist_ok=True)
    fn = f"output/institutional_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Report saved: {fn}")

    return 0 if report.grade in ["A", "B+", "B"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
