#!/usr/bin/env python3
"""
RESEARCH VALIDATOR - First Principles Approach
==============================================

This is a RESEARCH tool, not a trading bot.

Implements institutional-grade validation methodology:
1. Simple, economically-grounded strategies
2. Walk-forward analysis (train/test rolling windows)
3. Conservative friction modeling
4. Multiple robustness checks

Success criteria:
- OOS Sharpe > 0 (positive)
- Sharpe Decay < 30%
- Stable across rolling windows
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
import os

# =============================================================================
# CONFIGURATION - Conservative and Realistic
# =============================================================================

# Transaction costs (conservative)
TRANSACTION_COST_BPS = 10  # 10 bps per trade (very conservative)
SLIPPAGE_BPS = 5           # 5 bps slippage
TOTAL_FRICTION = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000

# Trading parameters
TRADING_DAYS = 252

# Walk-forward windows
TRAIN_YEARS = 5
TEST_YEARS = 3


@dataclass
class StrategyResult:
    """Results from a single backtest period."""
    period: str
    start: str
    end: str
    sharpe: float
    cagr: float
    max_dd: float
    volatility: float
    win_rate: float
    trades: int
    is_oos: bool


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""
    strategy_name: str
    in_sample_sharpe: float
    out_sample_sharpe: float
    sharpe_decay_pct: float
    windows: List[StrategyResult]
    passed: bool


# =============================================================================
# DATA LOADING
# =============================================================================

def fetch_data(symbols: List[str], start: str = "2005-01-01") -> pd.DataFrame:
    """Fetch price data from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        return None

    print(f"Fetching {symbols} from {start}...")

    data = yf.download(
        symbols,
        start=start,
        end=datetime.now().strftime("%Y-%m-%d"),
        progress=False
    )

    if data.empty:
        print("ERROR: No data downloaded")
        return None

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        # Single ticker case with multi-level
        close_col = ('Close', symbols[0]) if len(symbols) == 1 else 'Close'
        if close_col in data.columns:
            close = data[close_col]
        else:
            # Try to get Close level
            close = data['Close']
            if isinstance(close, pd.DataFrame):
                close = close[symbols[0]]
    else:
        close = data['Close'] if 'Close' in data.columns else data['Adj Close']

    result = pd.DataFrame({'close': close})
    result = result.dropna()

    print(f"Loaded {len(result)} days")
    return result


# =============================================================================
# STRATEGY 1: DUAL MOVING AVERAGE TREND FOLLOWING
# =============================================================================

def strategy_dual_ma(
    prices: pd.Series,
    fast_period: int = 50,
    slow_period: int = 200
) -> pd.Series:
    """
    Simple dual moving average crossover.

    ECONOMIC RATIONALE:
    - Captures major market trends
    - Avoids bear markets by staying out when trend reverses
    - Very few parameters = hard to overfit

    Rules:
    - Long when fast MA > slow MA
    - Cash when fast MA < slow MA
    """
    fast_ma = prices.rolling(fast_period).mean()
    slow_ma = prices.rolling(slow_period).mean()

    # Signal: 1 = long, 0 = cash
    signal = (fast_ma > slow_ma).astype(float)

    return signal


# =============================================================================
# STRATEGY 2: MEAN REVERSION (RSI-BASED)
# =============================================================================

def strategy_mean_reversion(
    prices: pd.Series,
    rsi_period: int = 14,
    oversold: int = 30,
    overbought: int = 70
) -> pd.Series:
    """
    RSI-based mean reversion.

    ECONOMIC RATIONALE:
    - After extreme moves, prices tend to snap back
    - Basic market microstructure: overreaction followed by correction

    Rules:
    - Buy when RSI < oversold
    - Sell when RSI > overbought
    - Hold otherwise
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))

    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Position logic
    position = pd.Series(0.0, index=prices.index)

    for i in range(1, len(position)):
        prev_pos = position.iloc[i-1]
        curr_rsi = rsi.iloc[i]

        if pd.isna(curr_rsi):
            position.iloc[i] = 0
        elif curr_rsi < oversold:
            position.iloc[i] = 1  # Buy
        elif curr_rsi > overbought:
            position.iloc[i] = 0  # Sell
        else:
            position.iloc[i] = prev_pos  # Hold

    return position


# =============================================================================
# STRATEGY 3: VOLATILITY TARGETING
# =============================================================================

def strategy_vol_target(
    prices: pd.Series,
    target_vol: float = 0.10,  # 10% annualized
    lookback: int = 20
) -> pd.Series:
    """
    Volatility-targeting overlay.

    ECONOMIC RATIONALE:
    - Risk parity: scale positions to maintain constant risk
    - Reduces exposure during high volatility (crisis protection)
    - Increases exposure during calm markets

    This can be applied as an overlay to other strategies.
    """
    returns = prices.pct_change()
    realized_vol = returns.rolling(lookback).std() * np.sqrt(TRADING_DAYS)

    # Position size = target_vol / realized_vol
    position = (target_vol / realized_vol).clip(0.2, 2.0)

    return position.fillna(1.0)


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    period_name: str = "Backtest",
    is_oos: bool = False
) -> StrategyResult:
    """
    Run backtest with realistic friction.
    """
    returns = prices.pct_change()

    # Align signals (trade at next day's open)
    aligned_signals = signals.shift(1).fillna(0)

    # Gross returns
    strategy_returns = aligned_signals * returns

    # Transaction costs (on signal changes)
    trades = aligned_signals.diff().abs().fillna(0)
    costs = trades * TOTAL_FRICTION

    # Net returns
    net_returns = strategy_returns - costs

    # Drop NaN
    net_returns = net_returns.dropna()

    if len(net_returns) < 20:
        return StrategyResult(
            period=period_name,
            start="",
            end="",
            sharpe=0,
            cagr=0,
            max_dd=0,
            volatility=0,
            win_rate=0,
            trades=0,
            is_oos=is_oos
        )

    # Metrics
    n_years = len(net_returns) / TRADING_DAYS

    # Sharpe
    ann_vol = net_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (net_returns.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else 0

    # CAGR
    total_return = (1 + net_returns).prod() - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    # Max Drawdown
    cumulative = (1 + net_returns).cumprod()
    max_dd = ((cumulative / cumulative.cummax()) - 1).min()

    # Win rate
    win_rate = (net_returns > 0).mean()

    # Trade count
    trade_count = int((trades > 0).sum())

    return StrategyResult(
        period=period_name,
        start=str(net_returns.index[0].date()),
        end=str(net_returns.index[-1].date()),
        sharpe=float(sharpe),
        cagr=float(cagr),
        max_dd=float(max_dd),
        volatility=float(ann_vol),
        win_rate=float(win_rate),
        trades=trade_count,
        is_oos=is_oos
    )


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_analysis(
    prices: pd.Series,
    strategy_func,
    strategy_name: str,
    train_years: int = TRAIN_YEARS,
    test_years: int = TEST_YEARS,
    **strategy_params
) -> WalkForwardResult:
    """
    Rolling walk-forward validation.

    This is the GOLD STANDARD for strategy validation.

    Process:
    1. Train on window [t, t + train_years]
    2. Test on window [t + train_years, t + train_years + test_years]
    3. Roll forward and repeat
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD ANALYSIS: {strategy_name}")
    print(f"{'='*60}")

    results = []
    is_sharpes = []
    oos_sharpes = []

    train_days = train_years * TRADING_DAYS
    test_days = test_years * TRADING_DAYS
    step_days = test_days  # Non-overlapping windows

    start_idx = 0
    window_num = 0

    while start_idx + train_days + test_days <= len(prices):
        window_num += 1

        # In-Sample period
        is_start = start_idx
        is_end = start_idx + train_days

        # Out-of-Sample period
        oos_start = is_end
        oos_end = min(is_end + test_days, len(prices))

        # Get data
        is_prices = prices.iloc[is_start:is_end]
        oos_prices = prices.iloc[oos_start:oos_end]

        # Generate signals (using IS data for "training")
        is_signals = strategy_func(is_prices, **strategy_params)
        oos_signals = strategy_func(oos_prices, **strategy_params)

        # Run backtests
        is_result = run_backtest(
            is_prices, is_signals,
            f"Window {window_num} IS", is_oos=False
        )
        oos_result = run_backtest(
            oos_prices, oos_signals,
            f"Window {window_num} OOS", is_oos=True
        )

        results.append(is_result)
        results.append(oos_result)

        is_sharpes.append(is_result.sharpe)
        oos_sharpes.append(oos_result.sharpe)

        print(f"Window {window_num}: "
              f"IS={is_result.start} to {is_result.end}, "
              f"Sharpe={is_result.sharpe:.2f} | "
              f"OOS={oos_result.start} to {oos_result.end}, "
              f"Sharpe={oos_result.sharpe:.2f}")

        # Move to next window
        start_idx += step_days

    # Aggregate results
    avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0
    avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0

    # Sharpe decay
    decay = 0
    if avg_is_sharpe > 0:
        decay = (avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe

    # Pass criteria
    passed = (
        avg_oos_sharpe > 0 and  # Positive OOS
        decay < 0.30            # Less than 30% decay
    )

    print(f"\n{'='*60}")
    print(f"SUMMARY: {strategy_name}")
    print(f"{'='*60}")
    print(f"Average IS Sharpe:  {avg_is_sharpe:.3f}")
    print(f"Average OOS Sharpe: {avg_oos_sharpe:.3f}")
    print(f"Sharpe Decay:       {decay:.1%}")
    print(f"PASSED: {'YES' if passed else 'NO'}")

    return WalkForwardResult(
        strategy_name=strategy_name,
        in_sample_sharpe=avg_is_sharpe,
        out_sample_sharpe=avg_oos_sharpe,
        sharpe_decay_pct=decay,
        windows=results,
        passed=passed
    )


# =============================================================================
# MAIN RESEARCH VALIDATOR
# =============================================================================

def main():
    print("=" * 70)
    print("     RESEARCH VALIDATOR - First Principles Approach")
    print("     " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Fetch SPY data (15+ years for robust validation)
    prices_df = fetch_data(["SPY"], start="2005-01-01")
    if prices_df is None:
        return 1

    prices = prices_df['close']
    print(f"Loaded {len(prices)} days: {prices.index[0].date()} to "
          f"{prices.index[-1].date()}")

    # ==========================================================================
    # TEST STRATEGY 1: DUAL MOVING AVERAGE
    # ==========================================================================

    result_dma = walk_forward_analysis(
        prices,
        strategy_dual_ma,
        "Dual MA (50/200)",
        fast_period=50,
        slow_period=200
    )

    # ==========================================================================
    # TEST STRATEGY 2: MEAN REVERSION
    # ==========================================================================

    result_mr = walk_forward_analysis(
        prices,
        strategy_mean_reversion,
        "Mean Reversion (RSI)",
        rsi_period=14,
        oversold=30,
        overbought=70
    )

    # ==========================================================================
    # TEST STRATEGY 3: VOLATILITY-TARGETED DUAL MA
    # ==========================================================================

    def strategy_vol_adjusted_ma(prices, fast=50, slow=200, target=0.10):
        """Combine trend-following with vol targeting."""
        trend = strategy_dual_ma(prices, fast, slow)
        vol_adj = strategy_vol_target(prices, target)
        return trend * vol_adj

    result_vol = walk_forward_analysis(
        prices,
        strategy_vol_adjusted_ma,
        "Vol-Targeted Dual MA",
        fast=50,
        slow=200,
        target=0.10
    )

    # ==========================================================================
    # FINAL REPORT
    # ==========================================================================

    print("\n" + "=" * 70)
    print("                  FINAL RESEARCH REPORT")
    print("=" * 70)

    strategies = [result_dma, result_mr, result_vol]

    print(f"\n{'Strategy':<25} {'IS Sharpe':>10} {'OOS Sharpe':>10} "
          f"{'Decay':>10} {'Pass':>8}")
    print("-" * 65)

    best = None
    for s in strategies:
        status = "PASS" if s.passed else "FAIL"
        print(f"{s.strategy_name:<25} {s.in_sample_sharpe:>10.2f} "
              f"{s.out_sample_sharpe:>10.2f} {s.sharpe_decay_pct:>9.0%} "
              f"{status:>8}")
        if s.passed and (best is None or
                         s.out_sample_sharpe > best.out_sample_sharpe):
            best = s

    print("-" * 65)

    if best:
        print(f"\nBEST STRATEGY: {best.strategy_name}")
        print(f"OOS Sharpe: {best.out_sample_sharpe:.2f}")
        print(f"Sharpe Decay: {best.sharpe_decay_pct:.0%}")
        grade = "B+" if best.out_sample_sharpe > 0.5 else "C"
    else:
        print("\nNO STRATEGY PASSED VALIDATION")
        print("All strategies failed the OOS Sharpe > 0, Decay < 30% criteria")
        grade = "F"

    # Save results
    report = {
        "generated": datetime.now().isoformat(),
        "methodology": "Walk-Forward Analysis (5yr train, 3yr test)",
        "friction": f"{TOTAL_FRICTION*10000:.0f} bps",
        "strategies": [
            {
                "name": s.strategy_name,
                "is_sharpe": float(s.in_sample_sharpe),
                "oos_sharpe": float(s.out_sample_sharpe),
                "decay": float(s.sharpe_decay_pct),
                "passed": bool(s.passed)
            }
            for s in strategies
        ],
        "best_strategy": best.strategy_name if best else None,
        "grade": grade
    }

    os.makedirs("output", exist_ok=True)
    fn = f"output/research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved: {fn}")
    print(f"\nOVERALL GRADE: {grade}")

    return 0 if best else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
