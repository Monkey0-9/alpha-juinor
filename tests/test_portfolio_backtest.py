import numpy as np
import pandas as pd
from nexus.research.portfolio_backtest import PortfolioBacktester


def test_portfolio_backtester_run():
    dates = pd.date_range(start="2024-01-01", periods=60, freq="B")
    symbols = ["AAA", "BBB", "CCC"]
    price_data = {
        symbol: pd.Series(100 + np.cumsum(np.random.normal(0.1, 1.0, len(dates))), index=dates)
        for symbol in symbols
    }
    signal_data = {
        symbol: pd.Series(np.random.uniform(-1, 1, len(dates)), index=dates)
        for symbol in symbols
    }

    backtester = PortfolioBacktester(initial_capital=1000000.0)
    result = backtester.run(price_data, signal_data, max_positions=2)

    assert result.equity_curve.index.equals(dates)
    assert result.total_return == float(result.equity_curve.iloc[-1] / 1000000.0 - 1)
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.max_drawdown, float)
    assert 0.0 <= result.win_rate <= 1.0


def test_portfolio_backtester_walkforward():
    dates = pd.date_range(start="2024-01-01", periods=120, freq="B")
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    price_data = {
        symbol: pd.Series(100 + np.cumsum(np.random.normal(0.1, 1.0, len(dates))), index=dates)
        for symbol in symbols
    }
    signal_data = {
        symbol: pd.Series(np.random.uniform(-1, 1, len(dates)), index=dates)
        for symbol in symbols
    }

    backtester = PortfolioBacktester(initial_capital=1000000.0)
    result = backtester.run_walkforward(
        price_data,
        signal_data,
        train_window=50,
        test_window=20,
        step=20,
        max_positions=2,
    )

    assert result["num_folds"] >= 1
    assert isinstance(result["total_return"], float)
    assert isinstance(result["average_sharpe"], float)
    assert isinstance(result["average_max_drawdown"], float)
    assert isinstance(result["fold_results"], list)
