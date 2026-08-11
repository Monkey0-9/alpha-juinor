import pytest
import os
import pandas as pd
import numpy as np
from nexus.research.walk_forward import WalkForwardEvaluator
from nexus.data.parquet_store import ParquetDataStore
from nexus.models.zoo.time_series import GradientBoostedTimeSeriesModel


def _create_sample_bars(rows=500):
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(rows) * 0.5)
    returns = np.zeros(rows)
    returns[1:] = np.diff(close) / close[:-1]

    df = pd.DataFrame({
        "close": close,
        "returns": returns,
        "rsi_14": np.random.uniform(30, 70, rows),
        "macd": np.random.randn(rows) * 0.1,
        "macd_signal": np.random.randn(rows) * 0.1,
        "macd_hist": np.random.randn(rows) * 0.1,
        "roc_10": np.random.randn(rows) * 0.05,
        "bb_mean": close,
        "bb_std": np.random.uniform(1, 5, rows),
        "atr_14": np.random.uniform(0.5, 2, rows)
    }, index=dates)
    return df


def test_walk_forward_splits():
    df = _create_sample_bars(350)
    evaluator = WalkForwardEvaluator(train_window=200, val_window=50, test_window=50, step_size=50)
    splits = evaluator.generate_splits(df)
    assert len(splits) >= 1
    assert "train" in splits[0]
    assert "val" in splits[0]
    assert "test" in splits[0]


def test_walk_forward_evaluation():
    df = _create_sample_bars(400)
    evaluator = WalkForwardEvaluator(train_window=200, val_window=50, test_window=50, step_size=50)
    results = evaluator.evaluate_model(GradientBoostedTimeSeriesModel, df)
    assert results["status"] == "success"
    assert "cumulative_return" in results
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results


def test_parquet_store_save_and_load(tmp_path):
    store = ParquetDataStore(store_dir=str(tmp_path))
    df = _create_sample_bars(100)

    success = store.save_bars("AAPL", df, timeframe="1D")
    assert success

    loaded = store.load_bars("AAPL", timeframe="1D", limit=50)
    assert not loaded.empty
    assert len(loaded) == 50

    symbols = store.list_available_symbols()
    assert len(symbols) == 1
    assert symbols[0]["symbol"] == "AAPL"
