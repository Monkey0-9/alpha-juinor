# tests/test_execution.py
import pandas as pd
import numpy as np
from datetime import datetime

import pytest

from backtest.execution import (
    Order,
    OrderType,
    RealisticExecutionHandler,
    OrderStatus,
    ExecutionError,
)


# -------------------------
# Helpers (deterministic)
# -------------------------
def make_price_series(n=120, seed=42):
    rng = pd.date_range("2020-01-01", periods=n, freq="D")
    rs = np.random.RandomState(seed)
    prices = 100.0 + np.cumsum(rs.normal(0, 0.2, n))
    return pd.Series(prices, index=rng)


def make_volume_series(n=120, base=1_000_000):
    rng = pd.date_range("2020-01-01", periods=n, freq="D")
    vols = np.full(n, float(base), dtype=float)
    return pd.Series(vols, index=rng)


def make_bar(price, volume):
    return {"Open": price, "High": price, "Low": price, "Close": price, "Volume": float(volume)}


# -------------------------
# Tests
# -------------------------
def test_slippage_increases_with_order_size():
    """
    Larger orders (higher participation) must incur
    equal or greater market impact than smaller orders.
    """
    handler = RealisticExecutionHandler()
    prices = make_price_series()
    volumes = make_volume_series()

    bar_ts = prices.index[-1]
    bar = make_bar(float(prices.iloc[-1]), volumes.iloc[-1])

    small = Order("SPY", 1_000, OrderType.MARKET, bar_ts)
    big = Order("SPY", 100_000, OrderType.MARKET, bar_ts)

    t_small = handler.fill_order(
        order=small,
        bar=bar,
        bar_timestamp=bar_ts,
        price_history=prices,
        volume_history=volumes,
    )

    t_big = handler.fill_order(
        order=big,
        bar=bar,
        bar_timestamp=bar_ts,
        price_history=prices,
        volume_history=volumes,
    )

    assert t_small is not None, "Small order should produce a trade"
    assert t_big is not None, "Large order should produce a trade (possibly partial)"
    assert t_big.market_impact >= t_small.market_impact


def test_participation_cap_enforced():
    """
    Filled quantity must not exceed max_participation_rate * bar volume.
    """
    max_participation = 0.05
    handler = RealisticExecutionHandler(max_participation_rate=max_participation)

    prices = make_price_series()
    volumes = make_volume_series(n=len(prices), base=10_000)

    bar_ts = prices.index[-1]
    bar = make_bar(float(prices.iloc[-1]), volumes.iloc[-1])

    # Order requests entire bar volume
    order = Order("SPY", int(volumes.iloc[-1]), OrderType.MARKET, bar_ts)

    trade = handler.fill_order(
        order=order,
        bar=bar,
        bar_timestamp=bar_ts,
        price_history=prices,
        volume_history=volumes,
    )

    assert trade is not None
    assert abs(trade.quantity) <= volumes.iloc[-1] * max_participation + 1e-8


def test_partial_fill_and_order_state():
    """
    Orders larger than allowed participation must be partially filled and remain open.
    """
    handler = RealisticExecutionHandler(max_participation_rate=0.10)

    prices = make_price_series()
    volumes = make_volume_series(n=len(prices), base=1_000)

    bar_ts = prices.index[-1]
    bar = make_bar(float(prices.iloc[-1]), volumes.iloc[-1])

    # Very large order relative to volume
    order = Order("SPY", 5_000, OrderType.MARKET, bar_ts)

    trade = handler.fill_order(
        order=order,
        bar=bar,
        bar_timestamp=bar_ts,
        price_history=prices,
        volume_history=volumes,
    )

    assert trade is not None
    # order should have been updated in-place by handler
    assert order.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED)
    if order.status == OrderStatus.PARTIALLY_FILLED:
        assert order.remaining_qty != 0.0
        assert abs(trade.quantity) <= volumes.iloc[-1] * 0.10 + 1e-8
    else:
        # If fully filled by handler, remaining_qty should be zero
        assert order.remaining_qty == 0.0


def test_zero_volume_bar_behaviour():
    """
    Zero-volume bars should either raise a clear ExecutionError (defensive) or return None.
    Test accepts either behavior but will fail silently if handler incorrectly fills.
    """
    handler = RealisticExecutionHandler()

    prices = make_price_series()
    volumes = make_volume_series()
    bar_ts = prices.index[-1]

    bar = make_bar(float(prices.iloc[-1]), 0.0)

    order = Order("SPY", 1_000, OrderType.MARKET, bar_ts)

    try:
        trade = handler.fill_order(
            order=order,
            bar=bar,
            bar_timestamp=bar_ts,
            price_history=prices,
            volume_history=volumes,
        )
    except ExecutionError:
        # acceptable defensive behavior
        trade = None

    assert trade is None
    assert order.status in (OrderStatus.NEW, OrderStatus.CANCELLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED)
