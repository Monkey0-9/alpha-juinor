#!/usr/bin/env python3
"""Sanity checks for Alpaca credential wiring and component construction."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def _has_alpaca_env() -> bool:
    return bool(
        os.getenv("ALPACA_API_KEY")
        and os.getenv("ALPACA_SECRET_KEY")
        and os.getenv("ALPACA_BASE_URL")
    )


@pytest.mark.skipif(not _has_alpaca_env(), reason="Alpaca credentials not configured")
def test_alpaca_env_present():
    assert os.getenv("ALPACA_API_KEY")
    assert os.getenv("ALPACA_SECRET_KEY")
    assert os.getenv("ALPACA_BASE_URL")


@pytest.mark.skipif(not _has_alpaca_env(), reason="Alpaca credentials not configured")
def test_alpaca_components_initialize():
    from mini_quant_fund.brokers.alpaca_broker import AlpacaExecutionHandler
    from mini_quant_fund.data.collectors.alpaca_collector import AlpacaDataProvider
    from mini_quant_fund.execution.alpaca_handler import (
        AlpacaExecutionHandler as ExecHandler,
    )

    broker = AlpacaExecutionHandler()
    provider = AlpacaDataProvider()
    exec_handler = ExecHandler(paper=True)

    assert broker is not None
    assert provider is not None
    assert exec_handler is not None
