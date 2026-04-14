import os

import pytest
import requests
from dotenv import load_dotenv

load_dotenv()


def _live_enabled() -> bool:
    return os.getenv("RUN_LIVE_API_TESTS", "").lower() in {"1", "true", "yes"}


def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }


@pytest.mark.skipif(not _live_enabled(), reason="Live API tests disabled")
def test_alpaca_stock_bars_endpoint():
    url = "https://data.alpaca.markets/v2/stocks/AAPL/bars"
    params = {"start": "2023-01-01", "end": "2023-01-10", "timeframe": "1Day"}
    r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
    assert r.status_code == 200


@pytest.mark.skipif(not _live_enabled(), reason="Live API tests disabled")
def test_alpaca_crypto_bars_endpoint():
    url = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
    params = {
        "symbols": "BTC/USD",
        "start": "2023-01-01",
        "end": "2023-01-10",
        "timeframe": "1Day",
    }
    r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
    assert r.status_code == 200
