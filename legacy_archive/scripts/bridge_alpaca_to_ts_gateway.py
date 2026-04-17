#!/usr/bin/env python3
"""
Bridge Alpaca websocket ticks into the TypeScript gateway ingest endpoint.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import websocket

from mini_quant_fund.data.providers.alpaca_websocket import get_alpaca_feed


def _to_epoch_ms(raw_ts: str) -> int:
    if not raw_ts:
        return int(time.time() * 1000)
    try:
        normalized = raw_ts.replace("Z", "+00:00")
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


def main() -> int:
    gateway_url = os.getenv("TS_GATEWAY_INGEST_URL", "ws://127.0.0.1:8787/ingest")
    symbols = os.getenv("ALPACA_STREAM_SYMBOLS", "AAPL,MSFT,TSLA").split(",")
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    feed_name = os.getenv("ALPACA_FEED", "iex")

    ws = websocket.create_connection(gateway_url, timeout=5)
    feed = get_alpaca_feed(feed=feed_name)

    def on_trade(trade: dict) -> None:
        tick = {
            "symbol": trade.get("symbol"),
            "price": float(trade.get("price", 0.0)),
            "volume": float(trade.get("size", 0.0)),
            "ts": _to_epoch_ms(str(trade.get("timestamp", ""))),
            "source": "alpaca_trade",
        }
        ws.send(json.dumps(tick))

    def on_quote(quote: dict) -> None:
        mid = (float(quote.get("bid", 0.0)) + float(quote.get("ask", 0.0))) / 2.0
        tick = {
            "symbol": quote.get("symbol"),
            "price": mid,
            "bid": float(quote.get("bid", 0.0)),
            "ask": float(quote.get("ask", 0.0)),
            "ts": _to_epoch_ms(str(quote.get("timestamp", ""))),
            "source": "alpaca_quote",
        }
        ws.send(json.dumps(tick))

    feed.on_trade(on_trade)
    feed.on_quote(on_quote)
    feed.subscribe_trades(symbols)
    feed.subscribe_quotes(symbols)
    feed.start()

    print(f"Bridging Alpaca feed '{feed_name}' for {symbols} -> {gateway_url}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()
        ws.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
