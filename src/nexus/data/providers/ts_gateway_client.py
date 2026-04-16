"""
Python consumer for the TypeScript live-data gateway.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import websockets

logger = logging.getLogger(__name__)


@dataclass
class GatewayTick:
    symbol: str
    price: float
    ts: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    source: str = "unknown"


class TSGatewayClient:
    def __init__(self, url: str = "ws://127.0.0.1:8787/stream") -> None:
        self.url = url
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest: Dict[str, GatewayTick] = {}
        self._callbacks: list[Callable[[GatewayTick], None]] = []

    def on_tick(self, callback: Callable[[GatewayTick], None]) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def get_latest(self, symbol: str) -> Optional[GatewayTick]:
        return self._latest.get(symbol)

    def _run_thread(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        reconnect_delay = 1.0
        while self._running:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=10) as ws:
                    reconnect_delay = 1.0
                    async for raw in ws:
                        if not self._running:
                            break
                        self._handle_message(raw)
            except Exception as exc:
                logger.warning("TS gateway reconnect after error: %s", exc)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2.0, 10.0)

    def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("t")
        if msg_type == "snapshot":
            for tick_data in msg.get("ticks", []):
                tick = self._parse_tick(tick_data)
                if tick:
                    self._publish(tick)
            return

        if msg_type == "tick":
            tick = self._parse_tick(msg)
            if tick:
                self._publish(tick)

    def _parse_tick(self, data: Dict[str, Any]) -> Optional[GatewayTick]:
        symbol = data.get("s")
        price = data.get("p")
        ts_ms = data.get("ts")
        if not isinstance(symbol, str) or not isinstance(price, (int, float)) or not isinstance(ts_ms, (int, float)):
            return None
        return GatewayTick(
            symbol=symbol,
            price=float(price),
            ts=datetime.utcfromtimestamp(float(ts_ms) / 1000.0),
            bid=float(data["b"]) if isinstance(data.get("b"), (int, float)) else None,
            ask=float(data["a"]) if isinstance(data.get("a"), (int, float)) else None,
            volume=float(data["v"]) if isinstance(data.get("v"), (int, float)) else None,
            source=str(data.get("src", "unknown")),
        )

    def _publish(self, tick: GatewayTick) -> None:
        self._latest[tick.symbol] = tick
        for callback in self._callbacks:
            try:
                callback(tick)
            except Exception:
                logger.exception("TS gateway callback failed")


_client_singleton: Optional[TSGatewayClient] = None


def get_ts_gateway_client(url: str = "ws://127.0.0.1:8787/stream") -> TSGatewayClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = TSGatewayClient(url=url)
    return _client_singleton
