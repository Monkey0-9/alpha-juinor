import asyncio
import json
from unittest.mock import AsyncMock

from nexus.execution.alpaca import AlpacaClient, AlpacaCredentials


class FakeResponse:
    def __init__(self, status, payload=None):
        self._status = status
        self._payload = payload or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def status(self):
        return self._status

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)


class FakeSession:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._response


def test_get_bars_deduplicates_concurrent_requests():
    async def run_test():
        response = FakeResponse(
            200,
            {
                "bars": [
                    {
                        "t": "2024-01-01T00:00:00Z",
                        "o": 1.0,
                        "h": 1.1,
                        "l": 0.9,
                        "c": 1.0,
                        "v": 100,
                    }
                ]
            },
        )
        session = FakeSession(response)
        client = AlpacaClient(credentials=AlpacaCredentials("key", "secret"))
        client._get_session = AsyncMock(return_value=session)

        results = await asyncio.gather(
            client.get_bars("AAPL"), client.get_bars("AAPL")
        )

        assert len(results[0]) == 1
        assert results[0] == results[1]
        assert len(session.calls) == 1

    asyncio.run(run_test())
