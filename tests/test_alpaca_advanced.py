import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp
from nexus.execution.alpaca import AlpacaClient, AlpacaCredentials


class MockAsyncContext:
    def __init__(self, response_or_exc):
        self.response_or_exc = response_or_exc

    async def __aenter__(self):
        if isinstance(self.response_or_exc, Exception):
            raise self.response_or_exc
        return self.response_or_exc

    async def __aexit__(self, exc_type, exc, tb):
        pass


class MockAsyncContextSequence:
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    async def __aenter__(self):
        resp = self.responses[self.index]
        self.index += 1
        return resp

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest.mark.asyncio
async def test_alpaca_rate_limit_retry():
    """Verify that the engine retries upon a 429 rate limit error when fetching bars."""
    service = AlpacaClient(
        credentials=AlpacaCredentials(api_key="TEST", api_secret="TEST")
    )
    service.simulated = False

    mock_response_429 = AsyncMock()
    mock_response_429.status = 429
    mock_response_429.headers = {"Retry-After": "1"}

    mock_response_200 = AsyncMock()
    mock_response_200.status = 200
    mock_response_200.json.return_value = {"bars": [{"close": 100}]}

    # session.get() is a normal function returning an async context manager
    mock_session = MagicMock()
    mock_session.get.return_value = MockAsyncContextSequence(
        [mock_response_429, mock_response_200]
    )

    with patch.object(
        service, "_get_session", new_callable=AsyncMock
    ) as mock_get_session:
        mock_get_session.return_value = mock_session
        bars = await service.get_bars("SPY", limit=1)
        assert len(bars) == 1
        assert bars[0]["close"] == 100


@pytest.mark.asyncio
async def test_alpaca_connection_error_fallback():
    """Verify that network errors return empty lists instead of blowing up."""
    service = AlpacaClient(
        credentials=AlpacaCredentials(api_key="TEST", api_secret="TEST")
    )
    service.simulated = False

    mock_session = MagicMock()
    exc = aiohttp.ClientConnectorError(
        connection_key=MagicMock(), os_error=OSError("Network down")
    )
    mock_session.get.return_value = MockAsyncContext(exc)

    with patch.object(
        service, "_get_session", new_callable=AsyncMock
    ) as mock_get_session:
        mock_get_session.return_value = mock_session
        bars = await service.get_bars("SPY", limit=1)
        assert bars == []


@pytest.mark.asyncio
async def test_alpaca_order_rejection():
    """Verify that order rejections return cleanly with success=False."""
    service = AlpacaClient(
        credentials=AlpacaCredentials(api_key="TEST", api_secret="TEST")
    )
    service.simulated = False

    mock_response_403 = AsyncMock()
    mock_response_403.status = 403
    mock_response_403.json.return_value = {
        "message": "Insufficient buying power"
    }

    mock_session = MagicMock()
    mock_session.post.return_value = MockAsyncContext(mock_response_403)

    with patch.object(
        service, "_get_session", new_callable=AsyncMock
    ) as mock_get_session:
        mock_get_session.return_value = mock_session
        result = await service.submit_order("SPY", 10, "buy")
        assert result["success"] is False
        assert "error" in result
