
import pytest
from unittest.mock import MagicMock, patch
from brokers.alpaca_broker import AlpacaExecutionHandler

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP Error")

@pytest.fixture
def alpaca_handler():
    # Use patch to avoid real network calls during init
    with patch('requests.Session') as mock_session:
        mock_instance = mock_session.return_value
        # Mock get_account call in __init__
        mock_instance.request.return_value = MockResponse({'status': 'ACTIVE', 'equity': '100000'})

        handler = AlpacaExecutionHandler('test_key', 'test_secret')
        # Ensure session is the mock
        handler.session = mock_instance
        return handler

def test_submit_order_success(alpaca_handler):
    # Mock _request_with_retry to return success
    with patch.object(alpaca_handler, '_request_with_retry') as mock_request:
        mock_request.return_value = MockResponse({'id': 'test_order_id', 'status': 'new'})

        result = alpaca_handler.submit_order('AAPL', 10, 'buy')

        assert result['success'] is True
        assert result['order']['id'] == 'test_order_id'
        assert result['error'] is None
        assert result['mapped_symbol'] == 'AAPL'

def test_submit_order_failure(alpaca_handler):
    # Mock _request_with_retry to raise exception (simulate exhaustion)
    with patch.object(alpaca_handler, '_request_with_retry') as mock_request:
        mock_request.side_effect = Exception("API Error")

        result = alpaca_handler.submit_order('AAPL', 10, 'buy')

        assert result['success'] is False
        assert result['order'] is None
        assert "API Error" in result['error']

def test_submit_order_negative_qty(alpaca_handler):
    result = alpaca_handler.submit_order('AAPL', -5, 'buy')
    assert result['success'] is False
    assert "Qty must be positive" in result['error']

def test_symbol_normalization(alpaca_handler):
    with patch.object(alpaca_handler, '_request_with_retry') as mock_request:
        mock_request.return_value = MockResponse({'id': 'test_crypto', 'status': 'new'})

        # Test Crypto
        result = alpaca_handler.submit_order('BTC-USD', 0.1, 'buy')
        assert result['mapped_symbol'] == 'BTC/USD'

        # Test Stock with hyphen
        result = alpaca_handler.submit_order('BRK-B', 1, 'buy')
        assert result['mapped_symbol'] == 'BRKB'
