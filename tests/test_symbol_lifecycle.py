
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
from governance.lifecycle_manager import LifecycleManager, SymbolState
from database.manager import SymbolGovernanceRecord

class TestLifecycleManager:

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def mock_data_router(self):
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_db, mock_data_router):
        return LifecycleManager(mock_db, mock_data_router)

    def test_active_symbol_stays_active_if_healthy(self, manager, mock_db, mock_data_router):
        """Test that a healthy ACTIVE symbol remains ACTIVE."""
        symbol = "AAPL"

        # Mock Gov Record
        mock_db.get_symbol_governance.return_value = {
            "symbol": symbol,
            "state": SymbolState.ACTIVE.value,
            "last_checked_ts": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "data_quality": 0.95
        }

        # Mock Data
        df = pd.DataFrame({"Close": [100.0] * 100})
        mock_data_router.get_daily_prices.return_value = df

        # Mock Quality Check (implicit in logic or mocked out)
        # We'll rely on the real compute_data_quality for now or mock if it's external
        with patch('governance.lifecycle_manager.compute_data_quality', return_value=(0.95, ["OK"])):
            manager.run_lifecycle_check(symbol)

        # Verify NO state change
        # Verify stats update but still ACTIVE
        mock_db.upsert_symbol_governance.assert_called_once()
        args = mock_db.upsert_symbol_governance.call_args[0][0]
        assert args.state == SymbolState.ACTIVE.value
        assert args.data_quality == 0.95

    def test_active_symbol_quarantined_on_bad_data(self, manager, mock_db, mock_data_router):
        """Test that ACTIVE symbol is QUARANTINED if quality drops."""
        symbol = "JUNK"

        mock_db.get_symbol_governance.return_value = {
            "symbol": symbol,
            "state": SymbolState.ACTIVE.value,
            "last_checked_ts": datetime.utcnow().isoformat(),
            "data_quality": 0.95
        }

        # Mock Bad Data (Empty or Quality < Threshold)
        df = pd.DataFrame({"Close": [100.0] * 10}) # Too short?
        mock_data_router.get_daily_prices.return_value = df

        with patch('governance.lifecycle_manager.compute_data_quality', return_value=(0.4, ["BAD_DATA"])):
             manager.run_lifecycle_check(symbol)

        # Verify State Change
        args = mock_db.upsert_symbol_governance.call_args[0][0]
        assert args.symbol == symbol
        assert args.state == SymbolState.QUARANTINED.value
        assert "BAD_DATA" in args.reason

    def test_quarantined_symbol_retired_after_timeout(self, manager, mock_db, mock_data_router):
        """Test that QUARANTINED symbol is RETIRED after 30 days."""
        symbol = "OLD_JUNK"

        # Quarantined 31 days ago
        last_checked = (datetime.utcnow() - timedelta(days=31)).isoformat()

        mock_db.get_symbol_governance.return_value = {
            "symbol": symbol,
            "state": SymbolState.QUARANTINED.value,
            "last_checked_ts": last_checked,
            "data_quality": 0.4,
            "metadata_json": '{"quarantined_at": "' + last_checked + '"}'
        }

        # Data still bad
        mock_data_router.get_daily_prices.return_value = pd.DataFrame() # No data

        with patch('governance.lifecycle_manager.compute_data_quality', return_value=(0.0, ["NO_DATA"])):
            manager.run_lifecycle_check(symbol)

        # Verify Retirement
        args = mock_db.upsert_symbol_governance.call_args[0][0]
        assert args.state == SymbolState.RETIRED.value
        assert "Timeout" in args.reason

    def test_quarantined_symbol_restored_if_fixed(self, manager, mock_db, mock_data_router):
        """Test that QUARANTINED symbol becomes ACTIVE if data issue resolved."""
        symbol = "FIXED_SYM"

        mock_db.get_symbol_governance.return_value = {
            "symbol": symbol,
            "state": SymbolState.QUARANTINED.value,
            "last_checked_ts": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "data_quality": 0.4
        }

        # Data is now good
        df = pd.DataFrame({"Close": [100.0] * 300})
        mock_data_router.get_daily_prices.return_value = df

        with patch('governance.lifecycle_manager.compute_data_quality', return_value=(0.9, ["OK"])):
             manager.run_lifecycle_check(symbol)

        args = mock_db.upsert_symbol_governance.call_args[0][0]
        assert args.state == SymbolState.ACTIVE.value
        assert args.reason == "Quality Restored"
