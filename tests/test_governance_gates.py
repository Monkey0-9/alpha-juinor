
import pytest
import pandas as pd
import os
import sys
import shutil
from unittest.mock import patch, MagicMock

# MOCK HEAVY DEPENDENCIES BEFORE IMPORTING main
sys.modules["strategies"] = MagicMock()
sys.modules["strategies.factory"] = MagicMock()
sys.modules["risk"] = MagicMock()
sys.modules["risk.engine"] = MagicMock()
sys.modules["portfolio"] = MagicMock()
sys.modules["portfolio.allocator"] = MagicMock()
sys.modules["brokers"] = MagicMock()
sys.modules["brokers.alpaca_broker"] = MagicMock()
sys.modules["brokers.mock_broker"] = MagicMock()
sys.modules["database"] = MagicMock()
sys.modules["database.manager"] = MagicMock()
sys.modules["data.collectors"] = MagicMock()
sys.modules["data.collectors.data_router"] = MagicMock()

# Now import the target modules
from data.governance.provider_matrix import select_provider, PROVIDER_CAPABILITIES
from data.governance.quality_engine import DataQualityEngine
from main import InstitutionalLiveAgent

class TestGovernanceGates:

    def test_provider_selection_logic(self):
        """Verify strict provider routing logic."""
        # 1. Asset Class Routing
        assert select_provider("AAPL", 5) in ["polygon", "yahoo", "alpaca"]
        assert select_provider("BTC-USD", 5) in ["yahoo", "alpaca", "polygon"]

        # 2. History Constraint (Alpaca max 1825 < 5000)
        # 6000 days should force Yahoo or Polygon
        # (Alpaca is 1825, Polygon 5000, Yahoo 10000)
        selected = select_provider("AAPL", 6000)
        assert selected == "yahoo"

        # 3. Entitlement Checks (Mocked)
        with patch.dict(os.environ, {"POLYGON_API_KEY": ""}, clear=True):
             # Polygon disabled, Yahoo should win for Stocks if Alpaca keys also missing
             with patch.dict(os.environ, {"ALPACA_API_KEY": ""}, clear=True):
                 selected = select_provider("AAPL", 100)
                 assert selected == "yahoo"

    def test_quality_engine_scoring(self):
        """Verify quality engine logic."""
        # Perfect Data
        df_perfect = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [95.0, 96.0, 97.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [1000, 1000, 1000]
        }, index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))

        score, details = DataQualityEngine.validate_ohlcv(df_perfect, "TEST")
        assert score == 1.0
        assert details["status"] == "SUCCESS"

        # Corrupt Data (Zero Price)
        df_zero = df_perfect.copy()
        # Set a Close price to 0
        df_zero.iloc[1, df_zero.columns.get_loc("Close")] = 0.0
        score, details = DataQualityEngine.validate_ohlcv(df_zero, "TEST")
        assert score == 0.0
        assert any("ZERO_NEGATIVE_PRICES" in f for f in details["flags"])

        # Gap Data
        # Ensure gap is strictly > 5 days as per engine logic (which is loose)
        dates = [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-10")] # 9 days diff
        df_gap = pd.DataFrame({
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [95.0, 96.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1000]
        }, index=pd.to_datetime(dates))

        score, details = DataQualityEngine.validate_ohlcv(df_gap, "TEST")
        assert score < 1.0
        assert any("MISSING_DATES_GAPS" in f for f in details["flags"])

    def test_kill_switch_activates(self):
        """Verify kill switch halts start."""
        # Clean env
        if os.path.exists("kill_switch.txt"):
            os.remove("kill_switch.txt")

        # Mock DatabaseManager for the agent init
        with patch("main.DatabaseManager") as mock_db:
             with patch("main.ConfigManager"): # config also needs mock
                 agent = InstitutionalLiveAgent(tickers=["AAPL"])

                 # Test False state
                 assert agent.check_kill_switch() is False

                 # Create kill switch file
                 with open("kill_switch.txt", "w") as f:
                     f.write("HALT")

                 try:
                     assert agent.check_kill_switch() is True
                 finally:
                     if os.path.exists("kill_switch.txt"):
                         os.remove("kill_switch.txt")
