"""
Test Suite for Alternative Data Integrations
===========================================

Tests for satellite, credit card, and geolocation adapters.
"""

import pytest

from alternative_data.integrations.credit_card_adapter import (
    CreditCardAdapter,
    TransactionSignal,
)
from alternative_data.integrations.geolocation_adapter import (
    GeolocationAdapter,
    MobilitySignal,
)
from alternative_data.integrations.satellite_adapter import (
    SatelliteAdapter,
    SatelliteSignal,
)


class TestSatelliteAdapter:
    """Test satellite imagery adapter."""

    def test_parking_lot_traffic(self):
        """Test parking lot traffic signal retrieval."""
        adapter = SatelliteAdapter()  # No API key = simulated data
        signal = adapter.get_parking_lot_traffic("WMT", "store_12345")

        assert signal is not None
        assert signal.symbol == "WMT"
        assert signal.metric_type == "parking_lot_traffic"
        assert signal.confidence > 0

    def test_industrial_activity(self):
        """Test industrial activity signal."""
        adapter = SatelliteAdapter()
        signal = adapter.get_industrial_activity("CAT", "facility_789")

        assert signal is not None
        assert signal.metric_type == "industrial_activity"
        assert signal.value >= 0

    def test_alpha_signal_conversion(self):
        """Test conversion to trading signal."""
        adapter = SatelliteAdapter()
        signal = adapter.get_parking_lot_traffic("TGT", "store_001")

        alpha = adapter.get_alpha_signal(signal)
        assert alpha in [-1, 0, 1]


class TestCreditCardAdapter:
    """Test credit card transaction adapter."""

    def test_revenue_growth(self):
        """Test revenue growth signal."""
        adapter = CreditCardAdapter()
        signal = adapter.get_revenue_growth("AMZN")

        assert signal is not None
        assert signal.metric_type == "revenue_growth"
        assert signal.confidence > 0

    def test_transaction_volume(self):
        """Test transaction volume signal."""
        adapter = CreditCardAdapter()
        signal = adapter.get_transaction_volume("V")

        assert signal is not None
        assert signal.metric_type == "transaction_volume"

    def test_aov_signal(self):
        """Test average order value signal."""
        adapter = CreditCardAdapter()
        signal = adapter.get_average_order_value("SHOP")

        assert signal is not None
        assert signal.value > 0

    def test_alpha_conversion(self):
        """Test trading signal conversion."""
        adapter = CreditCardAdapter()
        signal = adapter.get_revenue_growth("NFLX")

        alpha = adapter.get_alpha_signal(signal)
        assert alpha in [-1, 0, 1]


class TestGeolocationAdapter:
    """Test geolocation mobility adapter."""

    def test_foot_traffic(self):
        """Test foot traffic signal."""
        adapter = GeolocationAdapter()
        signal = adapter.get_foot_traffic("SBUX", "store_abc123")

        assert signal is not None
        assert signal.metric_type == "foot_traffic"
        assert signal.value >= 0

    def test_dwell_time(self):
        """Test dwell time signal."""
        adapter = GeolocationAdapter()
        signal = adapter.get_dwell_time("HD", "store_xyz789")

        assert signal is not None
        assert signal.metric_type == "dwell_time"
        assert signal.value > 0

    def test_trade_area_penetration(self):
        """Test trade area penetration."""
        adapter = GeolocationAdapter()
        signal = adapter.get_trade_area_penetration("MCD", "store_001", radius_km=3.0)

        assert signal is not None
        assert signal.metric_type == "trade_area_penetration"
        assert 0 <= signal.value <= 100

    def test_alpha_signal(self):
        """Test alpha signal conversion."""
        adapter = GeolocationAdapter()
        signal = adapter.get_foot_traffic("NKE", "store_123")

        alpha = adapter.get_alpha_signal(signal)
        assert alpha in [-1, 0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
