"""
Test institutional alert system.
"""
import pytest
from monitoring.institutional_alerts import (
    InstitutionalAlertManager,
    AlertSeverity,
    AlertDeduplicator
)
from datetime import timedelta


def test_alert_deduplication():
    """Test that alerts are deduplicated within cooldown window."""
    dedup = AlertDeduplicator(default_cooldown_minutes=1)
    
    # First alert should send
    assert dedup.should_send("TEST", "message1") is True
    
    # Same alert within cooldown should not send
    assert dedup.should_send("TEST", "message1") is False
    
    # Different message should send
    assert dedup.should_send("TEST", "message2") is True


def test_alert_severity_routing():
    """Test that severity routing is correct."""
    from monitoring.institutional_alerts import SEVERITY_ROUTING, AlertChannel
    
    # DEBUG and INFO should only go to file
    assert SEVERITY_ROUTING[AlertSeverity.DEBUG] == [AlertChannel.FILE_LOG]
    assert SEVERITY_ROUTING[AlertSeverity.INFO] == [AlertChannel.FILE_LOG]
    
    # ERROR and CRITICAL should go to Telegram
    assert AlertChannel.TELEGRAM in SEVERITY_ROUTING[AlertSeverity.ERROR]
    assert AlertChannel.TELEGRAM in SEVERITY_ROUTING[AlertSeverity.CRITICAL]


def test_alert_manager_basic():
    """Test basic alert manager functionality."""
    manager = InstitutionalAlertManager()
    
    # Should not raise
    manager.alert("Test message", severity=AlertSeverity.INFO)
    manager.alert("Test error", severity=AlertSeverity.ERROR, category="TEST")
    
    # Update daily summary
    manager.update_daily_summary(nav=100000, pnl=5.2, trades=10)
    assert manager.daily_summary_data["nav"] == 100000
    assert manager.daily_summary_data["pnl"] == 5.2


def test_risk_breach_alert():
    """Test risk breach alert."""
    manager = InstitutionalAlertManager()
    
    # Should create ERROR alert
    manager.risk_breach("VaR", 0.06, 0.04)
    
    # Second call should be deduplicated
    # (would need to mock time to test properly)


def test_circuit_breaker_alert():
    """Test circuit breaker alert."""
    manager = InstitutionalAlertManager()
    
    # Should create CRITICAL alert with PIN
    manager.circuit_breaker("FREEZE", "Drawdown > 18%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
