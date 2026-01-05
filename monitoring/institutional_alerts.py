"""
Institutional Alert System

Severity-based routing with deduplication and cooldown.
Philosophy: Silence = Healthy. Noise = Broken.
"""
import logging
import time
from enum import Enum
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels with routing rules."""
    DEBUG = "DEBUG"          # File logs only
    INFO = "INFO"            # File logs only
    WARNING = "WARNING"      # Dashboard/monitoring only
    ERROR = "ERROR"          # Telegram/Slack
    CRITICAL = "CRITICAL"    # Telegram/Slack + PIN


class AlertChannel(Enum):
    """Alert delivery channels."""
    FILE_LOG = "FILE_LOG"
    DASHBOARD = "DASHBOARD"
    TELEGRAM = "TELEGRAM"
    SLACK = "SLACK"


# Severity → Channel routing
SEVERITY_ROUTING = {
    AlertSeverity.DEBUG: [AlertChannel.FILE_LOG],
    AlertSeverity.INFO: [AlertChannel.FILE_LOG],
    AlertSeverity.WARNING: [AlertChannel.FILE_LOG, AlertChannel.DASHBOARD],
    AlertSeverity.ERROR: [AlertChannel.FILE_LOG, AlertChannel.DASHBOARD, AlertChannel.TELEGRAM, AlertChannel.SLACK],
    AlertSeverity.CRITICAL: [AlertChannel.FILE_LOG, AlertChannel.DASHBOARD, AlertChannel.TELEGRAM, AlertChannel.SLACK]
}


class AlertDeduplicator:
    """
    Deduplication with cooldown windows.
    Prevents alert spam for the same condition.
    """
    
    def __init__(self, default_cooldown_minutes: int = 30):
        self.default_cooldown = timedelta(minutes=default_cooldown_minutes)
        self._alert_history: Dict[str, datetime] = {}
        self._alert_counts: Dict[str, int] = defaultdict(int)
    
    def _get_alert_key(self, category: str, message: str) -> str:
        """Generate unique alert key."""
        key_str = f"{category}:{message}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def should_send(self, category: str, message: str, cooldown: Optional[timedelta] = None) -> bool:
        """
        Check if alert should be sent based on cooldown.
        
        Returns:
            True if alert should be sent, False if in cooldown
        """
        alert_key = self._get_alert_key(category, message)
        now = datetime.now()
        cooldown_period = cooldown or self.default_cooldown
        
        if alert_key in self._alert_history:
            last_sent = self._alert_history[alert_key]
            if now - last_sent < cooldown_period:
                # Still in cooldown
                self._alert_counts[alert_key] += 1
                return False
        
        # Send alert and update history
        self._alert_history[alert_key] = now
        self._alert_counts[alert_key] = 1
        return True
    
    def get_suppressed_count(self, category: str, message: str) -> int:
        """Get count of suppressed alerts."""
        alert_key = self._get_alert_key(category, message)
        return self._alert_counts.get(alert_key, 0) - 1  # -1 because first one was sent


class InstitutionalAlertManager:
    """
    Institutional-grade alert system.
    
    Rules:
    - Silence = Healthy
    - Every alert must demand attention
    - Deduplication mandatory
    - Daily summaries instead of spam
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.deduplicator = AlertDeduplicator(default_cooldown_minutes=30)
        self.daily_summary_data = {
            "trades": 0,
            "pnl": 0.0,
            "nav": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "risk_state": "NORMAL"
        }
        self.last_daily_summary = None
        
        # Channel handlers (to be configured)
        self._channel_handlers: Dict[AlertChannel, Callable] = {}
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]):
        """Load alert configuration."""
        # Placeholder for config loading
        pass
    
    def register_channel_handler(self, channel: AlertChannel, handler: Callable):
        """Register a handler for a specific channel."""
        self._channel_handlers[channel] = handler
    
    def alert(self, 
              message: str,
              severity: AlertSeverity = AlertSeverity.INFO,
              category: str = "SYSTEM",
              metadata: Optional[Dict[str, Any]] = None,
              cooldown_minutes: Optional[int] = None,
              pin: bool = False):
        """
        Send alert with severity-based routing and deduplication.
        
        Args:
            message: Alert message
            severity: Alert severity level
            category: Alert category for deduplication
            metadata: Additional context
            cooldown_minutes: Custom cooldown (overrides default)
            pin: Pin message (CRITICAL only)
        """
        # Deduplication check
        cooldown = timedelta(minutes=cooldown_minutes) if cooldown_minutes else None
        if not self.deduplicator.should_send(category, message, cooldown):
            suppressed = self.deduplicator.get_suppressed_count(category, message)
            logger.debug(f"Alert suppressed (cooldown): {message} (suppressed {suppressed} times)")
            return
        
        # Get routing channels
        channels = SEVERITY_ROUTING.get(severity, [AlertChannel.FILE_LOG])
        
        # Prepare alert payload
        alert_payload = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity.value,
            "category": category,
            "message": message,
            "metadata": metadata or {},
            "pin": pin and severity == AlertSeverity.CRITICAL
        }
        
        # Route to channels
        for channel in channels:
            self._send_to_channel(channel, alert_payload)
    
    def _send_to_channel(self, channel: AlertChannel, payload: Dict[str, Any]):
        """Send alert to specific channel."""
        if channel == AlertChannel.FILE_LOG:
            # Always log to file
            severity = payload["severity"]
            message = payload["message"]
            log_level = getattr(logging, severity, logging.INFO)
            logger.log(log_level, f"[{payload['category']}] {message}")
        
        elif channel in self._channel_handlers:
            # Use registered handler
            try:
                self._channel_handlers[channel](payload)
            except Exception as e:
                logger.error(f"Channel handler failed for {channel.value}: {e}")
        else:
            # No handler configured
            logger.debug(f"No handler for channel {channel.value}")
    
    def update_daily_summary(self, **kwargs):
        """Update daily summary data."""
        self.daily_summary_data.update(kwargs)
    
    def send_daily_summary(self, force: bool = False):
        """
        Send daily summary (ONCE per day).
        
        Args:
            force: Force send even if already sent today
        """
        now = datetime.now()
        
        # Check if already sent today
        if not force and self.last_daily_summary:
            if self.last_daily_summary.date() == now.date():
                logger.debug("Daily summary already sent today")
                return
        
        # Format summary
        summary = f"""📊 DAILY FUND SUMMARY
NAV: ${self.daily_summary_data.get('nav', 0):,.2f}
PnL: {self.daily_summary_data.get('pnl', 0):+.2f}%
Sharpe: {self.daily_summary_data.get('sharpe', 0):.2f}
Max DD: {self.daily_summary_data.get('max_dd', 0):.2f}%
Trades: {self.daily_summary_data.get('trades', 0)}
Risk: {self.daily_summary_data.get('risk_state', 'NORMAL')}
Status: ACTIVE"""
        
        # Send as INFO (will go to file + dashboard, NOT Telegram)
        # For Telegram, use WARNING severity
        self.alert(
            summary,
            severity=AlertSeverity.WARNING,
            category="DAILY_SUMMARY",
            cooldown_minutes=1440  # 24 hours
        )
        
        self.last_daily_summary = now
    
    def risk_breach(self, breach_type: str, current: float, limit: float):
        """Alert for risk limit breach (ERROR)."""
        message = f"⚠️ RISK BREACH: {breach_type} {current:.2%} > {limit:.2%}"
        self.alert(
            message,
            severity=AlertSeverity.ERROR,
            category="RISK_BREACH",
            metadata={"type": breach_type, "current": current, "limit": limit},
            cooldown_minutes=60
        )
    
    def circuit_breaker(self, action: str, reason: str):
        """Alert for circuit breaker activation (CRITICAL)."""
        message = f"🚨 CIRCUIT BREAKER: {action} - {reason}"
        self.alert(
            message,
            severity=AlertSeverity.CRITICAL,
            category="CIRCUIT_BREAKER",
            metadata={"action": action, "reason": reason},
            pin=True,
            cooldown_minutes=120
        )
    
    def trade_execution_failure(self, ticker: str, reason: str):
        """Alert for trade execution failure (ERROR)."""
        message = f"❌ EXECUTION FAILED: {ticker} - {reason}"
        self.alert(
            message,
            severity=AlertSeverity.ERROR,
            category="EXECUTION_FAILURE",
            metadata={"ticker": ticker, "reason": reason},
            cooldown_minutes=30
        )
    
    def system_crash(self, component: str, error: str):
        """Alert for system crash (CRITICAL)."""
        message = f"💥 SYSTEM CRASH: {component} - {error}"
        self.alert(
            message,
            severity=AlertSeverity.CRITICAL,
            category="SYSTEM_CRASH",
            metadata={"component": component, "error": error},
            pin=True,
            cooldown_minutes=0  # Always send
        )
    
    def deployment_success(self, version: str):
        """One-time deployment success alert (INFO)."""
        message = f"✅ Deployment Complete - v{version}"
        self.alert(
            message,
            severity=AlertSeverity.INFO,
            category="DEPLOYMENT",
            cooldown_minutes=1440  # Once per day max
        )


# Global instance
_alert_manager = None

def get_alert_manager() -> InstitutionalAlertManager:
    """Get or create global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = InstitutionalAlertManager()
    return _alert_manager
