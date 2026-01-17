# monitoring/alerts.py
import os
import requests
import logging
import asyncio
from typing import Optional, Dict, Any
from time import time
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels for institutional monitoring."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertCategory(Enum):
    """Alert categories for better organization."""
    RISK = "RISK"
    PERFORMANCE = "PERFORMANCE"
    SYSTEM = "SYSTEM"
    MARKET_DATA = "MARKET_DATA"
    TRADING = "TRADING"

@dataclass
class AlertMetrics:
    """Metrics tracking for institutional monitoring."""
    total_alerts: int = 0
    alerts_by_severity: Dict[str, int] = field(default_factory=dict)
    alerts_by_category: Dict[str, int] = field(default_factory=dict)
    alerts_last_24h: int = 0
    critical_alerts_last_24h: int = 0
    average_response_time: float = 0.0
    system_uptime: float = 0.0

@dataclass
class Alert:
    """Structured alert with institutional metadata."""
    message: str
    severity: AlertSeverity
    category: AlertCategory
    timestamp: float = field(default_factory=time)
    source: str = "quant-fund"
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None

class AlertManager:
    """
    Institutional Alerting System.
    Supports Telegram, Slack, and Discord.
    """
    
    def __init__(self):
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

        # Deduplication tracking
        self._last_sent = {}  # key -> timestamp
        self._dedup_window = int(os.getenv("ALERT_DEDUP_WINDOW_SECONDS", 600))  # 10 minutes default
        self._notify_level = os.getenv("MONITOR_NOTIFY_LEVEL", "INFO").upper()
        self._heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", 3600))  # 1 hour default
        self._last_heartbeat = 0
        self._last_status = None  # Track status changes
        
    def validate_config(self) -> bool:
        """Check if at least one alert channel is configured."""
        configured = any([
            self.telegram_bot_token and self.telegram_chat_id,
            self.slack_webhook_url,
            self.discord_webhook_url
        ])
        if not configured:
            logger.warning("No external alerting channels (Telegram/Slack/Discord) configured.")
        return configured
        
    def send_telegram(self, message: str):
        """Send message via Telegram."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
            
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": f"🚨 [QUANT-FUND] 🚨\n{message}",
            "parse_mode": "Markdown"
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")

    def send_slack(self, message: str):
        """Send message via Slack Webhook."""
        if not self.slack_webhook_url:
            return
            
        payload = {"text": f"*[QUANT-FUND]*: {message}"}
        try:
            requests.post(self.slack_webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")

    def send_discord(self, message: str):
        """Send message via Discord Webhook."""
        if not self.discord_webhook_url:
            return
            
        payload = {"content": f"**[QUANT-FUND]**: {message}"}
        try:
            requests.post(self.discord_webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Discord alert failed: {e}")

    def alert(self, message: str, level: str = "INFO"):
        """Broadcast alert to all enabled channels with deduplication."""
        # Check notification level
        level_upper = level.upper()
        if level_upper not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            level_upper = "INFO"

        notify_levels = {"ERROR": 3, "WARNING": 2, "INFO": 1, "DEBUG": 0}
        current_level_value = notify_levels.get(level_upper, 1)
        required_level_value = notify_levels.get(self._notify_level, 1)

        if current_level_value < required_level_value:
            return  # Don't send alerts below the configured level

        # Deduplication check
        key = f"{level}:{message}"
        now = time()
        last_sent = self._last_sent.get(key, 0)
        if (now - last_sent) < self._dedup_window:
            logger.debug(f"Deduplicating alert: {key}")
            return

        self._last_sent[key] = now

        msg = f"[{level}] {message}"
        logger.info(f"ALERT: {msg}")

        # Simple sync dispatch (good for main loop if not high freq)
        self.send_telegram(msg)
        self.send_slack(msg)
        self.send_discord(msg)

    def heartbeat(self, diagnosis: Optional[str] = None, force: bool = False):
        """Send a specialized heartbeat ping with resource usage - only on state changes or intervals."""
        now = time()

        # Check if enough time has passed since last heartbeat
        if not force and (now - self._last_heartbeat) < self._heartbeat_interval:
            return

        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            current_status = "ACTIVE"
            msg = f"🟢 HEARTBEAT | Mem: {mem_mb:.1f}MB | Status: {current_status}"
            if diagnosis:
                msg += f"\n\n{diagnosis}"
        except ImportError:
            current_status = "ACTIVE"
            msg = "🟢 HEARTBEAT | Mem: [psutil missing] | Status: ACTIVE"
        except Exception as e:
            current_status = "ERROR"
            msg = f"🟢 HEARTBEAT | Error: {e} | Status: {current_status}"

        # Only send if status changed or forced
        notify_on_change_only = os.getenv("NOTIFY_ON_STATE_CHANGE_ONLY", "false").lower() == "true"
        if notify_on_change_only and not force:
            if self._last_status == current_status:
                return
            self._last_status = current_status

        self._last_heartbeat = now
        self.alert(msg, level="HEARTBEAT")

    async def alert_async(self, message: str, level: str = "INFO"):
        """Async dispatch to prevent blocking trading loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.alert, message, level)

# Global singleton for system-wide access
alert_manager = AlertManager()

def alert(message: str, level: str = "INFO"):
    """Convenience function for system-wide alerting."""
    alert_manager.alert(message, level)
