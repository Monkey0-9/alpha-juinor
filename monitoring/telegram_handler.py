"""
Telegram Alert Handler

Sends ERROR and CRITICAL alerts to Telegram.
Implements rate limiting and message formatting.
"""
import os
import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class TelegramHandler:
    """
    Telegram alert handler with rate limiting.
    
    Only sends ERROR and CRITICAL alerts.
    Implements message formatting and PIN support.
    """
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            logger.warning("Telegram not configured (missing BOT_TOKEN or CHAT_ID)")
        
        # Rate limiting
        self._message_count = defaultdict(int)
        self._rate_limit_window = timedelta(hours=1)
        self._max_messages_per_hour = 20
        self._last_reset = datetime.now()
    
    def send(self, payload: Dict[str, Any]) -> bool:
        """
        Send alert to Telegram.
        
        Args:
            payload: Alert payload with severity, message, metadata
        
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        severity = payload.get("severity", "INFO")
        
        # Only send ERROR and CRITICAL
        if severity not in ["ERROR", "CRITICAL"]:
            logger.debug(f"Skipping Telegram for severity: {severity}")
            return False
        
        # Rate limiting check
        if not self._check_rate_limit():
            logger.warning("Telegram rate limit exceeded")
            return False
        
        # Format message
        message = self._format_message(payload)
        
        # Send to Telegram
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            
            # PIN if CRITICAL
            if payload.get("pin", False) and severity == "CRITICAL":
                self._pin_message(response.json()["result"]["message_id"])
            
            logger.info(f"Telegram alert sent: {severity}")
            return True
        
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    def _format_message(self, payload: Dict[str, Any]) -> str:
        """Format alert message for Telegram."""
        severity = payload.get("severity", "INFO")
        category = payload.get("category", "SYSTEM")
        message = payload.get("message", "")
        timestamp = payload.get("timestamp", datetime.now().isoformat())
        
        # Emoji mapping
        emoji_map = {
            "ERROR": "⚠️",
            "CRITICAL": "🚨"
        }
        emoji = emoji_map.get(severity, "ℹ️")
        
        # Format
        formatted = f"""<b>{emoji} {severity}</b>
<b>Category:</b> {category}
<b>Time:</b> {timestamp}

{message}"""
        
        return formatted
    
    def _pin_message(self, message_id: int):
        """PIN critical message."""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/pinChatMessage"
            data = {
                "chat_id": self.chat_id,
                "message_id": message_id,
                "disable_notification": False
            }
            requests.post(url, json=data, timeout=10)
        except Exception as e:
            logger.error(f"Failed to pin message: {e}")
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows sending."""
        now = datetime.now()
        
        # Reset counter if window expired
        if now - self._last_reset > self._rate_limit_window:
            self._message_count.clear()
            self._last_reset = now
        
        # Check limit
        current_count = sum(self._message_count.values())
        if current_count >= self._max_messages_per_hour:
            return False
        
        # Increment counter
        self._message_count[now.hour] += 1
        return True
