# monitoring/alerts.py
import os
import requests
import logging
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

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
        """Broadcast alert to all enabled channels."""
        msg = f"[{level}] {message}"
        logger.info(f"ALERT: {msg}")
        
        # Simple sync dispatch (good for main loop if not high freq)
        self.send_telegram(msg)
        self.send_slack(msg)
        self.send_discord(msg)

    def heartbeat(self):
        """Send a specialized heartbeat ping."""
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        msg = f"🟢 HEARTBEAT | Mem: {mem_mb:.1f}MB | Status: ACTIVE"
        self.alert(msg, level="HEARTBEAT")

    async def alert_async(self, message: str, level: str = "INFO"):
        """Async dispatch to prevent blocking trading loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.alert, message, level)
