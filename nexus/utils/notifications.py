import logging
import os
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

class NotificationSystem:
    """Institutional alerting system for platform health and trade events."""
    
    def __init__(self):
        self.telegram_token = os.getenv("NEXUS_TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("NEXUS_TELEGRAM_CHAT_ID")
        self.discord_webhook = os.getenv("NEXUS_DISCORD_WEBHOOK")

    async def notify(self, message: str, level: str = "INFO"):
        """Broadcast alert to configured channels."""
        prefix = f"[{level}] NEXUS: "
        full_message = prefix + message
        
        logger.info(f"Notification Sent: {full_message}")
        
        # 1. Telegram
        if self.telegram_token and self.telegram_chat_id:
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    await client.post(url, json={
                        "chat_id": self.telegram_chat_id,
                        "text": full_message,
                        "parse_mode": "Markdown"
                    })
            except Exception as e:
                logger.warning(f"Telegram notification failed: {e}")

        # 2. Discord
        if self.discord_webhook:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(self.discord_webhook, json={
                        "content": full_message,
                        "username": "Nexus Terminal"
                    })
            except Exception as e:
                logger.warning(f"Discord notification failed: {e}")

# Global singleton
notifier = NotificationSystem()
