# monitoring/alert_dispatcher.py
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

class AlertDispatcher:
    """
    Institutional Alert Routing System.
    Disseminates critical operational info to Console, Log, and external webhooks.
    """
    
    def __init__(self, telegram_token: Optional[str] = None, slack_url: Optional[str] = None):
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.slack_url = slack_url or os.getenv("SLACK_WEBHOOK_URL")

    def dispatch(self, message: str, level: str = "INFO"):
        """
        Routes the alert to appropriate channels based on level.
        levels: INFO, WARNING, CRITICAL, EMERGENCY
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{level}] {timestamp} - {message}"
        
        # 1. Console & Local Log
        if level in ["CRITICAL", "EMERGENCY"]:
            logger.error(formatted_msg)
        elif level == "WARNING":
            logger.warning(formatted_msg)
        else:
            logger.info(formatted_msg)
            
        # 2. External Hooks (Stubs)
        if self.telegram_token:
            self._send_telegram(formatted_msg)
            
        if self.slack_url:
            self._send_slack(formatted_msg)

    def _send_telegram(self, msg: str):
        # Stub for requests.post(...)
        # logger.debug(f"Routing to Telegram: {msg}")
        pass

    def _send_slack(self, msg: str):
        # Stub for requests.post(...)
        # logger.debug(f"Routing to Slack: {msg}")
        pass

# Global convenience instance
dispatcher = AlertDispatcher()

def alert(message: str, level: str = "INFO"):
    dispatcher.dispatch(message, level)
