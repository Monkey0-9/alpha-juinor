
import logging
from typing import Dict, Any, Optional
import math
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ProviderQuotaManager:
    """
    Manages API quotas for data providers.
    Enforces Monthly Limits and Daily Caps.
    Daily Cap = ceil(Monthly Limit / 30).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        config: dictionary containing 'providers' key with quota details.
        Example:
        {
            "providers": {
                "polygon": {"monthly_limit": 50000},
                "finnhub": {"monthly_limit": 60000}
            }
        }
        """
        self.config = config.get("providers", {})
        self.usage = {} # {provider_date: count}
        self.daily_caps = {}

        # Initialize Daily Caps
        for provider, details in self.config.items():
            monthly = details.get("monthly_limit", 100000)
            self.daily_caps[provider] = math.ceil(monthly / 30)
            logger.info(f"QuotaManager: {provider} Monthly={monthly}, Daily Cap={self.daily_caps[provider]}")

    def check_quota(self, provider: str) -> bool:
        """
        Returns True if request is allowed, False if quota exceeded.
        """
        if provider not in self.daily_caps:
            return True # No quota defined, allow

        today = datetime.now().strftime("%Y-%m-%d")
        key = f"{provider}:{today}"

        current_usage = self.usage.get(key, 0)
        limit = self.daily_caps.get(provider, 1000)

        if current_usage >= limit:
            logger.warning(f"Quota Exceeded for {provider}. Usage: {current_usage}/{limit}")
            return False

        return True

    def increment_usage(self, provider: str, cost: int = 1):
        """
        Track usage.
        """
        if provider not in self.daily_caps:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        key = f"{provider}:{today}"
        self.usage[key] = self.usage.get(key, 0) + cost

    def get_status(self) -> Dict[str, Any]:
        return {
            "usage": self.usage,
            "caps": self.daily_caps
        }
