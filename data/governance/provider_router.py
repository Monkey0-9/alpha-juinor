"""
data/governance/provider_router.py

Central routing logic for data provider selection.
Enforces entitlements, asset class restrictions, and history limits.
PREVENTS 403 loops by failing fast on forbidden requests.
"""

import re
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Provider Constants
PROVIDER_ALPACA = "alpaca"
PROVIDER_YAHOO = "yahoo"
PROVIDER_IBKR = "ibkr"

# Asset Classes
ASSET_STOCK = "STOCK"
ASSET_CRYPTO = "CRYPTO"
ASSET_FOREX = "FOREX"
ASSET_FUTURES = "FUTURES"

class ProviderCapabilityMatrix:
    """
    Defines what each provider CAN do.
    This configuration should eventually move to a config file/DB.
    """

    CAPABILITIES = {
        PROVIDER_ALPACA: {
            "supported_assets": {ASSET_STOCK, ASSET_CRYPTO},
            "max_history_days": 1825, # 5 years
            "requires_key": True,
            "cost": "low",
            "priority": 1
        },
        PROVIDER_YAHOO: {
            "supported_assets": {ASSET_STOCK, ASSET_CRYPTO, ASSET_FOREX, ASSET_FUTURES},
            "max_history_days": 36500, # Very long
            "requires_key": False,
            "cost": "free",
            "priority": 2 # Fallback
        },
        PROVIDER_IBKR: {
            "supported_assets": {ASSET_STOCK, ASSET_FUTURES, ASSET_FOREX},
            "max_history_days": 365,
            "requires_key": True,
            "cost": "high",
            "priority": 3
        }
    }

    # Entitlement overrides (e.g. if we know we don't have a paid subscription)
    ENTITLEMENTS = {
        PROVIDER_ALPACA: True, # Assuming we have keys, but check runtime
        PROVIDER_YAHOO: True,
        PROVIDER_IBKR: False   # Disabled by default until configured
    }

    @classmethod
    def classify_symbol(cls, symbol: str) -> str:
        """
        Heuristic classification of symbol to asset class.
        """
        # Yahoo/Forex notation
        if symbol.endswith("=X"):
            return ASSET_FOREX

        # Yahoo/Futures notation
        if symbol.endswith("=F"):
            return ASSET_FUTURES

        # Crypto notation (heuristic)
        if "-" in symbol and ("USD" in symbol or "BTC" in symbol or "ETH" in symbol):
            return ASSET_CRYPTO

        # Default to Stock
        return ASSET_STOCK

    @classmethod
    def get_eligible_providers(cls, symbol: str, history_days: int) -> List[str]:
        """
        Return list of providers capable of fulfilling the request, sorted by priority.
        """
        asset_class = cls.classify_symbol(symbol)
        candidates = []

        for provider, caps in cls.CAPABILITIES.items():
            # 1. Check Entitlement
            if not cls.ENTITLEMENTS.get(provider, False):
                continue

            # 2. Check Asset Class Support
            if asset_class not in caps["supported_assets"]:
                continue

            # 3. Check History Limits
            if history_days > caps["max_history_days"]:
                continue

            candidates.append(provider)

        # Sort by priority
        candidates.sort(key=lambda p: cls.CAPABILITIES[p]["priority"])

        return candidates

def select_provider(symbol: str, history_days: int = 100) -> Optional[str]:
    """
    Select the best provider for a given symbol and history requirement.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL', 'AUDUSD=X')
        history_days: Number of days of history needed

    Returns:
        Provider name (str) or None if no capable provider found.
    """
    eligible = ProviderCapabilityMatrix.get_eligible_providers(symbol, history_days)

    if not eligible:
        logger.warning(f"[PROVIDER_ROUTER] No eligible provider for {symbol} (days={history_days})")
        return None

    # Return top priority match
    selected = eligible[0]

    # Specific Logs for tricky routing to confirm fixes
    if symbol.endswith("=X") and selected == PROVIDER_ALPACA:
        logger.error(f"[ROUTING_ERROR] Forex {symbol} scheduled for Alpaca - Matrix config error!")
        return PROVIDER_YAHOO # Force fallback for safety

    return selected

def mark_provider_unavailable(provider: str, reason: str = "403_FORBIDDEN"):
    """
    Dynamically update entitlements if a provider fails hard (Circuit Breaker).
    """
    logger.critical(f"[CIRCUIT_BREAKER] Disabling {provider} due to: {reason}")
    ProviderCapabilityMatrix.ENTITLEMENTS[provider] = False
