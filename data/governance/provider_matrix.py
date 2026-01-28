"""
Institutional Provider Matrix & Entitlement Router.

Responsibilities:
1. Define immutable capabilities of each provider.
2. Check partial entitlements (API keys present).
3. Select best provider for a symbol/asset class.
4. Enforce NO_VALID_PROVIDER on 403 (no retries).
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("PROVIDER_MATRIX")

# -----------------------------------------------------------------------------
# 1. Immutable Capabilities Definition
# -----------------------------------------------------------------------------
PROVIDER_CAPABILITIES = {
    "alpaca": {
        "stocks": True,
        "fx": False,
        "crypto": True,
        "commodities": False,
        "max_history_days": 1825,  # 5 years (approx) - Alpaca allows full history for paid, but verify plan
        "requires_entitlement": True,
        "priority": 30
    },
    "yahoo": {
        "stocks": True,
        "fx": True,
        "crypto": True,
        "commodities": True,
        "max_history_days": 10000, # Very long history available
        "requires_entitlement": False,
        "priority": 20
    },
    "polygon": {
        "stocks": True,
        "fx": True,
        "crypto": True,
        "commodities": False,
        "max_history_days": 5000,
        "requires_entitlement": True,
        "priority": 10 # Highest priority if available
    }
}

# -----------------------------------------------------------------------------
# 2. Entitlement Checks
# -----------------------------------------------------------------------------
def is_provider_entitled(provider: str) -> bool:
    """
    Check if the provider is enabled and has necessary credentials.
    STRICT CHECK: If env vars are missing, we return False.
    """
    if provider == "yahoo":
        return True

    if provider == "alpaca":
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        # Ensure they are not default placeholders
        if key and secret and "YOUR_" not in key:
            return True
        return False

    if provider == "polygon":
        key = os.getenv("POLYGON_API_KEY")
        if key and "YOUR_" not in key:
            return True
        return False

    return False

# -----------------------------------------------------------------------------
# 3. Symbol Classification
# -----------------------------------------------------------------------------
def classify_symbol(symbol: str) -> str:
    """
    Classify symbol into asset class for routing.
    """
    if symbol.endswith("=X"):
        return "fx"
    if symbol.endswith("=F"):
        return "commodities"
    if "-USD" in symbol or "BTC" in symbol or "ETH" in symbol:
        return "crypto"
    return "stocks"

# -----------------------------------------------------------------------------
# 4. Selection Logic
# -----------------------------------------------------------------------------
def select_provider(symbol: str, required_history_days: int) -> str:
    """
    Select the best provider for the symbol given constraints.
    Returns provider name or 'NO_VALID_PROVIDER'.

    Logic:
    1. Filter by asset class support.
    2. Filter by history depth capability.
    3. Filter by Entitlement (API keys).
    4. Sort by Priority (descending).
    """
    asset_class = classify_symbol(symbol)
    candidates = []

    for name, caps in PROVIDER_CAPABILITIES.items():
        # 1. Asset Class Support
        if not caps.get(asset_class, False):
            continue

        # 2. History Depth
        if caps["max_history_days"] < required_history_days:
            continue

        # 3. Entitlement Check
        if caps["requires_entitlement"] and not is_provider_entitled(name):
            continue

        candidates.append((name, caps["priority"]))

    if not candidates:
        logger.warning(f"NO_VALID_PROVIDER for {symbol} (Asset: {asset_class}, Days: {required_history_days})")
        return "NO_VALID_PROVIDER"

    # 4. Sort by priority (higher is better)
    # Using negative priority for ascending sort, or verify logic.
    # Let's sort descending by priority.
    candidates.sort(key=lambda x: x[1], reverse=True)

    selected = candidates[0][0]
    return selected
