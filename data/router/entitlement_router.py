"""
data/router/entitlement_router.py

Strict Entitlement & Capabilities Router.
Enforces:
1. Asset Class Support
2. History Depth Limits
3. Active Entitlements (API Keys)
4. Runtime Block Lists (403/400 handling)
"""

import os
import yaml
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime

# Assuming SecretManager exists from previous work
from config.secrets_manager import secrets

logger = logging.getLogger("ROUTER")

# Load Configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "provider_capabilities.yaml")

class EntitlementRouter:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EntitlementRouter, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.capabilities = self._load_config()
        self.provider_priority = ["yahoo", "polygon", "alpaca"] # Default priority
        self.blocked_providers = {} # {provider: {symbol: reason}}
        self.blocked_file = "runtime/blocked_providers.json"

        # Ensure runtime dir
        os.makedirs("runtime", exist_ok=True)
        self._load_blocked()

    def _load_config(self) -> Dict:
        if not os.path.exists(CONFIG_PATH):
            logger.error(f"Provider config missing at {CONFIG_PATH}")
            return {}
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)

    def _load_blocked(self):
        if os.path.exists(self.blocked_file):
            try:
                with open(self.blocked_file, 'r') as f:
                    self.blocked_providers = json.load(f)
            except Exception:
                self.blocked_providers = {}

    def _persist_blocked(self):
        with open(self.blocked_file, 'w') as f:
            json.dump(self.blocked_providers, f, indent=2)

    def classify_symbol(self, symbol: str) -> str:
        if symbol.endswith("=X"): return "fx"
        if symbol.endswith("=F"): return "commodities"
        if "-USD" in symbol or "USD-" in symbol: return "crypto"
        return "stocks"

    def is_entitled(self, provider: str) -> bool:
        """Check if we have credentials for this provider."""
        # Config structure: providers -> name -> details
        # The YAML loads as {"providers": {"yahoo": ...}} or just {"yahoo": ...}?
        # Let's handle both.
        specs = self.capabilities.get("providers", self.capabilities)
        spec = specs.get(provider)
        if not spec: return False

        if not spec.get("requires_entitlement", False):
            return True

        # Check active secrets
        keys = {
            "alpaca": "ALPACA_API_KEY",
            "polygon": "POLYGON_API_KEY"
        }
        secret_key = keys.get(provider)
        if secret_key:
            return bool(secrets.get_secret(secret_key))
        return False

    def is_blocked(self, provider: str, symbol: str) -> bool:
        if provider in self.blocked_providers:
            if symbol in self.blocked_providers[provider]:
                return True
        return False

    def block_provider(self, provider: str, symbol: str, reason: str):
        """Permanently block provider for this symbol (runtime session)."""
        if provider not in self.blocked_providers:
            self.blocked_providers[provider] = {}

        self.blocked_providers[provider][symbol] = {
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._persist_blocked()
        logger.warning(f"[ENTITLEMENT_BLOCK] Blocked {provider} for {symbol}: {reason}")

    def select_provider(self, symbol: str, required_history_days: int) -> Dict[str, str]:
        """
        Select best provider.
        """
        asset_class = self.classify_symbol(symbol)

        specs = self.capabilities.get("providers", self.capabilities)

        for provider in self.provider_priority:
            spec = specs.get(provider)
            if not spec: continue

            # 1. Asset Class Support (Boolean check)
            # Checks 'top level' keys like 'stocks', 'fx' in the spec
            if not spec.get(asset_class, False):
                continue

            # 2. History Depth
            if required_history_days > spec.get("max_history_days", 0):
                continue

            # 3. Entitlement
            if not self.is_entitled(provider):
                continue

            # 4. Blocked?
            if self.is_blocked(provider, symbol):
                continue

            return {"provider": provider, "reason": "OK"}

        return {"provider": "NONE", "reason": f"No provider for {asset_class} with {required_history_days} days history"}

# Global Instance
router = EntitlementRouter()
