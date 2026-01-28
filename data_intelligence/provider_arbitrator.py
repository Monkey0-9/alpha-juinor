"""
data_intelligence/provider_arbitrator.py

Provider Arbitration Layer (Ticket 4)

Multi-Armed Bandit (MAB) based provider selection with Bayesian updates.
Automatically routes to best provider per symbol with failover.
"""

import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("PROVIDER_ARBITRATOR")


class ProviderStatus(str, Enum):
    """Provider health status."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


@dataclass
class ProviderConfig:
    """Configuration for a data provider."""
    name: str
    priority: int = 0  # Lower = higher priority
    enabled: bool = True
    rate_limit_per_min: int = 100
    timeout_seconds: float = 30.0
    retry_count: int = 3
    cost_per_call: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority,
            "enabled": self.enabled,
            "rate_limit_per_min": self.rate_limit_per_min,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "cost_per_call": self.cost_per_call
        }


@dataclass
class ProviderScore:
    """MAB score for a provider."""
    provider: str
    alpha: float = 1.0  # Beta distribution α (successes + 1)
    beta: float = 1.0   # Beta distribution β (failures + 1)
    ucb_score: float = 0.5  # Upper Confidence Bound
    last_updated: str = ""

    @property
    def mean_reward(self) -> float:
        """Expected reward (success rate)."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Thompson Sampling: draw from Beta distribution."""
        return random.betavariate(self.alpha, self.beta)


class ProviderArbitrator:
    """
    Multi-Armed Bandit Provider Selection.

    Uses Thompson Sampling for exploration-exploitation trade-off.
    Automatically switches providers on failure and learns optimal routing.

    Features:
    - Per-symbol provider routing
    - Automatic failover on errors
    - Bayesian score updates
    - UCB for exploration bonus
    - Cost-aware selection
    """

    # UCB exploration parameter
    UCB_C = 2.0  # Higher = more exploration

    # Minimum calls before exploitation
    MIN_EXPLORATION_CALLS = 10

    def __init__(self, providers: List[ProviderConfig] = None, db_manager=None):
        """
        Initialize ProviderArbitrator.

        Args:
            providers: List of configured providers
            db_manager: DatabaseManager instance
        """
        self._db = db_manager

        # Default providers
        self._providers: Dict[str, ProviderConfig] = {}
        if providers:
            for p in providers:
                self._providers[p.name] = p
        else:
            # Default provider configs
            self._providers = {
                "alpaca": ProviderConfig(name="alpaca", priority=1, rate_limit_per_min=200),
                "polygon": ProviderConfig(name="polygon", priority=2, rate_limit_per_min=100),
                "yfinance": ProviderConfig(name="yfinance", priority=3, rate_limit_per_min=60),
            }

        # Per-symbol provider scores
        self._scores: Dict[Tuple[str, str], ProviderScore] = {}  # (symbol, provider) -> score

        # Global provider health
        self._provider_health: Dict[str, ProviderStatus] = {}

        # Call counts for UCB
        self._total_calls = 0
        self._provider_calls: Dict[str, int] = {}

    @property
    def db(self):
        if self._db is None:
            from database.manager import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    def get_provider(
        self,
        symbol: str,
        exclude: List[str] = None
    ) -> Optional[str]:
        """
        Select best provider for a symbol using Thompson Sampling.

        Args:
            symbol: Stock symbol
            exclude: Providers to exclude (e.g., after failure)

        Returns:
            Provider name, or None if all disabled
        """
        exclude = exclude or []

        # Get available providers
        available = [
            p for p in self._providers.values()
            if p.enabled and p.name not in exclude and
            self._provider_health.get(p.name, ProviderStatus.HEALTHY) != ProviderStatus.DISABLED
        ]

        if not available:
            logger.warning(f"No available providers for {symbol}")
            return None

        # Exploration phase: try each provider at least MIN_EXPLORATION_CALLS times
        for p in available:
            calls = self._provider_calls.get(p.name, 0)
            if calls < self.MIN_EXPLORATION_CALLS:
                logger.debug(f"Exploration: selecting {p.name} for {symbol} (calls={calls})")
                return p.name

        # Thompson Sampling: sample from Beta distribution for each provider
        best_provider = None
        best_sample = -1.0

        for p in available:
            score = self._get_score(symbol, p.name)

            # Sample from posterior
            sample = score.sample()

            # Adjust for cost (lower cost = higher effective score)
            cost_factor = 1.0 / (1.0 + p.cost_per_call)
            adjusted_sample = sample * cost_factor

            if adjusted_sample > best_sample:
                best_sample = adjusted_sample
                best_provider = p.name

        logger.debug(f"Selected {best_provider} for {symbol} (sample={best_sample:.4f})")
        return best_provider

    def get_provider_with_fallback(
        self,
        symbol: str
    ) -> List[str]:
        """
        Get ordered list of providers to try (primary + fallbacks).

        Args:
            symbol: Stock symbol

        Returns:
            Ordered list of provider names
        """
        providers = []
        excluded = []

        # Get up to 3 providers in order of preference
        for _ in range(3):
            p = self.get_provider(symbol, exclude=excluded)
            if p:
                providers.append(p)
                excluded.append(p)
            else:
                break

        return providers

    def record_success(
        self,
        symbol: str,
        provider: str,
        latency_ms: Optional[float] = None
    ):
        """
        Record successful fetch.

        Args:
            symbol: Stock symbol
            provider: Provider that succeeded
            latency_ms: Fetch latency
        """
        score = self._get_score(symbol, provider)
        score.alpha += 1
        score.last_updated = datetime.utcnow().isoformat() + 'Z'
        self._update_ucb(score)
        self._scores[(symbol, provider)] = score

        # Update call counts
        self._total_calls += 1
        self._provider_calls[provider] = self._provider_calls.get(provider, 0) + 1

        # Clear failed status
        if self._provider_health.get(provider) == ProviderStatus.FAILED:
            self._provider_health[provider] = ProviderStatus.HEALTHY

        # Log choice
        logger.info(json.dumps({
            "event": "PROVIDER_SUCCESS",
            "symbol": symbol,
            "provider": provider,
            "latency_ms": latency_ms,
            "new_alpha": score.alpha,
            "mean_reward": round(score.mean_reward, 4)
        }))

        # Update data confidence service
        try:
            from data_intelligence.data_confidence import get_data_confidence_service
            confidence_svc = get_data_confidence_service()
            confidence_svc.record_success(symbol, provider, latency_ms)
        except Exception as e:
            logger.debug(f"Failed to update confidence: {e}")

    def record_failure(
        self,
        symbol: str,
        provider: str,
        error: str
    ):
        """
        Record failed fetch.

        Args:
            symbol: Stock symbol
            provider: Provider that failed
            error: Error message
        """
        score = self._get_score(symbol, provider)
        score.beta += 1
        score.last_updated = datetime.utcnow().isoformat() + 'Z'
        self._update_ucb(score)
        self._scores[(symbol, provider)] = score

        # Update call counts
        self._total_calls += 1
        self._provider_calls[provider] = self._provider_calls.get(provider, 0) + 1

        # Check if provider should be marked as failed
        if score.mean_reward < 0.3:
            self._provider_health[provider] = ProviderStatus.DEGRADED
        if score.mean_reward < 0.1:
            self._provider_health[provider] = ProviderStatus.FAILED
            logger.warning(json.dumps({
                "event": "PROVIDER_MARKED_FAILED",
                "provider": provider,
                "mean_reward": round(score.mean_reward, 4)
            }))

        # Log failure
        logger.warning(json.dumps({
            "event": "PROVIDER_FAILURE",
            "symbol": symbol,
            "provider": provider,
            "error": error[:200],
            "new_beta": score.beta,
            "mean_reward": round(score.mean_reward, 4)
        }))

        # Update data confidence service
        try:
            from data_intelligence.data_confidence import get_data_confidence_service
            confidence_svc = get_data_confidence_service()
            confidence_svc.record_failure(symbol, provider, error)
        except Exception as e:
            logger.debug(f"Failed to update confidence: {e}")

    def _get_score(self, symbol: str, provider: str) -> ProviderScore:
        """Get or create score for symbol-provider pair."""
        key = (symbol, provider)
        if key not in self._scores:
            # Initialize with slight prior based on provider priority
            priority = self._providers.get(provider, ProviderConfig(name=provider)).priority
            # Higher priority = higher initial alpha
            initial_alpha = max(1.0, 5.0 - priority)
            self._scores[key] = ProviderScore(
                provider=provider,
                alpha=initial_alpha,
                beta=1.0,
                last_updated=datetime.utcnow().isoformat() + 'Z'
            )
        return self._scores[key]

    def _update_ucb(self, score: ProviderScore):
        """Update UCB score."""
        n_provider = self._provider_calls.get(score.provider, 1)
        if n_provider > 0 and self._total_calls > 0:
            import math
            exploration_bonus = self.UCB_C * math.sqrt(math.log(self._total_calls) / n_provider)
            score.ucb_score = score.mean_reward + exploration_bonus
        else:
            score.ucb_score = score.mean_reward + self.UCB_C

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        stats = {}
        for name, config in self._providers.items():
            calls = self._provider_calls.get(name, 0)

            # Aggregate scores across symbols
            alphas = []
            betas = []
            for (sym, prov), score in self._scores.items():
                if prov == name:
                    alphas.append(score.alpha)
                    betas.append(score.beta)

            if alphas:
                total_alpha = sum(alphas)
                total_beta = sum(betas)
                mean_reward = total_alpha / (total_alpha + total_beta)
            else:
                mean_reward = 0.5

            stats[name] = {
                "enabled": config.enabled,
                "priority": config.priority,
                "status": self._provider_health.get(name, ProviderStatus.HEALTHY).value,
                "total_calls": calls,
                "mean_reward": round(mean_reward, 4),
                "symbols_tracked": len([1 for (s, p) in self._scores if p == name])
            }

        return stats

    def disable_provider(self, provider: str, reason: str):
        """Manually disable a provider."""
        if provider in self._providers:
            self._providers[provider].enabled = False
        self._provider_health[provider] = ProviderStatus.DISABLED
        logger.warning(json.dumps({
            "event": "PROVIDER_DISABLED",
            "provider": provider,
            "reason": reason
        }))

    def enable_provider(self, provider: str):
        """Re-enable a disabled provider."""
        if provider in self._providers:
            self._providers[provider].enabled = True
        self._provider_health[provider] = ProviderStatus.HEALTHY
        logger.info(json.dumps({
            "event": "PROVIDER_ENABLED",
            "provider": provider
        }))


# Singleton instance
_instance: Optional[ProviderArbitrator] = None


def get_provider_arbitrator() -> ProviderArbitrator:
    """Get singleton ProviderArbitrator instance."""
    global _instance
    if _instance is None:
        _instance = ProviderArbitrator()
    return _instance
