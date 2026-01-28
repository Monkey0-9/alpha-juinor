"""
Multi-Armed Bandit for Data Provider Selection.

Uses UCB1 (Upper Confidence Bound) algorithm to balance exploration/exploitation.
Integrates with database for metrics persistence and confidence tracking.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)

from database.manager import get_db


@dataclass
class ProviderStats:
    """Statistics for a single provider"""
    name: str
    pulls: int = 0
    successes: int = 0
    total_latency_ms: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    confidence: float = 0.5  # From ConfidenceAgent
    cost: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.pulls == 0:
            return 0.5  # Neutral prior
        return self.successes / self.pulls

    @property
    def avg_latency(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_latency_ms / self.pulls

    @property
    def avg_quality(self) -> float:
        if not self.quality_scores:
            return 0.5
        return sum(self.quality_scores) / len(self.quality_scores)


class ProviderBandit:
    """
    Multi-Armed Bandit for selecting data providers.

    Uses UCB1 algorithm with quality and latency penalties.
    Persists metrics to database for recovery and analysis.
    """

    def __init__(self, providers: List[str], exploration_factor: float = 2.0,
                 db=None):
        """
        Args:
            providers: List of provider names
            exploration_factor: UCB1 exploration parameter (default 2.0)
            db: Optional DatabaseManager instance
        """
        self.providers = {name: ProviderStats(name=name) for name in providers}
        self.exploration_factor = exploration_factor
        self.total_pulls = 0
        self.db = db

        # Load historical metrics from database
        if self.db:
            self._load_metrics()

    def _load_metrics(self):
        """Load provider metrics from database"""
        try:
            metrics = self.db.get_provider_metrics(days=90)
            for row in metrics:
                name = row.get('provider_name')
                if name in self.providers:
                    stats = self.providers[name]
                    stats.pulls = row.get('pulls', 0)
                    stats.successes = row.get('successes', 0)
                    stats.total_latency_ms = row.get('avg_latency_ms', 0) * stats.pulls
                    self.total_pulls += stats.pulls
            logger.info("Loaded provider metrics from database")
        except Exception as e:
            logger.warning(f"Could not load provider metrics: {e}")

    def select_provider(self, available_providers: Optional[List[str]] = None) -> str:
        """
        Select best provider using UCB1 algorithm.

        Args:
            available_providers: Subset of providers to consider

        Returns:
            Provider name
        """
        if available_providers is None:
            available_providers = list(self.providers.keys())

        if not available_providers:
            raise ValueError("No providers available")

        # Filter to available providers
        candidates = {name: stats for name, stats in self.providers.items()
                     if name in available_providers}

        if not candidates:
            raise ValueError(f"None of the available providers are registered: {available_providers}")

        # If any provider has never been tried, try it first (exploration)
        for name, stats in candidates.items():
            if stats.pulls == 0:
                logger.info(f"ProviderBandit: Exploring untried provider '{name}'")
                return name

        # UCB1: Select provider with highest upper confidence bound
        best_provider = None
        best_score = -float('inf')

        for name, stats in candidates.items():
            # Base score: success rate weighted by confidence and quality
            base_score = stats.success_rate * stats.confidence * stats.avg_quality

            # Exploration bonus: sqrt(2 * ln(total_pulls) / pulls)
            if stats.pulls > 0 and self.total_pulls > 0:
                exploration_bonus = self.exploration_factor * math.sqrt(
                    math.log(self.total_pulls) / stats.pulls
                )
            else:
                exploration_bonus = 0.0

            # Latency penalty (prefer faster providers)
            latency_penalty = 0.0
            if stats.avg_latency > 0:
                # Normalize latency: 1000ms = 0.1 penalty, 5000ms = 0.5 penalty
                latency_penalty = min(0.5, stats.avg_latency / 10000.0)

            ucb_score = base_score + exploration_bonus - latency_penalty

            logger.debug(f"ProviderBandit: {name} UCB={ucb_score:.3f} "
                        f"(base={base_score:.3f}, explore={exploration_bonus:.3f}, "
                        f"latency_penalty={latency_penalty:.3f})")

            if ucb_score > best_score:
                best_score = ucb_score
                best_provider = name

        logger.info(f"ProviderBandit: Selected '{best_provider}' (UCB={best_score:.3f})")
        return best_provider

    def update(self, provider: str, success: bool, latency_ms: float = 0.0,
               quality_score: float = 1.0, confidence: float = None,
               cost: float = 0.0):
        """
        Update provider statistics after a fetch attempt.

        Args:
            provider: Provider name
            success: Whether fetch was successful
            latency_ms: Fetch latency in milliseconds
            quality_score: Data quality score (0.0 to 1.0)
            confidence: Provider confidence from ConfidenceAgent (optional)
            cost: API cost for this request
        """
        if provider not in self.providers:
            logger.warning(f"ProviderBandit: Unknown provider '{provider}', adding it")
            self.providers[provider] = ProviderStats(name=provider)

        stats = self.providers[provider]
        stats.pulls += 1
        if success:
            stats.successes += 1
        stats.total_latency_ms += latency_ms
        stats.quality_scores.append(quality_score)
        stats.cost += cost

        # Keep only last 100 quality scores to avoid memory bloat
        if len(stats.quality_scores) > 100:
            stats.quality_scores.pop(0)

        if confidence is not None:
            stats.confidence = confidence

        self.total_pulls += 1

        logger.debug(f"ProviderBandit: Updated '{provider}' - "
                    f"pulls={stats.pulls}, success_rate={stats.success_rate:.2f}, "
                    f"avg_latency={stats.avg_latency:.0f}ms, "
                    f"avg_quality={stats.avg_quality:.2f}, confidence={stats.confidence:.2f}")

        # Persist to database
        self._persist_metrics(provider)

    def _persist_metrics(self, provider: str):
        """Persist provider metrics to database"""
        if not self.db:
            return

        try:
            stats = self.providers[provider]
            from database.schema import ProviderMetricsRecord
            record = ProviderMetricsRecord(
                provider_name=provider,
                date=time.strftime('%Y-%m-%d'),
                pulls=stats.pulls,
                successes=stats.successes,
                avg_latency_ms=stats.avg_latency,
                avg_quality_score=stats.avg_quality,
                cost=stats.cost
            )
            self.db.update_provider_metrics(record)
        except Exception as e:
            logger.warning(f"Could not persist provider metrics: {e}")

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all providers"""
        return {
            name: {
                "pulls": stats.pulls,
                "success_rate": stats.success_rate,
                "avg_latency_ms": stats.avg_latency,
                "avg_quality": stats.avg_quality,
                "confidence": stats.confidence,
                "cost": stats.cost
            }
            for name, stats in self.providers.items()
        }

    def get_best_provider(self) -> Tuple[str, float]:
        """Get the best provider based on overall score"""
        best_name = None
        best_score = -float('inf')

        for name, stats in self.providers.items():
            score = stats.success_rate * stats.avg_quality * stats.confidence
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, best_score

    def get_success_rates(self) -> Dict[str, float]:
        """Get success rates for all providers"""
        return {
            name: stats.success_rate
            for name, stats in self.providers.items()
        }


def get_bandit() -> ProviderBandit:
    """Get configured provider bandit instance"""
    db = get_db()
    return ProviderBandit(
        providers=['polygon', 'alpha_vantage', 'stooq', 'yahoo'],
        exploration_factor=2.0,
        db=db
    )

