"""
Provider Health Manager & Circuit Breaker.
Tracks provider success rate, latency, and implements cooldown/circuit breaking.
Prevents provider exhaustion and enforces 1 attempt per provider per symbol per cycle.
"""

import logging
import time
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ProviderHealth:
    """Health metrics for a single provider"""
    name: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_latency_ms: float = 0.0
    last_failure_time: float = 0.0
    circuit_open: bool = False
    cooldown_until: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 1.0  # Optimistic prior
        return self.successful_attempts / self.total_attempts

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_attempts == 0:
            return 0.0
        return self.total_latency_ms / self.successful_attempts

    def is_available(self) -> bool:
        """Check if provider is available (not in cooldown)"""
        if self.circuit_open:
            if time.time() >= self.cooldown_until:
                # Cooldown expired, close circuit
                self.circuit_open = False
                logger.info(f"Provider '{self.name}' circuit closed after cooldown")
                return True
            return False
        return True


class ProviderCircuitBreaker:
    """
    Circuit breaker for data providers.
    Enforces:
    - 1 attempt per provider per symbol per cycle
    - Cooldown after consecutive failures
    - Circuit open/close based on success rate
    """

    def __init__(
        self,
        failure_threshold: float = 0.5,  # Open circuit if success_rate < 50%
        min_attempts: int = 5,  # Minimum attempts before opening circuit
        cooldown_seconds: float = 300.0,  # 5 minute cooldown
        consecutive_failures_limit: int = 3  # Open circuit after 3 consecutive failures
    ):
        self.failure_threshold = failure_threshold
        self.min_attempts = min_attempts
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_failures_limit = consecutive_failures_limit

        self.providers: Dict[str, ProviderHealth] = {}
        self.consecutive_failures: Dict[str, int] = defaultdict(int)

        # Track attempts per symbol per cycle to enforce 1 attempt limit
        self.cycle_attempts: Dict[str, Set[str]] = defaultdict(set)  # {symbol: {provider1, provider2}}
        self.current_cycle_id: Optional[str] = None

    def register_provider(self, provider_name: str):
        """Register a provider for health tracking"""
        if provider_name not in self.providers:
            self.providers[provider_name] = ProviderHealth(name=provider_name)
            logger.info(f"Registered provider '{provider_name}' in circuit breaker")

    def start_cycle(self, cycle_id: str):
        """Start a new cycle - reset per-symbol attempt tracking"""
        self.current_cycle_id = cycle_id
        self.cycle_attempts.clear()
        logger.debug(f"Circuit breaker: Started cycle {cycle_id}")

    def can_attempt(self, provider_name: str, symbol: str) -> tuple[bool, str]:
        """
        Check if provider can be attempted for this symbol in this cycle.

        Returns:
            (can_attempt, reason)
        """
        # Ensure provider is registered
        if provider_name not in self.providers:
            self.register_provider(provider_name)

        provider = self.providers[provider_name]

        # Check if circuit is open
        if not provider.is_available():
            return False, f"CIRCUIT_OPEN_COOLDOWN_UNTIL_{provider.cooldown_until:.0f}"

        # Check if already attempted this provider for this symbol in this cycle
        if provider_name in self.cycle_attempts.get(symbol, set()):
            return False, f"ALREADY_ATTEMPTED_THIS_CYCLE"

        return True, "OK"

    def record_attempt(
        self,
        provider_name: str,
        symbol: str,
        success: bool,
        latency_ms: float = 0.0
    ):
        """
        Record a provider attempt.
        Updates health metrics and opens circuit if needed.
        """
        if provider_name not in self.providers:
            self.register_provider(provider_name)

        provider = self.providers[provider_name]

        # Record attempt
        provider.total_attempts += 1
        self.cycle_attempts[symbol].add(provider_name)

        if success:
            provider.successful_attempts += 1
            provider.total_latency_ms += latency_ms
            self.consecutive_failures[provider_name] = 0  # Reset consecutive failures
        else:
            provider.failed_attempts += 1
            provider.last_failure_time = time.time()
            self.consecutive_failures[provider_name] += 1

            # Check if we should open the circuit
            if self.consecutive_failures[provider_name] >= self.consecutive_failures_limit:
                self._open_circuit(provider_name, reason="CONSECUTIVE_FAILURES")
            elif (provider.total_attempts >= self.min_attempts and
                  provider.success_rate < self.failure_threshold):
                self._open_circuit(provider_name, reason="LOW_SUCCESS_RATE")

        logger.debug(f"Provider '{provider_name}' attempt for {symbol}: "
                    f"success={success}, success_rate={provider.success_rate:.2f}, "
                    f"consecutive_failures={self.consecutive_failures[provider_name]}")

    def _open_circuit(self, provider_name: str, reason: str):
        """Open circuit for a provider"""
        provider = self.providers[provider_name]
        provider.circuit_open = True
        provider.cooldown_until = time.time() + self.cooldown_seconds

        logger.warning(f"⚠️  Circuit OPENED for provider '{provider_name}' - {reason}. "
                      f"Cooldown until {provider.cooldown_until:.0f} "
                      f"(success_rate={provider.success_rate:.2f}, "
                      f"consecutive_failures={self.consecutive_failures[provider_name]})")

    def get_health_report(self) -> Dict[str, Dict]:
        """Get health report for all providers"""
        return {
            name: {
                "success_rate": health.success_rate,
                "total_attempts": health.total_attempts,
                "avg_latency_ms": health.avg_latency_ms,
                "circuit_open": health.circuit_open,
                "available": health.is_available()
            }
            for name, health in self.providers.items()
        }
