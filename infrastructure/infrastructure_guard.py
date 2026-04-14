"""
infrastructure/infrastructure_guard.py

InfrastructureGuard for mandatory pre-flight health checks.
Ensures all system dependencies are healthy before trading operations.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import aiohttp
import psutil
import pandas as pd

from monitoring.structured_logger import get_logger

logger = get_logger("infrastructure_guard")


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime
    summary: str


class InfrastructureGuard:
    """
    Comprehensive infrastructure health monitoring.
    Performs pre-flight checks before trading operations.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.last_health_check: Optional[SystemHealth] = None
        self.check_intervals = {
            "database": self.config.get("database_check_interval", 30),  # seconds
            "api": self.config.get("api_check_interval", 60),
            "system": self.config.get("system_check_interval", 120),
        }
        self.last_checks = {}

        # Health check thresholds
        self.thresholds = {
            "cpu_usage_max": self.config.get("cpu_usage_max", 80.0),
            "memory_usage_max": self.config.get("memory_usage_max", 85.0),
            "disk_usage_max": self.config.get("disk_usage_max", 90.0),
            "api_timeout_ms": self.config.get("api_timeout_ms", 5000),
            "db_connection_timeout_ms": self.config.get("db_connection_timeout_ms", 3000),
        }

    async def pre_flight_check(self, force: bool = False) -> SystemHealth:
        """
        Perform comprehensive pre-flight health check.
        Required before any trading operation.
        """
        self.logger.info("Starting pre-flight infrastructure health check")

        start_time = time.time()
        checks = []

        # Parallel health checks
        check_tasks = [
            self._check_database_health(),
            self._check_api_connectivity(),
            self._check_system_resources(),
            self._check_disk_space(),
            self._check_network_connectivity(),
            self._check_market_data_connectivity(),
            self._check_risk_system_health(),
        ]

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                checks.append(HealthCheck(
                    name="error",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(result)}",
                    response_time_ms=0,
                    timestamp=datetime.utcnow()
                ))
            else:
                checks.append(result)

        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        summary = self._generate_summary(checks, overall_status)

        system_health = SystemHealth(
            status=overall_status,
            checks=checks,
            timestamp=datetime.utcnow(),
            summary=summary
        )

        duration = (time.time() - start_time) * 1000
        self.logger.info(
            f"Pre-flight check completed in {duration:.2f}ms",
            overall_status=overall_status.value,
            total_checks=len(checks),
            healthy_checks=len([c for c in checks if c.status == HealthStatus.HEALTHY])
        )

        self.last_health_check = system_health
        return system_health

    def is_healthy_for_trading(self, force_check: bool = False) -> bool:
        """
        Quick check if system is healthy for trading.
        Returns True if all critical systems are healthy.
        """
        if force_check or not self.last_health_check:
            health = asyncio.run(self.pre_flight_check())
        else:
            health = self.last_health_check

        # Check if any critical systems are unhealthy
        critical_checks = ["database", "api", "market_data", "risk_system"]
        for check in health.checks:
            if check.name in critical_checks and check.status != HealthStatus.HEALTHY:
                self.logger.warning(
                    f"Critical system {check.name} is {check.status.value}: {check.message}",
                    check_name=check.name,
                    status=check.status.value
                )
                return False

        return health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    async def _check_database_health(self) -> HealthCheck:
        """Check database connectivity and performance."""
        start_time = time.time()

        try:
            # Test database connection
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
            from mini_quant_fund.database.manager import DatabaseManager
            db = DatabaseManager()

            # Simple query test
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT 1 as test")
                result = cursor.fetchone()

            response_time = (time.time() - start_time) * 1000

            if result and result[0] == 1:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    response_time_ms=response_time,
                    timestamp=datetime.utcnow(),
                    metadata={"query_test": "SELECT 1"}
                )
            else:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="Database query returned unexpected result",
                    response_time_ms=response_time,
                    timestamp=datetime.utcnow()
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.utcnow()
            )

    async def _check_api_connectivity(self) -> HealthCheck:
        """Check external API connectivity."""
        start_time = time.time()

        try:
            # Test Alpaca API connectivity
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("ALPACA_API_KEY")
            api_secret = os.getenv("ALPACA_SECRET_KEY")
            base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

            if not api_key or not api_secret:
                return HealthCheck(
                    name="api",
                    status=HealthStatus.DEGRADED,
                    message="Alpaca API credentials not configured",
                    response_time_ms=0,
                    timestamp=datetime.utcnow()
                )

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                headers = {
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": api_secret
                }
                async with session.get(f"{base_url}/account", headers=headers) as response:
                    response_time = (time.time() - start_time) * 1000

                    if response.status == 200:
                        return HealthCheck(
                            name="api",
                            status=HealthStatus.HEALTHY,
                            message="API connectivity successful",
                            response_time_ms=response_time,
                            timestamp=datetime.utcnow(),
                            metadata={"status_code": response.status}
                        )
                    else:
                        return HealthCheck(
                            name="api",
                            status=HealthStatus.DEGRADED,
                            message=f"API returned status {response.status}",
                            response_time_ms=response_time,
                            timestamp=datetime.utcnow(),
                            metadata={"status_code": response.status}
                        )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="api",
                status=HealthStatus.UNHEALTHY,
                message=f"API connectivity failed: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.utcnow()
            )

    async def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        start_time = time.time()

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            response_time = (time.time() - start_time) * 1000

            # Determine status based on thresholds
            if cpu_percent > self.thresholds["cpu_usage_max"] or memory.percent > self.thresholds["memory_usage_max"]:
                status = HealthStatus.DEGRADED
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resource usage normal: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"

            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_used_gb": memory.used / (1024**3)
                }
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.utcnow()
            )

    async def _check_disk_space(self) -> HealthCheck:
        """Check disk space availability."""
        start_time = time.time()

        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            response_time = (time.time() - start_time) * 1000

            if disk_percent > self.thresholds["disk_usage_max"]:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {disk_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space adequate: {disk_percent:.1f}% used"

            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                metadata={
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "disk_used_gb": disk.used / (1024**3),
                    "disk_total_gb": disk.total / (1024**3)
                }
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk space: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.utcnow()
            )

    async def _check_network_connectivity(self) -> HealthCheck:
        """Check basic network connectivity."""
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get("https://www.google.com", timeout=5) as response:
                    response_time = (time.time() - start_time) * 1000

                    if response.status == 200:
                        return HealthCheck(
                            name="network",
                            status=HealthStatus.HEALTHY,
                            message="Network connectivity successful",
                            response_time_ms=response_time,
                            timestamp=datetime.utcnow()
                        )
                    else:
                        return HealthCheck(
                            name="network",
                            status=HealthStatus.DEGRADED,
                            message=f"Network check returned status {response.status}",
                            response_time_ms=response_time,
                            timestamp=datetime.utcnow()
                        )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="network",
                status=HealthStatus.UNHEALTHY,
                message=f"Network connectivity failed: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.utcnow()
            )

    async def _check_market_data_connectivity(self) -> HealthCheck:
        """Check market data provider connectivity."""
        start_time = time.time()

        try:
            # Test Yahoo Finance connectivity
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
                async with session.get(url, timeout=5) as response:
                    response_time = (time.time() - start_time) * 1000

                    if response.status == 200:
                        return HealthCheck(
                            name="market_data",
                            status=HealthStatus.HEALTHY,
                            message="Market data connectivity successful",
                            response_time_ms=response_time,
                            timestamp=datetime.utcnow(),
                            metadata={"provider": "yahoo_finance"}
                        )
                    else:
                        return HealthCheck(
                            name="market_data",
                            status=HealthStatus.DEGRADED,
                            message=f"Market data returned status {response.status}",
                            response_time_ms=response_time,
                            timestamp=datetime.utcnow()
                        )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="market_data",
                status=HealthStatus.UNHEALTHY,
                message=f"Market data connectivity failed: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.utcnow()
            )

    async def _check_risk_system_health(self) -> HealthCheck:
        """Check risk management system health."""
        start_time = time.time()

        try:
            # Test risk manager initialization
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
            from mini_quant_fund.services.risk_enforcer import RiskEnforcer
            risk_manager = RiskEnforcer()

            # Test basic risk calculation
            import numpy as np
            test_weights = np.array([0.5, 0.5])
            test_returns = np.random.randn(1000, 2) * 0.01  # Mock returns

            # Test CVaR calculation directly
            cvar_result = risk_manager.portfolio_cvar(test_returns, test_weights)

            response_time = (time.time() - start_time) * 1000

            if isinstance(cvar_result, float):
                return HealthCheck(
                    name="risk_system",
                    status=HealthStatus.HEALTHY,
                    message="Risk system operational",
                    response_time_ms=response_time,
                    timestamp=datetime.utcnow(),
                    metadata={"cvar_result": cvar_result}
                )
            else:
                return HealthCheck(
                    name="risk_system",
                    status=HealthStatus.DEGRADED,
                    message="Risk system returned unexpected results",
                    response_time_ms=response_time,
                    timestamp=datetime.utcnow()
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="risk_system",
                status=HealthStatus.UNHEALTHY,
                message=f"Risk system check failed: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.utcnow()
            )

    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system health from individual checks."""
        if not checks:
            return HealthStatus.UNKNOWN

        # If any critical system is unhealthy, overall is unhealthy
        critical_unhealthy = any(
            check.status == HealthStatus.UNHEALTHY and check.name in ["database", "api", "risk_system"]
            for check in checks
        )
        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        # If any system is unhealthy, overall is degraded
        any_unhealthy = any(check.status == HealthStatus.UNHEALTHY for check in checks)
        if any_unhealthy:
            return HealthStatus.DEGRADED

        # If any system is degraded, overall is degraded
        any_degraded = any(check.status == HealthStatus.DEGRADED for check in checks)
        if any_degraded:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _generate_summary(self, checks: List[HealthCheck], overall_status: HealthStatus) -> str:
        """Generate human-readable health summary."""
        healthy = len([c for c in checks if c.status == HealthStatus.HEALTHY])
        degraded = len([c for c in checks if c.status == HealthStatus.DEGRADED])
        unhealthy = len([c for c in checks if c.status == HealthStatus.UNHEALTHY])
        unknown = len([c for c in checks if c.status == HealthStatus.UNKNOWN])

        summary = f"System {overall_status.value}: "
        summary += f"{healthy} healthy, {degraded} degraded, {unhealthy} unhealthy, {unknown} unknown"

        # Add details for unhealthy systems
        unhealthy_systems = [c.name for c in checks if c.status == HealthStatus.UNHEALTHY]
        if unhealthy_systems:
            summary += f". Issues: {', '.join(unhealthy_systems)}"

        return summary


# Global infrastructure guard instance
infrastructure_guard = InfrastructureGuard()

async def require_healthy_system():
    """Decorator or function to ensure system is healthy before operation."""
    if not infrastructure_guard.is_healthy_for_trading():
        health = await infrastructure_guard.pre_flight_check()
        raise RuntimeError(f"System not healthy for trading: {health.summary}")
    return True
