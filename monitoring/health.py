import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class HealthMonitor:
    """
    Institutional-grade health monitoring system.
    Tracks provider health, anomalies, and implements circuit breaker logic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_config = config.get('monitoring', {})

        # Circuit breaker settings
        self.circuit_breaker_threshold = self.monitoring_config.get('circuit_breaker_threshold', 5)
        self.anomaly_window_minutes = self.monitoring_config.get('anomaly_window_minutes', 10)
        self.health_check_interval = self.monitoring_config.get('health_check_interval_seconds', 300)

        # State tracking
        self.provider_health: Dict[str, Dict[str, Any]] = {}
        self.anomaly_history: list = []
        self.suppressed_exceptions: Dict[str, int] = defaultdict(int)
        self.last_health_check = time.time()
        self.circuit_breaker_tripped = False

        # Initialize provider health tracking
        self._initialize_provider_tracking()

    def _initialize_provider_tracking(self):
        """Initialize health tracking for all known providers."""
        providers = ['yahoo', 'polygon', 'binance', 'coingecko', 'bloomberg', 'reuters']
        for provider in providers:
            self.provider_health[provider] = {
                'last_successful_request': None,
                'total_requests': 0,
                'successful_requests': 0,
                'last_error': None,
                'error_count': 0,
                'is_healthy': True
            }

    def record_provider_request(self, provider_name: str, success: bool, error: Optional[str] = None):
        """Record a provider request outcome."""
        if provider_name not in self.provider_health:
            self.provider_health[provider_name] = {
                'last_successful_request': None,
                'total_requests': 0,
                'successful_requests': 0,
                'last_error': None,
                'error_count': 0,
                'is_healthy': True
            }

        health = self.provider_health[provider_name]
        health['total_requests'] += 1

        if success:
            health['successful_requests'] += 1
            health['last_successful_request'] = datetime.now()
            health['is_healthy'] = True
        else:
            health['error_count'] += 1
            health['last_error'] = error
            # Mark unhealthy after 3 consecutive failures
            if health['error_count'] >= 3:
                health['is_healthy'] = False

    def record_suppressed_exception(self, exception_type: str):
        """Record a suppressed exception for monitoring."""
        self.suppressed_exceptions[exception_type] += 1

    def record_anomaly(self, anomaly_type: str, details: str):
        """Record an anomaly detection."""
        anomaly = {
            'timestamp': datetime.now(),
            'type': anomaly_type,
            'details': details
        }
        self.anomaly_history.append(anomaly)

        # Clean old anomalies outside the window
        cutoff = datetime.now() - timedelta(minutes=self.anomaly_window_minutes)
        self.anomaly_history = [a for a in self.anomaly_history if a['timestamp'] > cutoff]

        # Check circuit breaker
        recent_anomalies = len(self.anomaly_history)
        if recent_anomalies >= self.circuit_breaker_threshold and not self.circuit_breaker_tripped:
            self._trip_circuit_breaker()

    def _trip_circuit_breaker(self):
        """Trip the circuit breaker and escalate."""
        self.circuit_breaker_tripped = True
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {len(self.anomaly_history)} anomalies in {self.anomaly_window_minutes} minutes")
        logger.critical("TRADING LOOP PAUSED - MANUAL INTERVENTION REQUIRED")

        # Escalation actions
        self._send_escalation_alerts()
        self._pause_trading()
        self._create_diagnostic_report()

    def _send_escalation_alerts(self):
        """Send multi-channel escalation alerts."""
        try:
            # Create alert message
            alert_msg = f"""
            🚨 CIRCUIT BREAKER TRIGGERED 🚨

            Timestamp: {datetime.now().isoformat()}
            Anomalies: {len(self.anomaly_history)} in {self.anomaly_window_minutes} minutes
            Threshold: {self.circuit_breaker_threshold}

            Recent Anomalies:
            {self._format_recent_anomalies()}

            ACTION REQUIRED: Manual intervention needed
            """

            # Log to alert file
            with open("runtime/CIRCUIT_BREAKER_ALERT.log", "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(alert_msg)
                f.write(f"\n{'='*80}\n")

            # TODO: In production, add:
            # - Email notification (via SMTP or SendGrid)
            # - SMS notification (via Twilio)
            # - PagerDuty/OpsGenie integration
            # - Slack/Discord webhook

            logger.info("Escalation alerts sent successfully")

        except Exception as e:
            logger.error(f"Failed to send escalation alerts: {e}")

    def _pause_trading(self):
        """Create kill switch file to pause trading."""
        try:
            import os
            os.makedirs("runtime", exist_ok=True)

            with open("runtime/KILL_SWITCH", "w") as f:
                f.write(f"Circuit breaker triggered at {datetime.now().isoformat()}\n")
                f.write(f"Anomalies: {len(self.anomaly_history)}\n")
                f.write("Manual reset required\n")

            logger.critical("KILL_SWITCH file created - trading paused")

        except Exception as e:
            logger.error(f"Failed to create KILL_SWITCH: {e}")

    def _create_diagnostic_report(self):
        """Create diagnostic report for troubleshooting."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'circuit_breaker_tripped': self.circuit_breaker_tripped,
                'anomalies': [
                    {
                        'time': a['timestamp'].isoformat(),
                        'type': a['type'],
                        'details': a['details']
                    } for a in self.anomaly_history
                ],
                'provider_health': self.provider_health,
                'suppressed_exceptions': dict(self.suppressed_exceptions)
            }

            import json
            with open("runtime/circuit_breaker_diagnostic.json", "w") as f:
                json.dump(report, f, indent=2)

            logger.info("Diagnostic report created: runtime/circuit_breaker_diagnostic.json")

        except Exception as e:
            logger.error(f"Failed to create diagnostic report: {e}")

    def _format_recent_anomalies(self) -> str:
        """Format recent anomalies for alert message."""
        if not self.anomaly_history:
            return "None"

        formatted = []
        for anomaly in self.anomaly_history[-10:]:  # Last 10
            formatted.append(
                f"  - {anomaly['timestamp'].strftime('%H:%M:%S')}: "
                f"{anomaly['type']} - {anomaly['details']}"
            )
        return "\n".join(formatted)

    def reset_circuit_breaker(self):
        """Reset the circuit breaker (manual intervention)."""
        self.circuit_breaker_tripped = False
        self.anomaly_history.clear()
        logger.warning("CIRCUIT BREAKER RESET - MONITORING CLOSELY")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        now = datetime.now()

        # Provider health summary
        provider_summary = {}
        for name, health in self.provider_health.items():
            success_rate = (health['successful_requests'] / health['total_requests']) if health['total_requests'] > 0 else 0
            last_success_age = None
            if health['last_successful_request']:
                last_success_age = (now - health['last_successful_request']).total_seconds()

            provider_summary[name] = {
                'healthy': health['is_healthy'],
                'success_rate': success_rate,
                'last_success_seconds': last_success_age,
                'error_count': health['error_count'],
                'last_error': health['last_error']
            }

        # Anomaly summary
        recent_anomalies = len(self.anomaly_history)

        return {
            'circuit_breaker_tripped': self.circuit_breaker_tripped,
            'recent_anomalies': recent_anomalies,
            'anomaly_threshold': self.circuit_breaker_threshold,
            'suppressed_exceptions': dict(self.suppressed_exceptions),
            'provider_health': provider_summary,
            'last_health_check': self.last_health_check
        }

    def perform_health_check(self):
        """Perform periodic health check."""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return

        self.last_health_check = current_time
        status = self.get_health_status()

        # Log health summary
        unhealthy_providers = [name for name, health in status['provider_health'].items() if not health['healthy']]
        if unhealthy_providers:
            logger.warning(f"UNHEALTHY PROVIDERS: {unhealthy_providers}")

        if status['circuit_breaker_tripped']:
            logger.critical("CIRCUIT BREAKER ACTIVE - SYSTEM IN SAFE MODE")
        elif status['recent_anomalies'] > 0:
            logger.info(f"Recent anomalies: {status['recent_anomalies']}")

        logger.debug(f"Health check completed. Suppressed exceptions: {sum(status['suppressed_exceptions'].values())}")

    def is_system_healthy(self) -> bool:
        """Check if system is healthy for trading."""
        if self.circuit_breaker_tripped:
            return False

        # Check if we have at least one healthy provider
        healthy_providers = [h for h in self.provider_health.values() if h['is_healthy']]
        return len(healthy_providers) > 0
