"""
Core Exception Classes for Institutional Trading System

Provides domain-specific exception hierarchy for proper error handling
and debugging in production environments.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }


class ConfigurationError(TradingSystemError):
    """Raised when system configuration is invalid."""
    pass


class DatabaseError(TradingSystemError):
    """Raised when database operations fail."""
    pass


class APIConnectionError(TradingSystemError):
    """Raised when external API connections fail."""
    pass


class MarketDataError(TradingSystemError):
    """Raised when market data is invalid or unavailable."""
    pass


class ExecutionError(TradingSystemError):
    """Raised when order execution fails."""
    pass


class RiskLimitError(TradingSystemError):
    """Raised when risk limits are breached."""
    pass


class PortfolioError(TradingSystemError):
    """Raised when portfolio operations fail."""
    pass


class ValidationError(TradingSystemError):
    """Raised when input validation fails."""
    pass


class InsufficientCapitalError(TradingSystemError):
    """Raised when insufficient capital for trading."""
    pass


class OrderSizeError(TradingSystemError):
    """Raised when order size violates limits."""
    pass


class MarketClosedError(TradingSystemError):
    """Raised when attempting to trade in closed market."""
    pass


class RateLimitError(TradingSystemError):
    """Raised when API rate limits are exceeded."""
    pass


class AuthenticationError(TradingSystemError):
    """Raised when API authentication fails."""
    pass


class TimeoutError(TradingSystemError):
    """Raised when operations timeout."""
    pass


class CircuitBreakerError(TradingSystemError):
    """Raised when circuit breaker is open."""
    pass


class DataQualityError(TradingSystemError):
    """Raised when data quality checks fail."""
    pass


class ComplianceError(TradingSystemError):
    """Raised when compliance rules are violated."""
    pass


class PositionLimitError(TradingSystemError):
    """Raised when position limits are exceeded."""
    pass


class LeverageLimitError(TradingSystemError):
    """Raised when leverage limits are exceeded."""
    pass


class LiquidityError(TradingSystemError):
    """Raised when insufficient market liquidity."""
    pass


class SlippageExceededError(TradingSystemError):
    """Raised when slippage exceeds acceptable limits."""
    pass


class LatencyError(TradingSystemError):
    """Raised when operation latency exceeds thresholds."""
    pass


class SystemHealthError(TradingSystemError):
    """Raised when system health checks fail."""
    pass


class ConcurrencyError(TradingSystemError):
    """Raised when concurrency issues occur."""
    pass


class ResourceExhaustionError(TradingSystemError):
    """Raised when system resources are exhausted."""
    pass


class SerializationError(TradingSystemError):
    """Raised when data serialization fails."""
    pass


class DeserializationError(TradingSystemError):
    """Raised when data deserialization fails."""
    pass


class CacheError(TradingSystemError):
    """Raised when cache operations fail."""
    pass


class MessageQueueError(TradingSystemError):
    """Raised when message queue operations fail."""
    pass


class ServiceUnavailableError(TradingSystemError):
    """Raised when required services are unavailable."""
    pass


class DataCorruptionError(TradingSystemError):
    """Raised when data corruption is detected."""
    pass


class BackupError(TradingSystemError):
    """Raised when backup operations fail."""
    pass


class RecoveryError(TradingSystemError):
    """Raised when recovery operations fail."""
    pass


class MonitoringError(TradingSystemError):
    """Raised when monitoring operations fail."""
    pass


class AlertingError(TradingSystemError):
    """Raised when alerting operations fail."""
    pass


class SecurityError(TradingSystemError):
    """Raised when security violations are detected."""
    pass


class AuditError(TradingSystemError):
    """Raised when audit operations fail."""
    pass


class ReportingError(TradingSystemError):
    """Raised when reporting operations fail."""
    pass


class AnalyticsError(TradingSystemError):
    """Raised when analytics operations fail."""
    pass


class MachineLearningError(TradingSystemError):
    """Raised when ML operations fail."""
    pass


class ModelError(TradingSystemError):
    """Raised when model operations fail."""
    pass


class FeatureEngineeringError(TradingSystemError):
    """Raised when feature engineering fails."""
    pass


class BacktestError(TradingSystemError):
    """Raised when backtest operations fail."""
    pass


class SimulationError(TradingSystemError):
    """Raised when simulation operations fail."""
    pass


class OptimizationError(TradingSystemError):
    """Raised when optimization operations fail."""
    pass


class CalibrationError(TradingSystemError):
    """Raised when calibration operations fail."""
    pass


class ValidationError(TradingSystemError):
    """Raised when validation operations fail."""
    pass


class SanityCheckError(TradingSystemError):
    """Raised when sanity checks fail."""
    pass


class IntegrationError(TradingSystemError):
    """Raised when integration operations fail."""
    pass


class DeploymentError(TradingSystemError):
    """Raised when deployment operations fail."""
    pass


class MaintenanceError(TradingSystemError):
    """Raised when maintenance operations fail."""
    pass


class UpgradeError(TradingSystemError):
    """Raised when upgrade operations fail."""
    pass


class MigrationError(TradingSystemError):
    """Raised when migration operations fail."""
    pass


class RollbackError(TradingSystemError):
    """Raised when rollback operations fail."""
    pass


class PerformanceError(TradingSystemError):
    """Raised when performance thresholds are exceeded."""
    pass


class ScalabilityError(TradingSystemError):
    """Raised when scalability limits are reached."""
    pass


class CapacityError(TradingSystemError):
    """Raised when capacity limits are exceeded."""
    pass


class ThroughputError(TradingSystemError):
    """Raised when throughput limits are exceeded."""
    pass


class AvailabilityError(TradingSystemError):
    """Raised when availability requirements are not met."""
    pass


class ReliabilityError(TradingSystemError):
    """Raised when reliability requirements are not met."""
    pass


class ConsistencyError(TradingSystemError):
    """Raised when data consistency issues are detected."""
    pass


class DurabilityError(TradingSystemError):
    """Raised when data durability issues are detected."""
    pass


class PartitionError(TradingSystemError):
    """Raised when partition issues occur."""
    pass


class ReplicationError(TradingSystemError):
    """Raised when replication issues occur."""
    pass


class SynchronizationError(TradingSystemError):
    """Raised when synchronization issues occur."""
    pass


class DeadlockError(TradingSystemError):
    """Raised when deadlock situations occur."""
    pass


class RaceConditionError(TradingSystemError):
    """Raised when race conditions are detected."""
    pass


class MemoryLeakError(TradingSystemError):
    """Raised when memory leaks are detected."""
    pass


class ResourceLeakError(TradingSystemError):
    """Raised when resource leaks are detected."""
    pass


class ConnectionPoolExhaustionError(TradingSystemError):
    """Raised when connection pools are exhausted."""
    pass


class ThreadPoolExhaustionError(TradingSystemError):
    """Raised when thread pools are exhausted."""
    pass


class ProcessLimitError(TradingSystemError):
    """Raised when process limits are exceeded."""
    pass


class FileHandleExhaustionError(TradingSystemError):
    """Raised when file handle limits are exceeded."""
    pass


class NetworkError(TradingSystemError):
    """Raised when network issues occur."""
    pass


class DNSError(TradingSystemError):
    """Raised when DNS resolution fails."""
    pass


class TCPIPError(TradingSystemError):
    """Raised when TCP/IP errors occur."""
    pass


class HTTPError(TradingSystemError):
    """Raised when HTTP errors occur."""
    pass


class WebSocketError(TradingSystemError):
    """Raised when WebSocket errors occur."""
    pass


class ProtocolError(TradingSystemError):
    """Raised when protocol errors occur."""
    pass


class FormatError(TradingSystemError):
    """Raised when format errors occur."""
    pass


class EncodingError(TradingSystemError):
    """Raised when encoding errors occur."""
    pass


class DecodingError(TradingSystemError):
    """Raised when decoding errors occur."""
    pass


class CompressionError(TradingSystemError):
    """Raised when compression errors occur."""
    pass


class DecompressionError(TradingSystemError):
    """Raised when decompression errors occur."""
    pass


class EncryptionError(TradingSystemError):
    """Raised when encryption errors occur."""
    pass


class DecryptionError(TradingSystemError):
    """Raised when decryption errors occur."""
    pass


class HashError(TradingSystemError):
    """Raised when hash operations fail."""
    pass


class SignatureError(TradingSystemError):
    """Raised when signature verification fails."""
    pass


class CertificateError(TradingSystemError):
    """Raised when certificate errors occur."""
    pass


class TLSError(TradingSystemError):
    """Raised when TLS errors occur."""
    pass


class SSLError(TradingSystemError):
    """Raised when SSL errors occur."""
    pass


class AuthenticationTokenError(TradingSystemError):
    """Raised when authentication token errors occur."""
    pass


class AuthorizationError(TradingSystemError):
    """Raised when authorization errors occur."""
    pass


class PermissionError(TradingSystemError):
    """Raised when permission errors occur."""
    pass


class AccessDeniedError(TradingSystemError):
    """Raised when access is denied."""
    pass


class ForbiddenError(TradingSystemError):
    """Raised when operations are forbidden."""
    pass


class NotFoundError(TradingSystemError):
    """Raised when resources are not found."""
    pass


class ConflictError(TradingSystemError):
    """Raised when conflicts occur."""
    pass


class PreconditionFailedError(TradingSystemError):
    """Raised when preconditions fail."""
    pass


class PayloadTooLargeError(TradingSystemError):
    """Raised when payload is too large."""
    pass


class UnsupportedMediaTypeError(TradingSystemError):
    """Raised when media type is unsupported."""
    pass


class UnsupportedMethodError(TradingSystemError):
    """Raised when method is unsupported."""
    pass


class UnsupportedVersionError(TradingSystemError):
    """Raised when version is unsupported."""
    pass


class DeprecatedError(TradingSystemError):
    """Raised when deprecated functionality is used."""
    pass


class ObsoleteError(TradingSystemError):
    """Raised when obsolete functionality is used."""
    pass


class ExperimentalError(TradingSystemError):
    """Raised when experimental features fail."""
    pass


class BetaFeatureError(TradingSystemError):
    """Raised when beta features fail."""
    pass


class AlphaFeatureError(TradingSystemError):
    """Raised when alpha features fail."""
    pass


class CustomFeatureError(TradingSystemError):
    """Raised when custom features fail."""
    pass


class ThirdPartyError(TradingSystemError):
    """Raised when third-party integrations fail."""
    pass


class VendorError(TradingSystemError):
    """Raised when vendor-specific errors occur."""
    pass


class PlatformError(TradingSystemError):
    """Raised when platform-specific errors occur."""
    pass


class OperatingSystemError(TradingSystemError):
    """Raised when operating system errors occur."""
    pass


class HardwareError(TradingSystemError):
    """Raised when hardware errors occur."""
    pass


class FirmwareError(TradingSystemError):
    """Raised when firmware errors occur."""
    pass


class DriverError(TradingSystemError):
    """Raised when driver errors occur."""
    pass


class LibraryError(TradingSystemError):
    """Raised when library errors occur."""
    pass


class DependencyError(TradingSystemError):
    """Raised when dependency errors occur."""
    pass


class CompatibilityError(TradingSystemError):
    """Raised when compatibility issues occur."""
    pass


class VersionMismatchError(TradingSystemError):
    """Raised when version mismatches occur."""
    pass


class ConfigurationMismatchError(TradingSystemError):
    """Raised when configuration mismatches occur."""
    pass


class EnvironmentMismatchError(TradingSystemError):
    """Raised when environment mismatches occur."""
    pass


class BuildError(TradingSystemError):
    """Raised when build errors occur."""
    pass


class CompileError(TradingSystemError):
    """Raised when compilation errors occur."""
    pass


class LinkError(TradingSystemError):
    """Raised when linking errors occur."""
    pass


class PackageError(TradingSystemError):
    """Raised when packaging errors occur."""
    pass


class DistributionError(TradingSystemError):
    """Raised when distribution errors occur."""
    pass


class InstallationError(TradingSystemError):
    """Raised when installation errors occur."""
    pass


class UninstallationError(TradingSystemError):
    """Raised when uninstallation errors occur."""
    pass


class UpdateError(TradingSystemError):
    """Raised when update errors occur."""
    pass


class PatchError(TradingSystemError):
    """Raised when patch errors occur."""
    pass


class HotfixError(TradingSystemError):
    """Raised when hotfix errors occur."""
    pass


class EmergencyError(TradingSystemError):
    """Raised when emergency situations occur."""
    pass


class CriticalError(TradingSystemError):
    """Raised when critical errors occur."""
    pass


class FatalError(TradingSystemError):
    """Raised when fatal errors occur."""
    pass


class CatastrophicError(TradingSystemError):
    """Raised when catastrophic errors occur."""
    pass
