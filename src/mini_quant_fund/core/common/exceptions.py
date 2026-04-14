
class QuantFundError(Exception):
    """Base exception for all Quant Fund errors."""
    pass

class EngineError(QuantFundError):
    """Errors related to the trading engine."""
    pass

class ConfigurationError(QuantFundError):
    """Errors related to system configuration."""
    pass

class SecurityError(QuantFundError):
    """Errors related to security and authentication."""
    pass

class DataError(QuantFundError):
    """Errors related to market data acquisition or processing."""
    pass

class ExecutionError(QuantFundError):
    """Errors related to trade execution."""
    pass

class RiskError(QuantFundError):
    """Errors related to risk management violations."""
    pass
