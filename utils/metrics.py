import time
from dataclasses import dataclass

@dataclass
class QuantMetrics:
    uptime_start: float = time.time()
    symbols_count: int = 0
    model_errors: int = 0
    arima_fallbacks: int = 0
    cycles: int = 0

    @property
    def uptime_sec(self) -> int:
        return int(time.time() - self.uptime_start)

# Global metrics instance
metrics = QuantMetrics()
