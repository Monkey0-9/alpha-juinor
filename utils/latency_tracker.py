import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class LatencyMeasurement:
    """Single latency measurement point"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

class InstitutionalLatencyTracker:
    """
    Top 1% Latency Tracking System.
    Provides nanosecond-precision tracking across the entire execution stack.
    """
    
    def __init__(self):
        self.measurements: List[LatencyMeasurement] = []
        self.active_spans: Dict[str, LatencyMeasurement] = {}
        
        # Performance thresholds (Institutional Standard)
        self.THRESHOLDS = {
            "signal_generation": 10.0,  # 10ms
            "risk_check": 5.0,          # 5ms
            "order_construction": 2.0,   # 2ms
            "broker_submission": 50.0,   # 50ms
            "total_execution": 100.0     # 100ms
        }
        
    def start_span(self, name: str, metadata: Optional[Dict] = None) -> str:
        """Start a new latency span"""
        span_id = f"{name}_{time.time_ns()}"
        measurement = LatencyMeasurement(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata or {}
        )
        self.active_spans[span_id] = measurement
        return span_id
        
    def end_span(self, span_id: str):
        """End a latency span and record it"""
        if span_id not in self.active_spans:
            logger.warning(f"Attempted to end unknown span: {span_id}")
            return
            
        measurement = self.active_spans.pop(span_id)
        measurement.end_time = time.perf_counter()
        self.measurements.append(measurement)
        
        # Check threshold
        if measurement.name in self.THRESHOLDS:
            threshold = self.THRESHOLDS[measurement.name]
            if measurement.duration_ms > threshold:
                logger.warning(
                    f"[LATENCY_ALERT] {measurement.name} exceeded threshold! "
                    f"Actual: {measurement.duration_ms:.4f}ms | Limit: {threshold}ms"
                )
                
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded latencies"""
        if not self.measurements:
            return {}
            
        summary = {}
        # Group by name
        by_name: Dict[str, List[float]] = {}
        for m in self.measurements:
            if m.name not in by_name:
                by_name[m.name] = []
            by_name[m.name].append(m.duration_ms)
            
        for name, durations in by_name.items():
            durations.sort()
            summary[name] = {
                "count": len(durations),
                "min_ms": durations[0],
                "max_ms": durations[-1],
                "mean_ms": sum(durations) / len(durations),
                "p50_ms": durations[len(durations)//2],
                "p95_ms": durations[int(len(durations)*0.95)],
                "p99_ms": durations[int(len(durations)*0.99)]
            }
            
        return summary

    def export_to_json(self, filepath: str):
        """Export all measurements to JSON for audit"""
        data = [
            {
                "name": m.name,
                "duration_ms": m.duration_ms,
                "timestamp": datetime.fromtimestamp(time.time()).isoformat(),
                "metadata": m.metadata
            }
            for m in self.measurements
        ]
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
# Global instance
tracker = InstitutionalLatencyTracker()
