from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ExecutionSlice:
    quantity: float
    target_time: datetime
    limit_price: Optional[float]
    status: str

class IcebergAlgorithm:
    """Iceberg order to hide large size from the book"""
    
    def __init__(self, display_pct: float = 0.1):
        self.display_pct = display_pct
        
    def generate_slices(self, total_qty: float) -> List[ExecutionSlice]:
        display_qty = total_qty * self.display_pct
        n_slices = int(1.0 / self.display_pct)
        
        slices = []
        now = datetime.utcnow()
        for i in range(n_slices):
            slices.append(ExecutionSlice(
                quantity=display_qty,
                target_time=now + timedelta(seconds=i * 10),
                limit_price=None,
                status="PENDING"
            ))
        return slices
