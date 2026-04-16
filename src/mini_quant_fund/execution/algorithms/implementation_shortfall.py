
import numpy as np
import math
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExecutionSlice:
    quantity: float
    target_time: datetime
    limit_price: Optional[float]
    status: str
    expected_impact_bps: float = 0.0

class ImplementationShortfallAlgorithm:
    """
    Implementation Shortfall (Almgren-Chriss) Optimal Execution Trajectory.
    
    This algorithm minimizes the sum of market impact and price risk (variance).
    """
    
    def __init__(self, risk_aversion: float = 0.1):
        self.risk_aversion = risk_aversion
        self.temp_impact_coef = 0.1  # Temporary impact coefficient (eta)
        self.perm_impact_coef = 0.05 # Permanent impact coefficient (gamma)
        
    def execute(self, 
                symbol: str, 
                quantity: float, 
                side: str, 
                volatility: float, 
                adv: float, 
                n_intervals: int = 10,
                interval_minutes: int = 5) -> List[ExecutionSlice]:
        """
        Calculates the optimal trading trajectory.
        """
        logger.info(f"Calculating IS trajectory for {symbol}: Q={quantity}, Vol={volatility}, ADV={adv}")
        
        # Almgren-Chriss Optimal Lambda (kappa)
        # kappa = sqrt(lambda * sigma^2 / eta)
        # where lambda is risk aversion, sigma is volatility, eta is temp impact
        
        # Normalize volatility and ADV to interval scale
        sigma = volatility * math.sqrt(interval_minutes / (252 * 6.5 * 60))
        eta = self.temp_impact_coef * (sigma / (0.01 * adv)) # Heuristic scaling
        
        kappa = math.sqrt((self.risk_aversion * (sigma**2)) / eta) if eta > 0 else 0.5
        
        # Optimal trajectory: n_j = (sinh(kappa * (T - t_j)) / sinh(kappa * T)) * X
        # where n_j is remaining quantity
        
        slices = []
        now = datetime.utcnow()
        remaining = quantity
        
        for j in range(1, n_intervals + 1):
            t_remaining = n_intervals - j + 1
            
            if kappa > 0:
                # Calculate what should be remaining AFTER this interval
                target_remaining = (math.sinh(kappa * (t_remaining - 1)) / math.sinh(kappa * n_intervals)) * quantity
                slice_qty = remaining - target_remaining
            else:
                slice_qty = quantity / n_intervals
                
            slice_qty = max(0, min(slice_qty, remaining))
            
            # Estimate impact for this slice
            impact_bps = (eta * (slice_qty / adv) * 10000)
            
            slices.append(ExecutionSlice(
                quantity=float(slice_qty),
                target_time=now + timedelta(minutes=(j-1) * interval_minutes),
                limit_price=None,
                status="PENDING",
                expected_impact_bps=impact_bps
            ))
            
            remaining -= slice_qty
            if remaining <= 0:
                break
                
        return slices

    def set_parameters(self, risk_aversion: float):
        self.risk_aversion = risk_aversion
        logger.info(f"IS Algorithm risk aversion set to {risk_aversion}")
