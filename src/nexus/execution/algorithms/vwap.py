import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from mini_quant_fund.data.collectors.data_router import DataRouter
from mini_quant_fund.execution.market_impact import MarketImpactModel

logger = logging.getLogger(__name__)

@dataclass
class ExecutionSlice:
    quantity: float
    target_time: datetime
    limit_price: Optional[float]
    status: str
    estimated_impact_bps: float = 0.0

class VWAPAlgorithm:
    """Institutional Volume-Weighted Average Price with Market Impact Modeling"""
    
    def __init__(self, data_router: Optional[DataRouter] = None):
        self.router = data_router or DataRouter()
        self.impact_model = MarketImpactModel()
        # Typical daily volume distribution (U-shape)
        self.default_profile = np.array([0.15, 0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3])
        self.default_profile /= self.default_profile.sum()
        
    def execute(self, 
                symbol: str, 
                quantity: float, 
                side: str, 
                duration_hours: float = 6.5,
                risk_aversion: float = 1e-5) -> List[ExecutionSlice]:
        """
        Generates execution slices based on historical volume profile.
        Incorporates Almgren-Chriss market impact modeling.
        """
        # 1. Get ADV and Volatility for impact modeling
        try:
            hist = self.router.get_price_history(symbol, 
                                               (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d'),
                                               allow_long_history=True)
            adv = hist['Volume'].mean() if not hist.empty else 1000000
            volatility = hist['Close'].pct_change().std() if not hist.empty else 0.02
            current_price = self.router.get_latest_price(symbol) or 100.0
        except Exception as e:
            logger.warning(f"Failed to fetch market data for {symbol}: {e}")
            adv = 1000000
            volatility = 0.02
            current_price = 100.0

        # 2. Determine number of slices (15-min intervals)
        interval_mins = 15
        n_slices = max(1, int(duration_hours * 60 / interval_mins))
        
        # 3. Use default profile interpolated to n_slices
        xp = np.linspace(0, 1, len(self.default_profile))
        x = np.linspace(0, 1, n_slices)
        weights = np.interp(x, xp, self.default_profile)
        weights /= weights.sum()
        
        slices = []
        now = datetime.utcnow()
        
        for i, w in enumerate(weights):
            slice_qty = quantity * w
            
            # 4. Estimate market impact for this slice
            impact = self.impact_model.estimate_impact(
                symbol=symbol,
                side=side,
                quantity=slice_qty,
                price=current_price,
                volatility=volatility,
                adv=adv / (6.5 * 60 / interval_mins) # Pro-rated ADV for this interval
            )
            
            slices.append(ExecutionSlice(
                quantity=slice_qty,
                target_time=now + timedelta(minutes=i * interval_mins),
                limit_price=None, # Market order for VWAP usually
                status="PENDING",
                estimated_impact_bps=impact.impact_bps
            ))
            
        logger.info(f"VWAP generated {len(slices)} slices for {symbol} | Total Qty: {quantity}")
        return slices
