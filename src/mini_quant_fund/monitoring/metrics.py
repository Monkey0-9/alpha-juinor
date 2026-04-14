
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Institutional Observability Layer.
    Tracks core KPIs and logs them in structured format for ingestion (ELK/Splunk ready).
    """
    def __init__(self, log_path: str = "logs/metrics.jsonl"):
        self.log_path = log_path
        self._previous_equity = None
        self._initial_capital = None
        
    def log_cycle(self, 
                  timestamp: pd.Timestamp, 
                  equity: float, 
                  positions: Dict[str, float], 
                  orders: list,
                  risk_state: str,
                  regime: str):
        """
        Record signals, execution, and portfolio state after a cycle.
        """
        if self._initial_capital is None:
            self._initial_capital = equity
        
        # 1. PnL Calculation
        daily_pnl = 0.0
        daily_ret = 0.0
        if self._previous_equity:
            daily_pnl = equity - self._previous_equity
            daily_ret = daily_pnl / self._previous_equity
            
        self._previous_equity = equity
        
        # 2. Drawdown
        # (Simplified calculation here, typically passed from RiskEngine)
        total_ret = (equity - self._initial_capital) / self._initial_capital
        
        # 3. Leverage
        # Approximate: sum(abs(pos_value)) / equity
        # Needs prices... we assume positions dict has quantities here? 
        # Actually positions from handler is just Qty. We usually need value.
        # For this lightweight logger, we just log counts.
        
        metrics = {
            "timestamp": timestamp.isoformat(),
            "nav": round(equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_return_bps": round(daily_ret * 10000, 1),
            "total_return_pct": round(total_ret * 100, 2),
            "positions_count": len([p for p in positions.values() if abs(p) > 0]),
            "orders_count": len(orders),
            "risk_state": risk_state,
            "market_regime": regime,
            "leverage_utilization": "N/A" # Calculated upstream usually
        }
        
        # Structured Log Output
        log_entry = json.dumps(metrics)
        
        # 1. Console (Human)
        logger.info(f"📊 METRICS: NAV=${equity:,.0f} | Day={daily_ret:.2%} | Risk={risk_state}")
        
        # 2. JSONL File (Machine)
        try:
            with open(self.log_path, 'a') as f:
                f.write(log_entry + "\n")
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")
            
        return metrics
