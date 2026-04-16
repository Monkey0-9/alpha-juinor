# monitoring/runtime_checks.py
import logging
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)

class RuntimeMonitor:
    """
    Operational Health Monitor.
    Detects anomalies in execution and data quality in real-time.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.max_trades_per_day = config.get('max_trades_day', 50)
        self.max_slippage_bps = config.get('max_slippage_bps', 50)  # 0.5%
        self.min_volume_threshold = 100.0
        
    def check_trade_sanity(self, trade_count: int):
        if trade_count > self.max_trades_per_day:
            msg = f"TRADE VOLUME ALERT: {trade_count} trades detected, exceeding limit of {self.max_trades_per_day}"
            logger.error(msg)
            return False, msg
        return True, ""
        
    def check_data_quality(self, ticker: str, current_bar: pd.Series):
        """Checks for zero volume or stale prices."""
        if current_bar['Volume'] < self.min_volume_threshold:
            msg = f"DATA QUALITY ALERT [{ticker}]: Volume is {current_bar['Volume']} (Possible data gap)"
            logger.warning(msg)
            return False, msg
            
        if current_bar['High'] == current_bar['Low'] and current_bar['Volume'] > 0:
            msg = f"DATA QUALITY ALERT [{ticker}]: Zero price range with volume > 0."
            logger.info(msg)
            
        return True, ""
        
    def check_execution_slippage(self, ticker: str, fill_price: float, expected_price: float):
        slippage = abs(fill_price - expected_price) / expected_price
        if slippage > (self.max_slippage_bps / 10000.0):
            msg = f"SLIPPAGE ALERT [{ticker}]: Fill {fill_price} vs Expected {expected_price} (Slippage: {slippage:.2%})"
            logger.warning(msg)
            return False, msg
        return True, ""

    def check_equity_drift(self, current_equity: float, previous_equity: float):
        """Abnormal equity jumps (>5% in 1 bar)."""
        if previous_equity > 0:
            drift = abs(current_equity - previous_equity) / previous_equity
            if drift > 0.05:
                msg = f"EQUITY VOLATILITY ALERT: Drift of {drift:.2%} detected in single bar."
                logger.error(msg)
                return False, msg
        return True, ""
