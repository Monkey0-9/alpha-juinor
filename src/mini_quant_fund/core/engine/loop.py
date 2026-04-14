
import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from mini_quant_fund.core.common.exceptions import EngineError

logger = logging.getLogger(__name__)

class TradingLoop:
    """Institutional-grade trading loop with high precision and reliability."""
    
    def __init__(self, tick_interval: float = 1.0, data_refresh_interval: int = 1800):
        self.tick_interval = tick_interval
        self.data_refresh_interval = data_refresh_interval
        self.running = False
        self.paused = False
        self._thread = None
        self._stop_event = threading.Event()
        
    def start(self, tick_func: callable, refresh_func: callable):
        """Start the trading loop."""
        if self.running:
            return
            
        self.running = True
        self._stop_event.clear()
        
        def _run():
            last_refresh = 0
            while not self._stop_event.is_set():
                if self.paused:
                    time.sleep(1)
                    continue
                    
                start_time = time.time()
                
                # Check for data refresh
                if start_time - last_refresh >= self.data_refresh_interval:
                    try:
                        refresh_func()
                        last_refresh = start_time
                    except Exception as e:
                        logger.error(f"Data refresh failed: {e}")
                
                # Run decision tick
                try:
                    tick_func()
                except Exception as e:
                    logger.error(f"Decision tick failed: {e}")
                
                # Precision sleep
                elapsed = time.time() - start_time
                sleep_time = max(0, self.tick_interval - elapsed)
                time.sleep(sleep_time)
                
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        logger.info(f"Trading loop started with tick_interval={self.tick_interval}s")

    def stop(self):
        """Stop the trading loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.running = False
        logger.info("Trading loop stopped")
