"""
Silent Heartbeat System

Monitors system health without noise.
Alerts ONLY when heartbeat is missing.
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SilentHeartbeat:
    """
    Silent heartbeat monitor.
    
    Philosophy:
    - Healthy systems are silent
    - Alert only when heartbeat stops
    - Internal monitoring, no spam
    """
    
    def __init__(self, 
                 interval_seconds: int = 60,
                 alert_threshold_minutes: int = 5,
                 alert_callback: Optional[Callable] = None):
        self.interval = interval_seconds
        self.alert_threshold = timedelta(minutes=alert_threshold_minutes)
        self.alert_callback = alert_callback
        
        self._last_heartbeat: Optional[datetime] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._alerted = False
    
    def start(self):
        """Start heartbeat and watchdog."""
        if self._running:
            return
        
        self._running = True
        self._last_heartbeat = datetime.now()
        
        # Start heartbeat thread
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        
        # Start watchdog thread
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        
        logger.info("Silent heartbeat started")
    
    def stop(self):
        """Stop heartbeat."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=5)
        logger.info("Silent heartbeat stopped")
    
    def _heartbeat_loop(self):
        """Internal heartbeat loop (silent)."""
        while self._running:
            self._last_heartbeat = datetime.now()
            # Log to file only (DEBUG level)
            logger.debug(f"Heartbeat: {self._last_heartbeat.isoformat()}")
            time.sleep(self.interval)
    
    def _watchdog_loop(self):
        """Watchdog monitors heartbeat and alerts if missing."""
        while self._running:
            time.sleep(30)  # Check every 30 seconds
            
            if self._last_heartbeat is None:
                continue
            
            time_since_heartbeat = datetime.now() - self._last_heartbeat
            
            if time_since_heartbeat > self.alert_threshold:
                if not self._alerted:
                    # Heartbeat missing - ALERT
                    self._send_alert(time_since_heartbeat)
                    self._alerted = True
            else:
                # Heartbeat recovered
                if self._alerted:
                    self._send_recovery()
                    self._alerted = False
    
    def _send_alert(self, time_since: timedelta):
        """Send alert for missing heartbeat."""
        message = f"🚨 HEARTBEAT MISSING: No heartbeat for {time_since.total_seconds() / 60:.1f} minutes"
        logger.critical(message)
        
        if self.alert_callback:
            try:
                self.alert_callback(message, severity="CRITICAL")
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _send_recovery(self):
        """Send recovery notification."""
        message = "✅ HEARTBEAT RECOVERED"
        logger.warning(message)
        
        if self.alert_callback:
            try:
                self.alert_callback(message, severity="WARNING")
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def pulse(self):
        """Manual heartbeat pulse (for external components)."""
        self._last_heartbeat = datetime.now()
