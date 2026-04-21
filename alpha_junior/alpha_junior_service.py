#!/usr/bin/env python3
"""
Alpha Junior - Windows Service (Kernel-Level Operation)
Runs as system service for 24/7 operation with full monitoring
"""

import win32serviceutil
import win32service
import win32event
import win32api
import win32con
import servicemanager
import socket
import sys
import os
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class AlphaJuniorService(win32serviceutil.ServiceFramework):
    _svc_name_ = "AlphaJuniorTrading"
    _svc_display_name_ = "Alpha Junior - Automated Trading Service"
    _svc_description_ = "Kernel-level automated stock trading service with Alpaca integration. Runs 24/7, executes trades, monitors portfolio performance."
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_running = False
        self.server_thread = None
        
    def SvcStop(self):
        """Called when service is stopped"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        self.is_running = False
        servicemanager.LogInfoMsg("Alpha Junior Service - Stopping...")
        
    def SvcDoRun(self):
        """Main service execution"""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        self.main()
        
    def setup_logging(self):
        """Setup comprehensive logging to Windows Event Log and file"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # File logging
        file_handler = RotatingFileHandler(
            'logs/service.log',
            maxBytes=10485760,
            backupCount=10
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Console logging (for service debugging)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def load_environment(self):
        """Load environment from .env file"""
        try:
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key] = value.strip().strip('"').strip("'")
                return True
        except Exception as e:
            servicemanager.LogErrorMsg(f"Failed to load .env: {e}")
        return False
        
    def run_flask_app(self, logger):
        """Run the Flask application"""
        try:
            from app import app
            
            logger.info("Starting Flask application on port 5000...")
            logger.info("Access URLs:")
            logger.info("  - http://localhost:5000 (Dashboard)")
            logger.info("  - http://localhost:5000/api/trading/account (Trading)")
            
            # Run Flask with threading
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
            
        except Exception as e:
            logger.error(f"Flask application error: {e}")
            logger.error(traceback.format_exc())
            servicemanager.LogErrorMsg(f"Alpha Junior Flask Error: {e}")
            
    def monitor_trades(self, logger):
        """Monitor trading activity and log to Windows Event Log"""
        try:
            import requests
            import time
            
            logger.info("Trade monitoring started...")
            
            while self.is_running:
                try:
                    # Check account status every 5 minutes
                    response = requests.get(
                        'http://localhost:5000/api/trading/account',
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        account = response.json()
                        if account.get('success'):
                            data = account.get('account', {})
                            portfolio_value = data.get('portfolio_value', 0)
                            cash = data.get('cash', 0)
                            
                            logger.info(f"Portfolio: ${portfolio_value:,.2f} | Cash: ${cash:,.2f}")
                            
                            # Log significant events to Windows Event Log
                            if portfolio_value != 100000:
                                change = portfolio_value - 100000
                                pct = (change / 100000) * 100
                                servicemanager.LogInfoMsg(
                                    f"Alpha Junior P&L: ${change:+.2f} ({pct:+.2f}%)"
                                )
                    
                    # Check positions
                    pos_response = requests.get(
                        'http://localhost:5000/api/trading/positions',
                        timeout=10
                    )
                    
                    if pos_response.status_code == 200:
                        positions = pos_response.json()
                        if positions.get('success'):
                            count = positions.get('count', 0)
                            if count > 0:
                                logger.info(f"Active positions: {count}")
                    
                except Exception as e:
                    logger.warning(f"Monitor check error: {e}")
                    
                # Wait 5 minutes before next check
                for _ in range(300):  # 300 seconds = 5 minutes
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Monitor thread error: {e}")
            
    def main(self):
        """Main service loop"""
        logger = self.setup_logging()
        
        logger.info("=" * 60)
        logger.info("Alpha Junior Service Starting (Kernel-Level)")
        logger.info("=" * 60)
        
        # Load environment
        if self.load_environment():
            logger.info("Environment loaded from .env")
        else:
            logger.warning("Could not load .env file")
            
        # Log Alpaca status
        alpaca_key = os.getenv('ALPACA_API_KEY', '')
        alpaca_secret = os.getenv('ALPACA_SECRET_KEY', '')
        
        if alpaca_key and alpaca_secret and alpaca_secret != 'YOUR_SECRET_KEY_HERE':
            logger.info("Alpaca Paper Trading: ENABLED")
            servicemanager.LogInfoMsg("Alpha Junior: Alpaca Trading ENABLED")
        else:
            logger.warning("Alpaca Paper Trading: DISABLED (add API keys)")
            servicemanager.LogWarningMsg("Alpha Junior: Alpaca Trading DISABLED")
            
        self.is_running = True
        
        # Start Flask in a thread
        flask_thread = threading.Thread(target=self.run_flask_app, args=(logger,))
        flask_thread.daemon = True
        flask_thread.start()
        
        logger.info("Flask server thread started")
        
        # Wait a moment for Flask to start
        time.sleep(3)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_trades, args=(logger,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info("Monitoring thread started")
        
        # Log service status
        servicemanager.LogInfoMsg("Alpha Junior Service Running on http://localhost:5000")
        
        # Keep service running
        while self.is_running:
            # Check if we should stop (every second)
            rc = win32event.WaitForSingleObject(self.stop_event, 1000)
            if rc == win32event.WAIT_OBJECT_0:
                # Stop signal received
                break
                
        logger.info("Service stopping...")
        self.is_running = False
        
        # Wait for threads to finish
        if flask_thread.is_alive():
            flask_thread.join(timeout=5)
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=5)
            
        logger.info("Service stopped")
        
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STOPPED,
            (self._svc_name_, '')
        )

def install_service():
    """Install the service"""
    try:
        win32serviceutil.InstallService(
            AlphaJuniorService._svc_class_,
            AlphaJuniorService._svc_name_,
            AlphaJuniorService._svc_display_name_,
            startType=win32service.SERVICE_AUTO_START
        )
        print(f"Service '{AlphaJuniorService._svc_display_name_}' installed successfully!")
        print("To start: net start AlphaJuniorTrading")
        print("To stop: net stop AlphaJuniorTrading")
        return True
    except Exception as e:
        print(f"Failed to install service: {e}")
        return False

def remove_service():
    """Remove the service"""
    try:
        win32serviceutil.RemoveService(AlphaJuniorService._svc_name_)
        print(f"Service '{AlphaJuniorService._svc_name_}' removed successfully!")
        return True
    except Exception as e:
        print(f"Failed to remove service: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Run as service
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(AlphaJuniorService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        # Handle command line (install, remove, etc.)
        if sys.argv[1] == 'install':
            install_service()
        elif sys.argv[1] == 'remove':
            remove_service()
        else:
            win32serviceutil.HandleCommandLine(AlphaJuniorService)
