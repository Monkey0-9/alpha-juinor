#!/usr/bin/env python3
"""
Alpha Junior - Windows Service Runner
Runs 24/7 as a Windows service with auto-restart
"""

import sys
import os
import time
import logging
import traceback
from logging.handlers import RotatingFileHandler

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/service.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('AlphaJunior')

def run_server():
    """Run the Flask server"""
    try:
        from app import app
        logger.info('Starting Alpha Junior server...')
        
        # Production settings
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False  # Important for service
        )
    except Exception as e:
        logger.error(f'Server error: {e}')
        logger.error(traceback.format_exc())
        raise

def main_loop():
    """Main loop with auto-restart"""
    restart_count = 0
    max_restarts = 100
    
    while restart_count < max_restarts:
        try:
            logger.info(f'Starting server (attempt {restart_count + 1})')
            run_server()
            # If we get here, server stopped normally
            logger.info('Server stopped normally')
            break
            
        except KeyboardInterrupt:
            logger.info('Received shutdown signal')
            break
            
        except Exception as e:
            restart_count += 1
            logger.error(f'Server crashed: {e}')
            logger.info(f'Restarting in 10 seconds... (attempt {restart_count}/{max_restarts})')
            time.sleep(10)
    
    if restart_count >= max_restarts:
        logger.error('Max restarts reached. Giving up.')
        return 1
    
    return 0

class WindowsService:
    """Windows Service wrapper"""
    
    def __init__(self):
        self.running = False
    
    def run(self):
        """Run as service"""
        logger.info('Alpha Junior Windows Service starting...')
        self.running = True
        
        try:
            return main_loop()
        except Exception as e:
            logger.error(f'Service error: {e}')
            return 1
    
    def stop(self):
        """Stop the service"""
        logger.info('Stopping Alpha Junior service...')
        self.running = False

if __name__ == '__main__':
    # Check if running as Windows service
    if len(sys.argv) > 1 and sys.argv[1] in ['install', 'start', 'stop', 'remove']:
        # Windows service commands
        import win32serviceutil
        import win32service
        import win32event
        import servicemanager
        
        class AlphaJuniorService(win32serviceutil.ServiceFramework):
            _svc_name_ = 'AlphaJunior'
            _svc_display_name_ = 'Alpha Junior Fund Platform'
            _svc_description_ = 'Institutional Fund Management Platform - Runs 24/7'
            
            def __init__(self, args):
                win32serviceutil.ServiceFramework.__init__(self, args)
                self.stop_event = win32event.CreateEvent(None, 0, 0, None)
                self.service = WindowsService()
            
            def SvcDoRun(self):
                servicemanager.LogInfoMsg('Alpha Junior starting...')
                self.service.run()
            
            def SvcStop(self):
                servicemanager.LogInfoMsg('Alpha Junior stopping...')
                self.service.stop()
                win32event.SetEvent(self.stop_event)
        
        win32serviceutil.HandleCommandLine(AlphaJuniorService)
    else:
        # Run directly
        logger.info('Starting Alpha Junior in standalone mode...')
        sys.exit(main_loop())
