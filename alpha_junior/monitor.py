#!/usr/bin/env python3
"""
Alpha Junior - Health Monitor
Checks if server is running and alerts if down
"""

import requests
import time
import logging
import os
from datetime import datetime

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Monitor')

def check_health():
    """Check if Alpha Junior is healthy"""
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                logger.info('✓ Server is healthy')
                return True
        logger.warning(f'⚠ Health check failed: {response.status_code}')
        return False
    except requests.exceptions.ConnectionError:
        logger.error('✗ Server is DOWN - Connection refused')
        return False
    except Exception as e:
        logger.error(f'✗ Health check error: {e}')
        return False

def restart_server():
    """Attempt to restart the server"""
    logger.info('Attempting to restart server...')
    
    # Kill existing Python processes running app.py
    import subprocess
    try:
        # Find and kill existing processes
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                              capture_output=True, text=True)
        
        if 'python.exe' in result.stdout:
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], capture_output=True)
            time.sleep(2)
        
        # Start server
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.Popen(
            [sys.executable, 'app.py'],
            cwd=script_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        logger.info('Server restart initiated')
        
    except Exception as e:
        logger.error(f'Failed to restart: {e}')

def main():
    """Main monitoring loop"""
    logger.info('=' * 50)
    logger.info('Alpha Junior Health Monitor Started')
    logger.info('Checking http://localhost:5000 every 30 seconds')
    logger.info('=' * 50)
    
    fail_count = 0
    max_fails = 3
    
    while True:
        if check_health():
            fail_count = 0
        else:
            fail_count += 1
            logger.warning(f'Failure count: {fail_count}/{max_fails}')
            
            if fail_count >= max_fails:
                logger.error('Server appears to be down! Attempting restart...')
                restart_server()
                fail_count = 0
                time.sleep(10)  # Give server time to start
        
        time.sleep(30)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info('Monitor stopped by user')
    except Exception as e:
        logger.error(f'Monitor error: {e}')
