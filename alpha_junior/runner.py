#!/usr/bin/env python3
"""
Alpha Junior - 24/7 Production Runner
Called by run_24_7.bat
"""

import sys
import traceback
import os

# Load environment variables from .env file
try:
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip().strip('"').strip("'")
    print("✓ Environment loaded from .env")
except Exception as e:
    print(f"⚠ Could not load .env: {e}")

try:
    from app import app
    import logging
    from logging.handlers import RotatingFileHandler

    # Setup file logging
    if not os.path.exists('logs'):
        os.makedirs('logs')

    handler = RotatingFileHandler('logs/alpha_junior.log', maxBytes=10485760, backupCount=5)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    print('Starting Alpha Junior production server on port 5000...')
    print('Press Ctrl+C to stop')
    sys.stdout.flush()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

except Exception as e:
    print(f"ERROR: {e}")
    print(traceback.format_exc())
    sys.exit(1)
