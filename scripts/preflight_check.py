import sys
import subprocess
import importlib.util
import os

def check_module(name):
    return importlib.util.find_spec(name) is not None

def check_redis():
    try:
        import redis
        # Try localhost
        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2)
        if r.ping():
            return True

        # Try WSL IP
        import subprocess
        wsl_ip = subprocess.check_output(['wsl', 'hostname', '-I'], text=True).strip().split()[0]
        r = redis.Redis(host=wsl_ip, port=6379, socket_connect_timeout=2)
        return r.ping()
    except:
        return False

def check_wsl_redis():
    try:
        # Check if WSL is available and redis is running there
        result = subprocess.run(['wsl', 'redis-cli', 'ping'], capture_output=True, text=True, timeout=5)
        return 'PONG' in result.stdout
    except:
        return False

print('=== INSTITUTIONAL PREFLIGHT CHECK ===')

checks = [
    ('Redis Python Client', check_module('redis')),
    ('Pandas', check_module('pandas')),
    ('Statsmodels', check_module('statsmodels')),
    ('Redis Server (Local/WSL)', check_redis() or check_wsl_redis()),
]

all_passed = True
for check_name, passed in checks:
    status = '✅' if passed else '❌'
    print(f'{status} {check_name}')
    if not passed:
        all_passed = False

print('\n' + '='*40)

if all_passed:
    print('✅ SYSTEM READY FOR LAUNCH')
    sys.exit(0)
else:
    print('🚨 SYSTEM NOT READY')
    sys.exit(1)
