
import sys
import os
import logging
import json
import yaml
import redis
from datetime import datetime

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.manager import DatabaseManager
from monitoring.health import HealthMonitor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProdHealthCheck")

def check_database():
    try:
        db = DatabaseManager()
        status = db.health_check()
        return status.get("healthy", False), status
    except Exception as e:
        return False, str(e)

def check_kill_switch():
    try:
        import yaml
        import redis
        config_path = "configs/kill_switch_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)['kill_switch']
        else:
            config = {'redis_host': 'localhost', 'redis_port': 6379}

        r = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            socket_connect_timeout=2
        )
        if r.ping():
            return True, f"Redis connected at {config['redis_host']}:{config['redis_port']}"
        return False, "Redis ping failed"
    except Exception as e:
        return False, f"Kill Switch Error: {e}"

def check_monitoring():
    try:
        monitor = HealthMonitor({'monitoring': {}})
        status = monitor.get_health_status()
        return not status['circuit_breaker_tripped'], status
    except Exception as e:
        return False, str(e)

def main():
    logger.info("Starting production health check...")

    checks = {
        "Database": check_database,
        "KillSwitch": check_kill_switch,
        "Monitoring": check_monitoring
    }

    results = {}
    all_passed = True

    for name, check_fn in checks.items():
        try:
            passed, details = check_fn()
            results[name] = {
                "passed": passed,
                "details": details
            }
            if passed:
                logger.info(f"✅ {name}: PASSED")
            else:
                logger.error(f"❌ {name}: FAILED - {details}")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {name}: CRASHED - {e}")
            results[name] = {"passed": False, "details": str(e)}
            all_passed = False

    # Save report
    os.makedirs("runtime", exist_ok=True)
    report_path = "runtime/prod_health_check_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Health check report saved to {report_path}")

    if all_passed:
        logger.info(">>> SYSTEM STATUS: GO <<<")
        sys.exit(0)
    else:
        logger.error(">>> SYSTEM STATUS: NO-GO <<<")
        sys.exit(1)

if __name__ == "__main__":
    main()
