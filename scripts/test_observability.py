
import requests
import sys
import os
import time
import logging

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.prometheus_exporter import metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ObservabilityTest")

def run_test():
    logger.info("Testing Observability Stack...")

    # 1. Update some metrics
    metrics.set_nav(105000.0, "test_cycle_1")
    metrics.inc_order("buy", "market")
    metrics.observe_latency("decision_latency", 0.45)

    # 2. Scrape Endpoint
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            logger.info("Successfully scraped /metrics endpoint.")
            content = response.text

            # Check for keys
            if "quant_fund_portfolio_nav" in content:
                logger.info("PASS: Found NAV metric.")
            else:
                logger.error("FAIL: NAV metric missing.")

            if "quant_fund_order_count_total" in content:
                 logger.info("PASS: Found Order metric.")
            else:
                 logger.error("FAIL: Order metric missing.")

            if "quant_fund_decision_latency_seconds_count" in content:
                  logger.info("PASS: Found Latency metric.")
            else:
                  logger.error("FAIL: Latency metric missing.")

        else:
            logger.error(f"FAIL: Endpoint returned {response.status_code}")

    except Exception as e:
        logger.error(f"FAIL: Could not connect to metrics server: {e}")

if __name__ == "__main__":
    # Give server a moment to bind if just imported
    time.sleep(1)
    run_test()
