from core.global_session_tracker import get_global_session_tracker
from data.edgar.scraper import EdgarScraper
from research.network_analyzer import get_network_analyzer
from alpha.network_alpha import get_network_alpha
from control.rl_meta_controller import get_rl_controller
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemVerifier")

def verify_all():
    logger.info("=== 1. Testing Global Session Tracker ===")
    tracker = get_global_session_tracker()
    sessions = tracker.get_active_sessions()
    logger.info(f"Active Sessions: {sessions}")
    logger.info(f"NYSE Open? {tracker.is_market_open('NYSE')}")
    logger.info(f"Liquidity State: {tracker.get_liquidity_state()}")

    logger.info("\n=== 2. Testing Network Alpha (P1) ===")
    analyzer = get_network_analyzer()
    anomalies = analyzer.detect_anomalous_clusters()
    logger.info(f"Anomalies Detected: {len(anomalies)}")

    alpha = get_network_alpha()
    signals = alpha.generate_signals(None)
    logger.info(f"Alpha Signals Generated: {len(signals)}")
    for s in signals:
        logger.info(f"  Signal: {s}")

    logger.info("\n=== 3. Testing RL Controller (P2) ===")
    rl = get_rl_controller()
    action = rl.predict_action(rl.get_state(None))
    logger.info(f"RL Action: {action}")
    weights = rl.get_allocation_weights(["StratA", "StratB"])
    logger.info(f"Alloc Weights: {weights}")

if __name__ == "__main__":
    verify_all()
