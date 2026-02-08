"""
Simple runner script to execute a single paper trading cycle.
Demonstrates the complete production-ready system.
"""

import logging
import os
import sys

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit.decision_log import SystemHalt, shutdown
from orchestration.cycle_orchestrator import CycleOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


import argparse


def main():
    """Run a single paper trading cycle"""
    parser = argparse.ArgumentParser(description="Run Paper Cycle")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument("--run_id", type=str, default="manual", help="Run ID")
    args = parser.parse_args()

    try:
        logger.info(f"Initializing Cycle Orchestrator (RunID: {args.run_id})...")
        pm_config = {"confidence_threshold": args.confidence}
        orch = CycleOrchestrator(mode="paper", pm_config=pm_config)
        # Override initial run_id if needed, or just let Orchestrator gen one.
        # But we log it.
        logger.info(f"PM Config: {pm_config}")

        logger.info("Starting cycle...")
        results = orch.run_cycle()

        logger.info(f"\n✓ Cycle completed successfully")
        logger.info(f"✓ Generated {len(results)} decisions")
        logger.info(f"✓ Audit records written to runtime/audit.db")

        # Execution Phase
        logger.info("Starting execution phase...")
        try:
            from execution.alpaca_handler import AlpacaExecutionHandler

            # Initialize with paper=True to match run mode
            executor = AlpacaExecutionHandler(paper=True)

            executed_count = 0
            for d in results:
                # Execute valid decisions with attached orders
                if d.decision == "EXECUTE" and d.order:
                    try:
                        symbol = d.order.get("symbol")
                        qty = float(d.order.get("quantity", 0))
                        side = d.order.get("side", "BUY").upper()

                        # Adjust sign for Alpaca handler
                        if side == "SELL":
                            qty = -abs(qty)
                        else:
                            qty = abs(qty)

                        if qty != 0:
                            logger.info(
                                f"Submitting order for {symbol}: {side} {abs(qty)}"
                            )
                            executor.submit_order(symbol, qty)
                            executed_count += 1
                        else:
                            logger.warning(f"Skipping zero quantity order for {symbol}")

                    except Exception as e:
                        logger.error(f"Failed to execute order for {d.symbol}: {e}")

            logger.info(f"✓ Executed {executed_count} orders")

        except ImportError:
            logger.error("Could not import AlpacaExecutionHandler. Execution skipped.")
        except Exception as e:
            logger.error(f"Execution phase failed: {e}", exc_info=True)

        # Flush audit logs to ensure all records are written before exit
        shutdown()

        return 0

    except SystemHalt as e:
        logger.critical(f"SYSTEM HALT: {e}")
        logger.critical("Trading cannot proceed - fix critical issue and restart")
        return 1
    except KeyboardInterrupt:
        logger.warning("User interrupted cycle")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
