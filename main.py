#!/usr/bin/env python3
"""
Nexus Quant Platform - Institutional Entry Point
================================================
Standardized entry point for historical research, backtesting, and live execution.
"""

import argparse
import sys
import os
import signal
from pathlib import Path

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nexus.core.context import engine_context

def handle_shutdown(signum, frame):
    """Graceful shutdown handler."""
    engine_context.logger.info("Shutdown signal received")
    engine_context.set_running(False)
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Nexus Institutional Quant Platform")
    parser.add_argument("--mode", type=str, choices=["backtest", "live", "sim"], default="sim",
                        help="Execution mode")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Register signals
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Initialize Engine
    engine_context.logger.info("Initializing Nexus Quant Platform Engine...")
    
    # Start Engine
    engine_context.set_running(True)
    
    # Placeholder for Engine Loop (Phase 1-5 will populate this with actual components)
    try:
        if args.mode == "backtest":
            engine_context.logger.info("Backtest mode initialized. Ready for research phase.")
        elif args.mode == "sim":
            engine_context.logger.info("Market simulation mode active.")
        elif args.mode == "live":
            # Extra safety check for live mode
            if not engine_context.config.trading_enabled:
                engine_context.logger.fatal("TRADING_ENABLED is False in config. Cannot start live mode.")
                sys.exit(1)
            engine_context.logger.warn("LIVE TRADING MODE INITIALIZED. EXECUTING IN PRODUCTION ENVIRONMENT.")
            
        # Keep engine alive if needed or run a specific task
        # For now, we wait for a signal
        import time
        while engine_context.engine_state["is_running"]:
            time.sleep(1)
            
    except Exception as e:
        engine_context.logger.fatal(f"System failure: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
