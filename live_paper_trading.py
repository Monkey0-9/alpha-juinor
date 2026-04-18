#!/usr/bin/env python3
"""
=============================================================================
NEXUS INSTITUTIONAL - LIVE PAPER TRADING WITH NEWS MONITORING
=============================================================================
Real-time monitoring system for paper trading
Monitors news, market events, sentiment, and generates trading signals
Designed to match elite firm capabilities (Jane Street, Citadel, Virtu, etc.)

Quick Start:
  python live_paper_trading.py --mode paper --interval 60

Features:
  ✓ Real-time news monitoring (RSS feeds from Bloomberg, Reuters, CNBC)
  ✓ Market data streaming (alpaca, yfinance, alpha_vantage, IB)
  ✓ Sentiment analysis (bullish/bearish signals)
  ✓ Event-driven trading signals
  ✓ Live portfolio monitoring
  ✓ Trade execution tracking
  ✓ Risk management
  ✓ 24/7 operational readiness
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nexus.institutional.live_monitor import LiveTradingMonitor
from src.nexus.core.context import engine_context


def main():
    """Main entry point for live paper trading monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Nexus Institutional - Live Paper Trading Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading with news monitoring (default)
  python live_paper_trading.py
  
  # Run for 1 hour
  python live_paper_trading.py --duration 3600
  
  # Update every 30 seconds
  python live_paper_trading.py --interval 30
  
  # Verbose logging
  python live_paper_trading.py --log-level DEBUG
"""
    )
    
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="paper",
        help="Execution mode (default: paper - no real money)"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        help="Run for N seconds (default: infinite)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("LivePaperTrading")
    
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*10 + "NEXUS INSTITUTIONAL - LIVE PAPER TRADING MONITOR" + " "*20 + "║")
    logger.info("║" + " "*15 + "Real-Time News & Event-Driven Trading" + " "*26 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Mode: {args.mode.upper()}")
    logger.info(f"  Update Interval: {args.interval}s")
    logger.info(f"  Duration: {args.duration}s" if args.duration else "  Duration: Infinite")
    logger.info(f"  Log Level: {args.log_level}")
    logger.info("")
    
    # Initialize engine
    engine_context.initialize("config/production.yaml")
    
    # Create and run monitor
    monitor = LiveTradingMonitor(update_interval=args.interval)
    
    # Run async monitoring loop
    try:
        asyncio.run(monitor.run_monitoring_loop(duration_seconds=args.duration))
    except KeyboardInterrupt:
        logger.info("\nShutdown initiated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
