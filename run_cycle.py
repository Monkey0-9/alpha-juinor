#!/usr/bin/env python3
"""
Institutional Trading System - Main Entry Point

This script runs the complete institutional trading pipeline:
1. Fetches 5 years of daily market data for the full universe
2. Validates data quality per symbol
3. Computes comprehensive features
4. Runs all agent models (Momentum, MeanReversion, Volatility, etc.)
5. Aggregates decisions via Meta-Brain
6. Applies risk checks (CVaR, exposure limits, etc.)
7. Generates orders
8. Persists everything to the database
9. Produces cycle summary JSON

Usage:
    python run_cycle.py [--paper] [--universe <path>] [--symbols <comma-separated>]

Options:
    --paper        Run in paper mode (default: True)
    --live         Run in live mode (requires broker connection)
    --universe     Path to universe JSON file
    --symbols      Comma-separated list of symbols to process
    --workers      Number of parallel workers (default: 10)
    --test         Run test mode with sample data
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/cycle_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Institutional Trading Pipeline')
    parser.add_argument('--paper', action='store_true', default=True,
                        help='Run in paper mode (default)')
    parser.add_argument('--live', action='store_true',
                        help='Run in live mode (requires broker connection)')
    parser.add_argument('--universe', type=str, default='configs/universe.json',
                        help='Path to universe JSON file')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols to process')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel workers')
    parser.add_argument('--test', action='store_true',
                        help='Run test mode with sample data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run full cycle without persisting results to DB')


    args = parser.parse_args()

    # Mode validation
    paper_mode = args.paper and not args.live

    logger.info("=" * 80)
    logger.info("INSTITUTIONAL TRADING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Mode: {'PAPER' if paper_mode else 'LIVE'}")
    logger.info(f"Universe: {args.universe}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Test Mode: {args.test}")
    logger.info("=" * 80)

    # Ensure runtime directory exists
    Path('runtime').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    try:
        # Import after path setup
        from orchestration.cycle_runner import CycleOrchestrator, run_institutional_cycle
        from database.manager import get_db

        # Initialize database
        db = get_db()
        logger.info(f"Database initialized at {db.db_path}")

        # Health check
        health = db.health_check()
        logger.info(f"Database health: {json.dumps(health, indent=2, default=str)}")

        if args.test or args.symbols:
            # Subset mode
            logger.info("Running with symbol subset...")

            if args.test:
                from configs.universe import load_universe
                universe = load_universe(args.universe)
                target_symbols = universe[:5]
            else:
                target_symbols = [s.strip() for s in args.symbols.split(',')]

            logger.info(f"Target symbols: {target_symbols}")

            # Temporarily modify universe
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({'active_tickers': target_symbols}, f)
                temp_universe = f.name

            result = run_institutional_cycle(
                universe_path=temp_universe,
                lookback_years=1 if args.test else 5,
                max_workers=args.workers if not args.test else 2,
                paper_mode=True,
                dry_run=args.dry_run
            )

            os.unlink(temp_universe)
        else:
            # Full run
            result = run_institutional_cycle(
                universe_path=args.universe,
                lookback_years=5,
                max_workers=args.workers,
                paper_mode=paper_mode,
                dry_run=args.dry_run
            )



        # Print summary
        logger.info("=" * 80)
        logger.info("CYCLE SUMMARY")
        logger.info("=" * 80)
        print(json.dumps(result.to_dict(), indent=2, default=str))

        # Save result
        output_path = Path(f'output/cycle_result_{result.cycle_id}.json')
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Result saved to {output_path}")

        # Return success
        return 0

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

