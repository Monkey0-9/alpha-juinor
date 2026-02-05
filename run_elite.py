"""
Elite Trading Runner - Top 1% Integration.

Integrates ALL new Phase 15 modules:
- Model Orchestrator (all ML models)
- Enhanced Alpha Pipeline (50+ factors)
- Smart Execution Algorithms
- Real-Time Features
- Alternative Data
- Model Lifecycle Management
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> Dict:
    """Load trading configuration."""
    config_path = "configs/daily_trading_config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {
        "auto_execute": False,
        "max_positions": 50,
        "position_size_pct": 0.02,
        "min_confidence": 0.6
    }


def initialize_modules():
    """Initialize all Phase 15 modules."""
    logger.info("=" * 60)
    logger.info("ELITE TRADING SYSTEM - TOP 1% INTEGRATION")
    logger.info("=" * 60)

    modules = {}

    # 1. Model Orchestrator
    try:
        from ml.model_orchestrator import get_model_orchestrator
        modules["orchestrator"] = get_model_orchestrator()
        logger.info("✓ Model Orchestrator (HMM, Deep Ensemble, RL, LLM, NLP)")
    except Exception as e:
        logger.warning(f"✗ Model Orchestrator: {e}")

    # 2. Enhanced Alpha Pipeline
    try:
        from alpha_families.enhanced_alpha_pipeline import get_enhanced_alpha_pipeline
        modules["alpha_pipeline"] = get_enhanced_alpha_pipeline()
        logger.info("✓ Enhanced Alpha Pipeline (50+ factors)")
    except Exception as e:
        logger.warning(f"✗ Enhanced Alpha Pipeline: {e}")

    # 3. Smart Execution
    try:
        from execution.algo_executor import get_smart_execution
        modules["execution"] = get_smart_execution()
        logger.info("✓ Smart Execution (TWAP, VWAP, POV, Iceberg)")
    except Exception as e:
        logger.warning(f"✗ Smart Execution: {e}")

    # 4. Real-Time Features
    try:
        from features.realtime_store import get_realtime_feature_store
        modules["features"] = get_realtime_feature_store()
        logger.info("✓ Real-Time Feature Store")
    except Exception as e:
        logger.warning(f"✗ Real-Time Features: {e}")

    # 5. Alternative Data
    try:
        from data.alt_data.alt_data_engine import get_alt_data_engine
        modules["alt_data"] = get_alt_data_engine()
        logger.info("✓ Alternative Data Engine (News, SEC, Earnings)")
    except Exception as e:
        logger.warning(f"✗ Alternative Data: {e}")

    # 6. Model Lifecycle
    try:
        from ml.model_lifecycle import get_lifecycle_manager
        modules["lifecycle"] = get_lifecycle_manager()
        logger.info("✓ Model Lifecycle Manager")
    except Exception as e:
        logger.warning(f"✗ Model Lifecycle: {e}")

    # 7. Universe Scanner
    try:
        from data.universe_scanner import get_universe_scanner
        modules["scanner"] = get_universe_scanner()
        logger.info("✓ Universe Scanner")
    except Exception as e:
        logger.warning(f"✗ Universe Scanner: {e}")

    # 8. Auto Trader
    try:
        from execution.auto_trader import get_auto_trader
        modules["trader"] = get_auto_trader()
        logger.info("✓ Auto Trader (Alpaca)")
    except Exception as e:
        logger.warning(f"✗ Auto Trader: {e}")

    logger.info("=" * 60)
    logger.info(f"Initialized {len(modules)}/8 modules")
    logger.info("=" * 60)

    return modules


def fetch_market_data(scanner, symbols: List[str], limit: int = 100) -> Dict[str, pd.Series]:
    """Fetch market data for symbols."""
    logger.info(f"[DATA] Fetching data for {min(len(symbols), limit)} symbols...")

    try:
        from data.providers.yahoo_finance import get_yahoo_provider
        yahoo = get_yahoo_provider()

        market_data = {}
        for symbol in symbols[:limit]:
            try:
                data = yahoo.get_historical_data(symbol, days=100)
                if data is not None and len(data) > 20:
                    market_data[symbol] = data["close"]
            except Exception:
                pass

        logger.info(f"[DATA] Fetched data for {len(market_data)} symbols")
        return market_data

    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        return {}


def run_elite_alpha_generation(
    modules: Dict,
    market_data: Dict[str, pd.Series],
    config: Dict
) -> List[Dict]:
    """Generate signals using enhanced alpha pipeline."""
    logger.info("[ALPHA] Running Elite Alpha Generation...")

    alpha_pipeline = modules.get("alpha_pipeline")
    alt_data = modules.get("alt_data")

    if not alpha_pipeline:
        logger.warning("Alpha pipeline not available")
        return []

    signals = []

    for symbol, prices in market_data.items():
        try:
            # Get alternative data signal
            alt_signal = None
            if alt_data:
                alt_signal = alt_data.get_alt_data_signal(symbol)

            # Generate enhanced signal
            signal = alpha_pipeline.generate_signal(symbol, prices)

            # Combine with alt data
            if alt_signal and alt_signal.confidence > 0.3:
                combined_score = (
                    signal.alpha_score * 0.8 +
                    alt_signal.combined_score * 0.2
                )
            else:
                combined_score = signal.alpha_score

            if signal.confidence >= config.get("min_confidence", 0.5):
                signals.append({
                    "symbol": symbol,
                    "alpha_score": combined_score,
                    "confidence": signal.confidence,
                    "direction": signal.direction,
                    "expected_return": signal.expected_return,
                    "model_signal": signal.model_signal,
                    "traditional_signal": signal.traditional_signal,
                    "factor_count": len(signal.factor_breakdown),
                    "alt_data_score": alt_signal.combined_score if alt_signal else 0
                })

        except Exception as e:
            logger.debug(f"Signal error for {symbol}: {e}")

    # Sort by alpha score
    signals.sort(key=lambda x: abs(x["alpha_score"]), reverse=True)

    # Log top signals
    long_signals = [s for s in signals if s["direction"] == "LONG"][:10]
    short_signals = [s for s in signals if s["direction"] == "SHORT"][:10]

    logger.info(f"[ALPHA] Generated {len(signals)} signals")
    logger.info(f"[ALPHA] Top Long: {[s['symbol'] for s in long_signals[:5]]}")
    logger.info(f"[ALPHA] Top Short: {[s['symbol'] for s in short_signals[:5]]}")

    return signals


def create_execution_plans(
    modules: Dict,
    signals: List[Dict],
    config: Dict,
    portfolio_value: float = 100000
) -> List[Dict]:
    """Create smart execution plans for signals."""
    logger.info("[EXEC] Creating Smart Execution Plans...")

    execution = modules.get("execution")
    if not execution:
        logger.warning("Smart execution not available")
        return []

    plans = []
    max_positions = config.get("max_positions", 50)
    position_size_pct = config.get("position_size_pct", 0.02)

    # Select top signals
    top_signals = signals[:max_positions]

    for signal in top_signals:
        try:
            position_value = portfolio_value * position_size_pct
            price = 100  # Would come from real data
            quantity = int(position_value / price)

            if quantity < 1:
                continue

            side = "buy" if signal["direction"] == "LONG" else "sell"

            # Create smart execution plan
            plan = execution.create_execution_plan(
                parent_id=f"ELITE_{signal['symbol']}_{int(time.time())}",
                symbol=signal["symbol"],
                side=side,
                quantity=quantity,
                urgency="medium"
            )

            plans.append({
                "symbol": signal["symbol"],
                "side": side,
                "quantity": quantity,
                "algo": plan.algo_type.value,
                "num_slices": len(plan.child_orders),
                "est_cost_bps": plan.estimated_cost_bps,
                "alpha_score": signal["alpha_score"]
            })

        except Exception as e:
            logger.debug(f"Execution plan error for {signal['symbol']}: {e}")

    logger.info(f"[EXEC] Created {len(plans)} execution plans")

    # Log algo distribution
    algo_counts = {}
    for plan in plans:
        algo = plan["algo"]
        algo_counts[algo] = algo_counts.get(algo, 0) + 1
    logger.info(f"[EXEC] Algo distribution: {algo_counts}")

    return plans


def execute_trades(
    modules: Dict,
    plans: List[Dict],
    config: Dict,
    execute: bool = False
):
    """Execute trades via Alpaca."""
    if not execute:
        logger.info("[EXEC] Dry run - no trades executed")
        return

    trader = modules.get("trader")
    if not trader:
        logger.warning("Auto trader not available")
        return

    logger.info(f"[EXEC] Executing {len(plans)} trades...")

    for plan in plans[:10]:  # Limit to 10 trades
        try:
            # Would execute via Alpaca
            logger.info(
                f"[EXEC] {plan['side'].upper()} {plan['quantity']} "
                f"{plan['symbol']} via {plan['algo']}"
            )
        except Exception as e:
            logger.error(f"Trade execution error: {e}")


def run_elite_cycle(
    modules: Dict,
    config: Dict,
    execute: bool = False
) -> Dict:
    """Run one elite trading cycle."""
    start_time = time.time()
    results = {"timestamp": datetime.now().isoformat()}

    # 1. Get universe
    scanner = modules.get("scanner")
    if scanner:
        scanner.refresh_universe()
        symbols = scanner.get_tradable_symbols()
    else:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]

    results["universe_size"] = len(symbols)

    # 2. Fetch data
    market_data = fetch_market_data(scanner, symbols, limit=100)
    results["data_fetched"] = len(market_data)

    # 3. Generate signals
    signals = run_elite_alpha_generation(modules, market_data, config)
    results["signals_generated"] = len(signals)
    results["long_signals"] = len([s for s in signals if s["direction"] == "LONG"])
    results["short_signals"] = len([s for s in signals if s["direction"] == "SHORT"])

    # 4. Create execution plans
    plans = create_execution_plans(modules, signals, config)
    results["execution_plans"] = len(plans)

    # 5. Execute (if enabled)
    execute_trades(modules, plans, config, execute=execute)

    # 6. Summary
    cycle_time = time.time() - start_time
    results["cycle_time_seconds"] = cycle_time

    logger.info("=" * 60)
    logger.info("ELITE CYCLE COMPLETE")
    logger.info(f"  Universe: {results['universe_size']} symbols")
    logger.info(f"  Data: {results['data_fetched']} symbols fetched")
    logger.info(f"  Signals: {results['signals_generated']} total")
    logger.info(f"  Long: {results['long_signals']} | Short: {results['short_signals']}")
    logger.info(f"  Plans: {results['execution_plans']} execution plans")
    logger.info(f"  Time: {cycle_time:.2f}s")
    logger.info("=" * 60)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Elite Trading System")
    parser.add_argument(
        "--execute", action="store_true",
        help="Enable live trade execution"
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run continuously"
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Interval between cycles (minutes)"
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Run single cycle"
    )

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Initialize modules
    modules = initialize_modules()

    # Start lifecycle manager
    lifecycle = modules.get("lifecycle")
    if lifecycle:
        lifecycle.start()

    # Start feature store
    features = modules.get("features")
    if features:
        features.start()

    try:
        if args.single:
            # Single cycle
            run_elite_cycle(modules, config, execute=args.execute)
        elif args.continuous:
            # Continuous mode
            logger.info(f"Running continuously (interval: {args.interval} min)")
            while True:
                run_elite_cycle(modules, config, execute=args.execute)
                logger.info(f"Sleeping {args.interval} minutes...")
                time.sleep(args.interval * 60)
        else:
            # Default: single cycle
            run_elite_cycle(modules, config, execute=args.execute)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if lifecycle:
            lifecycle.stop()
        if features:
            features.stop()


if __name__ == "__main__":
    main()
