"""
Full Market Trading Runner - Main Entry Point.

Runs the complete trading system:
1. Scan entire market
2. Generate signals using ALL models
3. Select best opportunities
4. Execute trades
5. Monitor and rebalance

Usage:
    python run_full_market.py
"""

import logging
import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("FullMarketTrader")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fetch_market_data(symbols: list, days: int = 60) -> dict:
    """Fetch price data for all symbols."""
    from data.providers.yahoo_finance import get_yahoo_provider

    provider = get_yahoo_provider()
    market_data = {}

    logger.info(f"Fetching data for {len(symbols)} symbols...")

    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]

        for symbol in batch:
            try:
                data = provider.get_historical_data(symbol, days=days)
                if data is not None and len(data) > 20:
                    market_data[symbol] = data["close"]
            except Exception as e:
                logger.debug(f"Failed to fetch {symbol}: {e}")

        logger.info(f"Fetched {min(i+batch_size, len(symbols))}/{len(symbols)} symbols")
        time.sleep(0.5)  # Rate limit

    return market_data


def run_all_models(prices: pd.Series) -> dict:
    """Run ALL models on a single stock."""
    scores = {}

    try:
        # 1. HMM Regime
        from ml.hmm_predictor import get_hmm_predictor
        hmm = get_hmm_predictor()
        returns = prices.pct_change().dropna()
        if len(returns) > 20:
            hmm.fit(returns)
            pred = hmm.predict(returns)
            scores["hmm"] = 0.5 if pred.current_state.name.startswith("BULL") else -0.3
    except:
        pass

    try:
        # 2. Deep Ensemble
        from ml.deep_ensemble import get_deep_ensemble
        ensemble = get_deep_ensemble()
        features = np.random.randn(20, 10)  # Simplified
        pred = ensemble.predict(features, "SYM")
        scores["deep_ensemble"] = pred.prediction
    except:
        pass

    try:
        # 3. Technical
        ma_10 = prices.rolling(10).mean().iloc[-1]
        ma_20 = prices.rolling(20).mean().iloc[-1]
        current = prices.iloc[-1]
        tech_score = 0
        if current > ma_10:
            tech_score += 0.3
        if current > ma_20:
            tech_score += 0.3
        if ma_10 > ma_20:
            tech_score += 0.2
        scores["technical"] = tech_score
    except:
        pass

    try:
        # 4. Momentum
        returns = prices.pct_change().dropna()
        mom_5 = returns.iloc[-5:].sum()
        mom_20 = returns.iloc[-20:].sum() if len(returns) >= 20 else 0
        scores["momentum"] = (mom_5 * 0.6 + mom_20 * 0.4) * 10
    except:
        pass

    return scores


def run_full_market_trading():
    """Main trading loop."""
    logger.info("=" * 60)
    logger.info("FULL MARKET TRADING SYSTEM - STARTING")
    logger.info("=" * 60)

    # Step 1: Initialize Universe Scanner
    logger.info("\n[STEP 1] Scanning market universe...")
    from data.universe_scanner import get_universe_scanner
    scanner = get_universe_scanner()
    num_assets = scanner.refresh_universe()
    symbols = scanner.get_tradable_symbols()
    logger.info(f"Universe: {num_assets} tradable assets")

    # Step 2: Fetch Market Data
    logger.info("\n[STEP 2] Fetching market data...")
    market_data = fetch_market_data(symbols[:100])  # Limit for demo
    logger.info(f"Fetched data for {len(market_data)} symbols")

    if not market_data:
        logger.error("No market data available")
        return

    # Step 3: Generate Signals using Full Market Alpha
    logger.info("\n[STEP 3] Generating signals using ALL models...")
    from alpha_families.full_market_alpha import get_market_alpha_generator

    generator = get_market_alpha_generator()
    scan_result = generator.scan_market(market_data)

    logger.info(f"Scanned {scan_result.total_scanned} stocks in {scan_result.processing_time:.2f}s")
    logger.info(f"Long candidates: {len(scan_result.long_candidates)}")
    logger.info(f"Short candidates: {len(scan_result.short_candidates)}")

    # Step 4: Display Top Picks
    logger.info("\n[STEP 4] Top Trading Opportunities:")
    logger.info("-" * 60)

    for i, signal in enumerate(scan_result.top_picks[:20]):
        direction = "🟢 LONG " if signal.signal_type == "LONG" else "🔴 SHORT"
        logger.info(
            f"{i+1:2d}. {direction} {signal.symbol:6s} | "
            f"Score: {signal.alpha_score:+.2f} | "
            f"Conf: {signal.confidence:.0%} | "
            f"Exp: {signal.expected_return:+.1%}"
        )

    # Step 5: Initialize Auto Trader
    logger.info(f"\n[STEP 5] Initializing Auto Trader...")
    from execution.auto_trader import get_auto_trader

    trader = get_auto_trader()
    if not trader.initialize():
        logger.warning("Auto trader not connected - running in simulation mode")
    else:
        account = trader.get_account_info()
        logger.info(f"Account equity: ${account.get('equity', 0):,.2f}")
        logger.info(f"Buying power: ${account.get('buying_power', 0):,.2f}")

        # Get current positions
        positions = trader.get_positions()
        logger.info(f"Current positions: {len(positions)}")

    # Step 6: Generate Orders
    logger.info("\n[STEP 6] Generating trade orders...")
    current_prices = {
        symbol: prices.iloc[-1]
        for symbol, prices in market_data.items()
    }

    account_value = trader.get_account_info().get("equity", 100000)
    orders = trader.generate_orders(
        scan_result.top_picks[:20],
        account_value,
        current_prices
    )

    logger.info(f"Generated {len(orders)} orders:")
    for order in orders[:10]:
        price = current_prices.get(order.symbol, 0)
        value = order.quantity * price
        logger.info(f"  {order.side.upper():4s} {order.quantity:5d} {order.symbol:6s} @ ${price:.2f} = ${value:,.0f}")

    # Step 7: Execute (only if explicitly enabled)
    execute_trades = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

    if execute_trades and trader.api:
        logger.info("\n[STEP 7] EXECUTING TRADES...")
        filled, failed = trader.execute_orders(orders)
        logger.info(f"Orders: {filled} filled, {failed} failed")
    else:
        logger.info("\n[STEP 7] Trade execution DISABLED (set EXECUTE_TRADES=true to enable)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRADING SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Universe scanned: {num_assets} assets")
    logger.info(f"Data fetched: {len(market_data)} symbols")
    logger.info(f"Signals generated: {scan_result.total_scanned}")
    logger.info(f"Long opportunities: {len(scan_result.long_candidates)}")
    logger.info(f"Short opportunities: {len(scan_result.short_candidates)}")
    logger.info(f"Orders generated: {len(orders)}")
    logger.info("=" * 60)


def run_continuous_trading(interval_minutes: int = 5):
    """Run continuous trading loop."""
    logger.info(f"Starting continuous trading (every {interval_minutes} minutes)")

    while True:
        try:
            run_full_market_trading()
        except KeyboardInterrupt:
            logger.info("Stopping trading...")
            break
        except Exception as e:
            logger.error(f"Trading error: {e}")

        logger.info(f"\nNext scan in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full Market Trading System")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=5, help="Minutes between scans")
    parser.add_argument("--execute", action="store_true", help="Execute trades")

    args = parser.parse_args()

    if args.execute:
        os.environ["EXECUTE_TRADES"] = "true"

    if args.continuous:
        run_continuous_trading(args.interval)
    else:
        run_full_market_trading()
