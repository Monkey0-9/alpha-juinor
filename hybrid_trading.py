#!/usr/bin/env python3
"""
=============================================================================
NEXUS INSTITUTIONAL - HYBRID TRADING SYSTEM
=============================================================================
Combines:
  1. News/Event Trading (Tier 1) - 60-second reaction time
  2. High-Frequency Trading (HFT) - <100 microsecond execution

This dual-mode system can run both strategies simultaneously:
- News trading: Identifies opportunities from market events
- HFT: Executes liquidity provision and micro-arbitrage at speed

Usage:
  python hybrid_trading.py --mode paper --hft enabled --news enabled
  python hybrid_trading.py --mode paper --hft enabled --news disabled
  python hybrid_trading.py --mode paper --hft disabled --news enabled
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nexus.institutional.live_monitor import LiveTradingMonitor
from src.nexus.institutional.hft_engine import HFTEngine, HFTStrategy
from src.nexus.core.context import engine_context


class HybridTradingSystem:
    """
    Hybrid trading system combining news/sentiment and HFT.
    
    Architecture:
    ┌─────────────────┬──────────────────┐
    │  News Trading   │   HFT Engine     │
    │  (60 sec)       │   (<100 μs)      │
    ├─────────────────┼──────────────────┤
    │ • News feeds    │ • Market making  │
    │ • Sentiment     │ • Latency arb    │
    │ • Events        │ • Stat arb       │
    │ • Risk mgmt     │ • Sub-tick trade │
    └─────────────────┴──────────────────┘
            ↓               ↓
         Portfolio Manager & Risk Enforcer
                     ↓
              Combined Order Stream
                     ↓
           Execution Engine (Live/Paper)
    """
    
    def __init__(self, 
                 news_enabled: bool = True,
                 hft_enabled: bool = True,
                 news_interval: int = 60,
                 hft_update_interval_ms: int = 1):
        """
        Initialize hybrid trading system.
        
        Args:
            news_enabled: Enable news/sentiment trading
            hft_enabled: Enable HFT strategies
            news_interval: News update frequency (seconds)
            hft_update_interval_ms: HFT cycle frequency (milliseconds)
        """
        self.news_enabled = news_enabled
        self.hft_enabled = hft_enabled
        self.news_interval = news_interval
        self.hft_update_interval_ms = hft_update_interval_ms
        
        # Initialize components
        self.news_monitor = LiveTradingMonitor(update_interval=news_interval) if news_enabled else None
        self.hft_engine = HFTEngine(
            enabled_strategies=[
                HFTStrategy.MARKET_MAKING,
                HFTStrategy.LATENCY_ARBITRAGE,
                HFTStrategy.STATISTICAL_ARBITRAGE
            ]
        ) if hft_enabled else None
        
        # Portfolio tracking
        self.portfolio = {
            'cash': 1000000,
            'positions': {},
            'value': 1000000
        }
        
        # Combined signals
        self.news_signals = []
        self.hft_signals = []
        
        self.running = False
        
        logger = logging.getLogger("HybridTrading")
    
    async def run_hybrid_system(self, duration_seconds: Optional[int] = None):
        """Run hybrid trading system - news and HFT working together."""
        
        logger = logging.getLogger("HybridTrading")
        
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*15 + "NEXUS INSTITUTIONAL - HYBRID TRADING SYSTEM" + " "*21 + "║")
        logger.info("║" + " "*10 + "News/Sentiment Trading + High-Frequency Trading" + " "*22 + "║")
        logger.info("╚" + "="*78 + "╝")
        logger.info("")
        logger.info("System Components:")
        logger.info(f"  News Trading:    {'✅ ENABLED' if self.news_enabled else '❌ DISABLED'}")
        logger.info(f"  HFT Engine:      {'✅ ENABLED' if self.hft_enabled else '❌ DISABLED'}")
        logger.info("")
        logger.info("Configuration:")
        logger.info(f"  News Interval:   {self.news_interval}s")
        logger.info(f"  HFT Cycles:      ~{1000/self.hft_update_interval_ms:.0f} per second")
        logger.info("")
        
        self.running = True
        
        # Create concurrent tasks
        tasks = []
        
        if self.news_enabled:
            tasks.append(self._run_news_trading(duration_seconds))
        
        if self.hft_enabled:
            tasks.append(self._run_hft_trading(duration_seconds))
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except KeyboardInterrupt:
            logger.info("\nHybrid system interrupted by user")
        finally:
            self.shutdown()
    
    async def _run_news_trading(self, duration_seconds: Optional[int] = None):
        """Run news/sentiment trading component."""
        logger = logging.getLogger("NewsTrading")
        logger.info("Starting News/Sentiment Trading Module...")
        
        if self.news_monitor:
            await self.news_monitor.run_monitoring_loop(duration_seconds=duration_seconds)
    
    async def _run_hft_trading(self, duration_seconds: Optional[int] = None):
        """Run HFT component."""
        logger = logging.getLogger("HFTTrading")
        logger.info("Starting HFT Engine Module...")
        
        if self.hft_engine:
            # Simulate order book updates for demo
            async def simulate_books():
                import random
                while self.running:
                    for symbol in ['SPY', 'QQQ', 'IWM']:
                        mid = 100 + random.gauss(0, 1)
                        await self.hft_engine.update_order_book(
                            symbol,
                            bid_prices=[mid - 0.01, mid - 0.02],
                            bid_sizes=[100, 200],
                            ask_prices=[mid + 0.01, mid + 0.02],
                            ask_sizes=[100, 200]
                        )
                    await asyncio.sleep(0.01)
            
            await asyncio.gather(
                self.hft_engine.run_hft_loop(duration_seconds=duration_seconds),
                simulate_books(),
                return_exceptions=True
            )
    
    def shutdown(self):
        """Shutdown hybrid system gracefully."""
        logger = logging.getLogger("HybridTrading")
        logger.info("")
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*20 + "HYBRID SYSTEM SHUTDOWN" + " "*36 + "║")
        logger.info("╚" + "="*78 + "╝")
        
        if self.news_enabled and self.news_monitor:
            logger.info(f"\nNews Trading Stats:")
            logger.info(f"  Alerts Generated: {len(self.news_monitor.alerts_generated)}")
            logger.info(f"  Trades Executed: {len(self.news_monitor.trades_executed)}")
        
        if self.hft_enabled and self.hft_engine:
            metrics = self.hft_engine.get_metrics()
            logger.info(f"\nHFT Engine Stats:")
            logger.info(f"  Total Cycles: {metrics['cycles']}")
            logger.info(f"  Total Trades: {metrics['trades']}")
            logger.info(f"  Avg Latency: {metrics['avg_latency_us']:.2f} μs")
            logger.info(f"  Max Latency: {metrics['max_latency_us']:.2f} μs")
            logger.info(f"  P&L: ${metrics['pnl']:,.2f}")
        
        logger.info(f"\nFinal Portfolio Value: ${self.portfolio['value']:,.2f}")
        
        self.running = False


def main():
    """Main entry point for hybrid trading system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Nexus Institutional - Hybrid Trading System (News + HFT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both news trading and HFT
  python hybrid_trading.py --mode paper
  
  # News trading only
  python hybrid_trading.py --mode paper --hft disabled
  
  # HFT only
  python hybrid_trading.py --mode paper --news disabled
  
  # Run for 1 hour
  python hybrid_trading.py --duration 3600
  
  # Verbose logging
  python hybrid_trading.py --log-level DEBUG
"""
    )
    
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Execution mode (default: paper - no real money)"
    )
    
    parser.add_argument(
        "--news",
        choices=["enabled", "disabled"],
        default="enabled",
        help="Enable/disable news/sentiment trading"
    )
    
    parser.add_argument(
        "--hft",
        choices=["enabled", "disabled"],
        default="enabled",
        help="Enable/disable HFT engine"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        help="Run for N seconds"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("Main")
    
    # Initialize engine
    engine_context.initialize("config/production.yaml")
    
    # Create and run system
    news_enabled = args.news == "enabled"
    hft_enabled = args.hft == "enabled"
    
    system = HybridTradingSystem(
        news_enabled=news_enabled,
        hft_enabled=hft_enabled
    )
    
    try:
        asyncio.run(system.run_hybrid_system(duration_seconds=args.duration))
    except KeyboardInterrupt:
        logger.info("Shutdown initiated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
