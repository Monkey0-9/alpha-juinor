"""
=============================================================================
NEXUS INSTITUTIONAL - HIGH-FREQUENCY TRADING (HFT) MODULE
=============================================================================
Ultra-low latency trading: <100 microsecond execution targeting
Combines news sentiment trading with HFT strategies
Designed for: Market-making, latency arbitrage, stat arb at speed

Features:
- Sub-second order response times
- Multiple HFT strategies (market-making, arbitrage, statistical)
- Ultra-low latency optimization (target <100μs)
- Real-time order book monitoring
- Automated liquidity provision
- Volatility-based position sizing
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger("HFTEngine")


class HFTStrategy(Enum):
    """High-frequency trading strategies."""
    MARKET_MAKING = "market_making"           # Provide liquidity, earn spread
    LATENCY_ARBITRAGE = "latency_arbitrage"   # Exploit venue delays (<10ms)
    STATISTICAL_ARBITRAGE = "stat_arb"        # Pairs trading at speed
    MOMENTUM_HFT = "momentum_hft"              # Capture micro-trends
    MEAN_REVERSION_HFT = "mean_reversion_hft" # Fast mean reversion


class OrderType(Enum):
    """Order types for HFT."""
    LIMIT = "limit"
    MARKET = "market"
    PEGGED = "pegged"  # Price pegging for market-making


class OrderSide(Enum):
    """Buy/Sell sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class HFTOrder:
    """Ultra-low latency order structure."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    order_type: OrderType
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_us: float = 0.0  # Microseconds to execute
    filled_quantity: float = 0.0
    status: str = "pending"  # pending, partial, filled, cancelled
    
    def to_dict(self):
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "type": self.order_type.value,
            "execution_time_us": self.execution_time_us,
            "filled": self.filled_quantity,
            "status": self.status,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OrderBook:
    """Real-time order book snapshot."""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Bid side (buy orders)
    bid_prices: List[float] = field(default_factory=list)
    bid_sizes: List[float] = field(default_factory=list)
    
    # Ask side (sell orders)
    ask_prices: List[float] = field(default_factory=list)
    ask_sizes: List[float] = field(default_factory=list)
    
    def get_spread_bps(self) -> float:
        """Get bid-ask spread in basis points."""
        if not self.bid_prices or not self.ask_prices:
            return 0.0
        spread = (self.ask_prices[0] - self.bid_prices[0]) / self.bid_prices[0]
        return spread * 10000  # Convert to bps
    
    def get_mid_price(self) -> float:
        """Get mid price."""
        if not self.bid_prices or not self.ask_prices:
            return 0.0
        return (self.bid_prices[0] + self.ask_prices[0]) / 2
    
    def get_micro_trend(self) -> float:
        """Detect micro trend (0=neutral, 1=up, -1=down)."""
        if len(self.bid_prices) < 2 or len(self.ask_prices) < 2:
            return 0.0
        
        # Compare top of book movement
        bid_change = (self.bid_prices[0] - self.bid_prices[-1]) if len(self.bid_prices) > 1 else 0
        ask_change = (self.ask_prices[0] - self.ask_prices[-1]) if len(self.ask_prices) > 1 else 0
        
        trend = (bid_change + ask_change) / (abs(bid_change) + abs(ask_change) + 0.001)
        return trend
    
    def to_dict(self):
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid_prices[0] if self.bid_prices else None,
            "bid_size": self.bid_sizes[0] if self.bid_sizes else None,
            "ask": self.ask_prices[0] if self.ask_prices else None,
            "ask_size": self.ask_sizes[0] if self.ask_sizes else None,
            "mid": self.get_mid_price(),
            "spread_bps": self.get_spread_bps()
        }


class MarketMakingStrategy:
    """Market-making HFT strategy - provide liquidity, earn spread."""
    
    def __init__(self, target_spread_bps: float = 2.0, max_position_size: float = 100):
        """
        Initialize market-making strategy.
        
        Args:
            target_spread_bps: Target spread in basis points
            max_position_size: Maximum position in shares
        """
        self.target_spread_bps = target_spread_bps
        self.max_position_size = max_position_size
        self.position = 0.0
        self.pnl = 0.0
    
    async def generate_orders(self, order_book: OrderBook) -> List[HFTOrder]:
        """Generate market-making orders."""
        orders = []
        
        mid_price = order_book.get_mid_price()
        if mid_price == 0:
            return orders
        
        # Calculate bid/ask prices around mid
        spread_price = mid_price * (self.target_spread_bps / 10000)
        bid_price = mid_price - spread_price / 2
        ask_price = mid_price + spread_price / 2
        
        # Inventory management: reduce size if long/short
        base_quantity = 50
        bid_quantity = max(1, base_quantity - int(self.position * 0.5))
        ask_quantity = max(1, base_quantity + int(self.position * 0.5))
        
        # Generate bid order (buy)
        if self.position < self.max_position_size:
            bid_order = HFTOrder(
                order_id=f"MM_BID_{int(time.time() * 1e6)}",
                symbol=order_book.symbol,
                side=OrderSide.BUY,
                quantity=bid_quantity,
                price=bid_price,
                order_type=OrderType.LIMIT
            )
            orders.append(bid_order)
        
        # Generate ask order (sell)
        if self.position > -self.max_position_size:
            ask_order = HFTOrder(
                order_id=f"MM_ASK_{int(time.time() * 1e6)}",
                symbol=order_book.symbol,
                side=OrderSide.SELL,
                quantity=ask_quantity,
                price=ask_price,
                order_type=OrderType.LIMIT
            )
            orders.append(ask_order)
        
        return orders


class LatencyArbitrageStrategy:
    """Latency arbitrage - exploit venue delays (<10ms)."""
    
    def __init__(self, latency_threshold_ms: float = 5.0, position_limit: float = 50):
        """
        Initialize latency arbitrage.
        
        Args:
            latency_threshold_ms: Minimum latency to exploit
            position_limit: Max position size
        """
        self.latency_threshold_ms = latency_threshold_ms
        self.position_limit = position_limit
        self.last_price_time = {}
        self.price_history = {}
    
    async def detect_arbitrage(self, order_books: Dict[str, OrderBook]) -> List[HFTOrder]:
        """
        Detect latency arbitrage opportunities.
        
        Works by: observing price differences between venues
                  if one venue hasn't updated yet from news,
                  we can arbitrage the difference.
        """
        orders = []
        
        if len(order_books) < 2:
            return orders
        
        symbols = list(order_books.keys())
        for i, symbol_a in enumerate(symbols):
            for symbol_b in symbols[i+1:]:
                book_a = order_books[symbol_a]
                book_b = order_books[symbol_b]
                
                if not book_a.bid_prices or not book_b.ask_prices:
                    continue
                
                # Check if symbol_a mid > symbol_b mid (potential arb)
                mid_a = book_a.get_mid_price()
                mid_b = book_b.get_mid_price()
                
                if mid_a == 0 or mid_b == 0:
                    continue
                
                price_diff_bps = abs(mid_a - mid_b) / mid_b * 10000
                
                # If spread is wide enough to arbitrage (>3bps after fees)
                if price_diff_bps > 3:
                    if mid_a > mid_b:
                        # Sell expensive, buy cheap
                        orders.append(HFTOrder(
                            order_id=f"ARB_SELL_{int(time.time() * 1e6)}",
                            symbol=symbol_a,
                            side=OrderSide.SELL,
                            quantity=10,
                            price=book_a.ask_prices[0],
                            order_type=OrderType.MARKET
                        ))
                        orders.append(HFTOrder(
                            order_id=f"ARB_BUY_{int(time.time() * 1e6)}",
                            symbol=symbol_b,
                            side=OrderSide.BUY,
                            quantity=10,
                            price=book_b.bid_prices[0],
                            order_type=OrderType.MARKET
                        ))
        
        return orders


class StatisticalArbitrageStrategy:
    """Statistical arbitrage - pair trading at high frequency."""
    
    def __init__(self, lookback_ticks: int = 100, zscore_threshold: float = 2.0):
        """
        Initialize stat arb strategy.
        
        Args:
            lookback_ticks: How many price updates to track
            zscore_threshold: Z-score threshold for entry
        """
        self.lookback_ticks = lookback_ticks
        self.zscore_threshold = zscore_threshold
        self.price_spreads = deque(maxlen=lookback_ticks)
    
    async def generate_signals(self, order_books: Dict[str, OrderBook]) -> List[HFTOrder]:
        """Generate stat arb signals."""
        orders = []
        
        if len(order_books) < 2:
            return orders
        
        # Example: trade SPY/QQQ pair
        if 'SPY' not in order_books or 'QQQ' not in order_books:
            return orders
        
        spy_book = order_books['SPY']
        qqq_book = order_books['QQQ']
        
        spy_mid = spy_book.get_mid_price()
        qqq_mid = qqq_book.get_mid_price()
        
        if spy_mid == 0 or qqq_mid == 0:
            return orders
        
        # Calculate spread (simple ratio)
        spread = spy_mid / qqq_mid
        self.price_spreads.append(spread)
        
        if len(self.price_spreads) < 10:
            return orders
        
        # Calculate z-score
        spreads_list = list(self.price_spreads)
        mean_spread = sum(spreads_list) / len(spreads_list)
        variance = sum((x - mean_spread) ** 2 for x in spreads_list) / len(spreads_list)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return orders
        
        zscore = (spread - mean_spread) / std_dev
        
        # Mean reversion signal
        if zscore > self.zscore_threshold:
            # Spread too high, SPY relatively expensive
            # Buy QQQ, sell SPY
            orders.append(HFTOrder(
                order_id=f"STAT_BUY_{int(time.time() * 1e6)}",
                symbol='QQQ',
                side=OrderSide.BUY,
                quantity=20,
                price=qqq_book.bid_prices[0],
                order_type=OrderType.LIMIT
            ))
            orders.append(HFTOrder(
                order_id=f"STAT_SELL_{int(time.time() * 1e6)}",
                symbol='SPY',
                side=OrderSide.SELL,
                quantity=20,
                price=spy_book.ask_prices[0],
                order_type=OrderType.LIMIT
            ))
        
        elif zscore < -self.zscore_threshold:
            # Spread too low, SPY relatively cheap
            # Buy SPY, sell QQQ
            orders.append(HFTOrder(
                order_id=f"STAT_BUY_{int(time.time() * 1e6)}",
                symbol='SPY',
                side=OrderSide.BUY,
                quantity=20,
                price=spy_book.bid_prices[0],
                order_type=OrderType.LIMIT
            ))
            orders.append(HFTOrder(
                order_id=f"STAT_SELL_{int(time.time() * 1e6)}",
                symbol='QQQ',
                side=OrderSide.SELL,
                quantity=20,
                price=qqq_book.ask_prices[0],
                order_type=OrderType.LIMIT
            ))
        
        return orders


class HFTEngine:
    """Main HFT engine - orchestrates high-frequency trading."""
    
    def __init__(self, enabled_strategies: List[HFTStrategy] = None):
        """
        Initialize HFT engine.
        
        Args:
            enabled_strategies: List of strategies to run
        """
        self.enabled_strategies = enabled_strategies or [
            HFTStrategy.MARKET_MAKING,
            HFTStrategy.LATENCY_ARBITRAGE,
            HFTStrategy.STATISTICAL_ARBITRAGE
        ]
        
        # Initialize strategies
        self.market_maker = MarketMakingStrategy()
        self.latency_arb = LatencyArbitrageStrategy()
        self.stat_arb = StatisticalArbitrageStrategy()
        
        # Order tracking
        self.orders_placed = deque(maxlen=10000)
        self.orders_filled = deque(maxlen=10000)
        self.order_latencies = deque(maxlen=1000)
        
        # Performance metrics
        self.hft_pnl = 0.0
        self.hft_trades = 0
        self.avg_latency_us = 0.0
        self.max_latency_us = 0.0
        
        # Order books
        self.order_books = {}
        
        self.running = False
    
    async def update_order_book(self, symbol: str, bid_prices: List[float], 
                               bid_sizes: List[float], ask_prices: List[float], 
                               ask_sizes: List[float]):
        """Update order book for a symbol."""
        self.order_books[symbol] = OrderBook(
            symbol=symbol,
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes
        )
    
    async def process_cycle(self) -> List[HFTOrder]:
        """
        Process one HFT cycle.
        This should run as fast as possible (<1ms target).
        """
        orders = []
        start_time_us = time.time_ns() / 1000  # Convert to microseconds
        
        # Market-making
        if HFTStrategy.MARKET_MAKING in self.enabled_strategies:
            for symbol, book in self.order_books.items():
                mm_orders = await self.market_maker.generate_orders(book)
                orders.extend(mm_orders)
        
        # Latency arbitrage
        if HFTStrategy.LATENCY_ARBITRAGE in self.enabled_strategies:
            arb_orders = await self.latency_arb.detect_arbitrage(self.order_books)
            orders.extend(arb_orders)
        
        # Statistical arbitrage
        if HFTStrategy.STATISTICAL_ARBITRAGE in self.enabled_strategies:
            stat_orders = await self.stat_arb.generate_signals(self.order_books)
            orders.extend(stat_orders)
        
        # Track latency
        end_time_us = time.time_ns() / 1000
        cycle_latency_us = end_time_us - start_time_us
        
        self.order_latencies.append(cycle_latency_us)
        if self.order_latencies:
            self.avg_latency_us = sum(self.order_latencies) / len(self.order_latencies)
            self.max_latency_us = max(self.order_latencies)
        
        # Record orders
        for order in orders:
            order.execution_time_us = cycle_latency_us
            self.orders_placed.append(order)
            self.hft_trades += 1
        
        return orders
    
    async def run_hft_loop(self, duration_seconds: Optional[int] = None):
        """
        Run continuous HFT loop.
        Target: Execute one cycle every 1-10 milliseconds (100-10 microseconds execution time)
        """
        logger.info("="*80)
        logger.info("STARTING HFT ENGINE - HIGH-FREQUENCY TRADING")
        logger.info("="*80)
        logger.info(f"Enabled Strategies: {[s.value for s in self.enabled_strategies]}")
        logger.info(f"Target Latency: <100 microseconds per cycle")
        logger.info("")
        
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration_seconds) if duration_seconds else None
        cycle_count = 0
        
        try:
            while self.running:
                if end_time and datetime.now() > end_time:
                    break
                
                cycle_count += 1
                
                # Process one HFT cycle
                orders = await self.process_cycle()
                
                # Log every 100 cycles
                if cycle_count % 100 == 0:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"HFT CYCLE #{cycle_count}")
                    logger.info(f"{'='*80}")
                    logger.info(f"Orders Generated: {len(orders)}")
                    logger.info(f"Total Trades: {self.hft_trades}")
                    logger.info(f"Avg Latency: {self.avg_latency_us:.1f} μs")
                    logger.info(f"Max Latency: {self.max_latency_us:.1f} μs")
                    logger.info(f"P&L: ${self.hft_pnl:,.2f}")
                    
                    # Show top orders
                    if orders:
                        logger.info(f"\nLast Orders Generated:")
                        for order in orders[-3:]:
                            logger.info(f"  {order.symbol} {order.side.value} {order.quantity} @ {order.price:.2f}")
                
                # Sleep for next cycle (target 1-10ms per cycle)
                # This allows ~100-1000 cycles per second
                await asyncio.sleep(0.001)  # 1ms between cycles
        
        except KeyboardInterrupt:
            logger.info("\nHFT Engine interrupted by user")
        except Exception as e:
            logger.error(f"Error in HFT loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown HFT engine gracefully."""
        logger.info("")
        logger.info("="*80)
        logger.info("SHUTTING DOWN HFT ENGINE")
        logger.info("="*80)
        
        uptime = datetime.now()
        logger.info(f"Total Cycles: {len(self.orders_placed)}")
        logger.info(f"Total Trades: {self.hft_trades}")
        logger.info(f"Avg Latency: {self.avg_latency_us:.2f} μs (TARGET: <100 μs)")
        logger.info(f"Max Latency: {self.max_latency_us:.2f} μs")
        logger.info(f"Final P&L: ${self.hft_pnl:,.2f}")
        
        self.running = False
    
    def get_metrics(self) -> Dict:
        """Get current HFT metrics."""
        return {
            "cycles": len(self.orders_placed),
            "trades": self.hft_trades,
            "avg_latency_us": self.avg_latency_us,
            "max_latency_us": self.max_latency_us,
            "pnl": self.hft_pnl,
            "strategies": [s.value for s in self.enabled_strategies]
        }


async def demo_hft():
    """Demo HFT engine with simulated order books."""
    
    # Create engine with all strategies
    engine = HFTEngine(enabled_strategies=[
        HFTStrategy.MARKET_MAKING,
        HFTStrategy.LATENCY_ARBITRAGE,
        HFTStrategy.STATISTICAL_ARBITRAGE
    ])
    
    # Simulate order book updates
    async def simulate_order_books():
        """Simulate live order book updates."""
        import random
        
        while engine.running:
            # Simulate SPY order book
            spy_mid = 500.0 + random.gauss(0, 0.5)
            await engine.update_order_book(
                'SPY',
                bid_prices=[spy_mid - 0.01, spy_mid - 0.02],
                bid_sizes=[100, 200],
                ask_prices=[spy_mid + 0.01, spy_mid + 0.02],
                ask_sizes=[100, 200]
            )
            
            # Simulate QQQ order book
            qqq_mid = 380.0 + random.gauss(0, 0.4)
            await engine.update_order_book(
                'QQQ',
                bid_prices=[qqq_mid - 0.01, qqq_mid - 0.02],
                bid_sizes=[100, 200],
                ask_prices=[qqq_mid + 0.01, qqq_mid + 0.02],
                ask_sizes=[100, 200]
            )
            
            await asyncio.sleep(0.01)  # Update every 10ms
    
    # Run HFT and order book simulation concurrently
    try:
        await asyncio.gather(
            engine.run_hft_loop(duration_seconds=60),  # Run for 60 seconds
            simulate_order_books(),
            return_exceptions=True
        )
    except Exception as e:
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Nexus HFT Engine - High-Frequency Trading")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--strategies", nargs="+", help="Strategies to enable")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_hft())
