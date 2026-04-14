#!/usr/bin/env python3
"""
CROSS-ASSET ARBITRAGE ENGINE
============================

Multi-asset arbitrage across equities, crypto, FX, and derivatives.

Arbitrage Types:
- Spatial: Price differences across exchanges
- Temporal: Calendar spreads, roll yields
- Cross-Asset: BTC/USD vs BTC futures, gold spot vs GLD
- Statistical: Mean reversion, cointegration
- Triangular: FX crosses, crypto triangles

Author: MiniQuantFund Arbitrage Team
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading

import numpy as np
import pandas as pd
from scipy import stats

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT not available - crypto arbitrage disabled")

from mini_quant_fund.data.collectors.data_router import DataRouter
from mini_quant_fund.execution.alpaca_handler import AlpacaExecutionHandler

logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Types of arbitrage strategies."""
    SPATIAL = "spatial"  # Cross-exchange
    TEMPORAL = "temporal"  # Calendar spreads
    CROSS_ASSET = "cross_asset"  # Different assets
    STATISTICAL = "statistical"  # Mean reversion
    TRIANGULAR = "triangular"  # FX/crypto triangles


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity details."""
    id: str
    type: ArbitrageType
    legs: List[Dict]  # List of trades to execute
    expected_profit: float  # Expected profit in basis points
    confidence: float  # 0-1 confidence score
    holding_period: timedelta
    risk_factors: List[str]
    timestamp: datetime
    
    # Execution details
    buy_exchange: str
    sell_exchange: str
    symbol: str
    quantity: float
    buy_price: float
    sell_price: float


@dataclass
class TriangularArbPath:
    """Triangular arbitrage path (e.g., BTC/USD -> ETH/BTC -> ETH/USD)."""
    path: List[str]  # e.g., ["BTC/USD", "ETH/BTC", "ETH/USD"]
    rates: List[float]
    implied_rate: float
    actual_rate: float
    deviation_bps: float


class CrossExchangeArbitrage:
    """
    Monitor and execute cross-exchange arbitrage.
    
    Scans multiple exchanges for price discrepancies and executes
    simultaneous buy/sell orders to capture spreads.
    """
    
    def __init__(self, exchanges: List[str] = None):
        self.exchanges = exchanges or ['binance', 'coinbase', 'kraken']
        self.exchange_apis: Dict[str, any] = {}
        
        # Price cache
        self.price_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_update: Dict[str, datetime] = {}
        
        # Threading
        self._lock = threading.RLock()
        
        if CCXT_AVAILABLE:
            self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize exchange API connections."""
        for exchange_id in self.exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                self.exchange_apis[exchange_id] = exchange_class({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                    }
                })
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_id}: {e}")
    
    async def fetch_prices(self, symbol: str) -> Dict[str, float]:
        """
        Fetch current prices from all exchanges.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
        
        Returns:
            Dict of exchange -> price
        """
        if not CCXT_AVAILABLE:
            return {}
        
        prices = {}
        
        tasks = []
        for exchange_id, exchange in self.exchange_apis.items():
            task = self._fetch_single_price(exchange, symbol)
            tasks.append((exchange_id, task))
        
        # Gather results
        for exchange_id, task in tasks:
            try:
                price = await asyncio.wait_for(task, timeout=2.0)
                if price:
                    prices[exchange_id] = price
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {symbol} from {exchange_id}")
        
        with self._lock:
            self.price_cache[symbol] = prices
            self.last_update[symbol] = datetime.utcnow()
        
        return prices
    
    async def _fetch_single_price(self, exchange, symbol: str) -> Optional[float]:
        """Fetch price from single exchange."""
        try:
            ticker = await exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            return None
    
    def find_spatial_arbitrage(self, 
                               symbol: str,
                               min_spread_bps: float = 10.0,
                               min_profit_usd: float = 10.0) -> Optional[ArbitrageOpportunity]:
        """
        Find cross-exchange arbitrage opportunity.
        
        Args:
            symbol: Trading pair
            min_spread_bps: Minimum spread in basis points
            min_profit_usd: Minimum profit in USD
        
        Returns:
            ArbitrageOpportunity if found
        """
        with self._lock:
            prices = self.price_cache.get(symbol, {})
        
        if len(prices) < 2:
            return None
        
        # Find best buy (lowest ask) and sell (highest bid)
        buy_exchange = min(prices.items(), key=lambda x: x[1])
        sell_exchange = max(prices.items(), key=lambda x: x[1])
        
        spread = sell_exchange[1] - buy_exchange[1]
        spread_bps = (spread / buy_exchange[1]) * 10000
        
        if spread_bps < min_spread_bps:
            return None
        
        # Estimate profit (accounting for fees ~0.1% per leg)
        estimated_fees = buy_exchange[1] * 0.002 + sell_exchange[1] * 0.002
        net_profit = spread - estimated_fees
        
        if net_profit * 100 < min_profit_usd:  # Assuming 100 units
            return None
        
        return ArbitrageOpportunity(
            id=f"spatial_{symbol}_{datetime.utcnow().strftime('%H%M%S%f')}",
            type=ArbitrageType.SPATIAL,
            legs=[
                {'exchange': buy_exchange[0], 'side': 'buy', 'price': buy_exchange[1]},
                {'exchange': sell_exchange[0], 'side': 'sell', 'price': sell_exchange[1]}
            ],
            expected_profit=spread_bps,
            confidence=0.8,
            holding_period=timedelta(seconds=1),
            risk_factors=['execution_risk', 'transfer_risk'],
            timestamp=datetime.utcnow(),
            buy_exchange=buy_exchange[0],
            sell_exchange=sell_exchange[0],
            symbol=symbol,
            quantity=0.0,  # Would calculate based on available capital
            buy_price=buy_exchange[1],
            sell_price=sell_exchange[1]
        )


class TriangularArbitrage:
    """
    Detect and execute triangular arbitrage.
    
    Common in FX and crypto markets where three currency pairs
    form a triangle (e.g., BTC -> ETH -> USD -> BTC).
    """
    
    def __init__(self):
        self.base_currencies = ['USD', 'USDT', 'USDC', 'BTC', 'ETH']
        self.triangles: List[List[str]] = []
        self._build_triangles()
    
    def _build_triangles(self):
        """Build list of triangular paths."""
        # BTC/USD -> ETH/BTC -> ETH/USD triangle
        self.triangles = [
            ['BTC/USD', 'ETH/BTC', 'ETH/USD'],
            ['ETH/USD', 'BTC/ETH', 'BTC/USD'],
            ['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
            ['ETH/USDT', 'BTC/ETH', 'BTC/USDT'],
            # FX triangles
            ['EUR/USD', 'GBP/EUR', 'GBP/USD'],
            ['USD/JPY', 'EUR/USD', 'EUR/JPY'],
        ]
    
    def calculate_triangular_arbitrage(self,
                                       rates: Dict[str, float],
                                       triangle: List[str]) -> Optional[TriangularArbPath]:
        """
        Calculate arbitrage for a triangular path.
        
        Args:
            rates: Dict of symbol -> exchange rate
            triangle: List of 3 pairs forming triangle
        
        Returns:
            TriangularArbPath if arbitrage exists
        """
        if not all(pair in rates for pair in triangle):
            return None
        
        # Calculate implied rate through triangle
        implied_rate = 1.0
        for pair in triangle:
            rate = rates[pair]
            
            # Determine direction
            if pair.startswith(triangle[0].split('/')[1]):
                # Reverse pair (e.g., ETH/BTC when going BTC->ETH)
                implied_rate /= rate
            else:
                implied_rate *= rate
        
        # Compare to direct rate
        direct_pair = f"{triangle[0].split('/')[0]}/{triangle[-1].split('/')[1]}"
        actual_rate = rates.get(direct_pair, 0)
        
        if actual_rate == 0:
            return None
        
        # Calculate deviation
        deviation = (implied_rate - actual_rate) / actual_rate
        deviation_bps = deviation * 10000
        
        if abs(deviation_bps) < 5:  # Less than 5 bps
            return None
        
        return TriangularArbPath(
            path=triangle,
            rates=[rates[p] for p in triangle],
            implied_rate=implied_rate,
            actual_rate=actual_rate,
            deviation_bps=deviation_bps
        )


class StatisticalArbitrage:
    """
    Statistical arbitrage using mean reversion and cointegration.
    
    Pairs trading, basket trading, and factor-based arbitrage.
    """
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.pairs: Dict[Tuple[str, str], Dict] = {}
        self.data_router = DataRouter()
    
    def find_cointegrated_pairs(self, 
                                symbols: List[str],
                                pvalue_threshold: float = 0.05) -> List[Tuple[str, str]]:
        """
        Find cointegrated pairs using Engle-Granger test.
        
        Args:
            symbols: List of symbols to test
            pvalue_threshold: Maximum p-value for cointegration
        
        Returns:
            List of cointegrated pairs
        """
        cointegrated = []
        
        # Fetch historical prices
        prices = {}
        for symbol in symbols:
            try:
                hist = self.data_router.get_price_history(
                    symbol,
                    start_date=(datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
                )
                if not hist.empty:
                    prices[symbol] = hist['Close'].values
            except Exception as e:
                logger.debug(f"Failed to fetch {symbol}: {e}")
        
        # Test all pairs
        symbols_with_data = list(prices.keys())
        
        for i, sym1 in enumerate(symbols_with_data):
            for sym2 in symbols_with_data[i+1:]:
                if len(prices[sym1]) != len(prices[sym2]):
                    continue
                
                # Engle-Granger cointegration test
                score, pvalue, _ = statsmodels.tsa.stattools.coint(
                    prices[sym1], prices[sym2]
                ) if 'statsmodels' in sys.modules else (0, 1, None)
                
                if pvalue < pvalue_threshold:
                    cointegrated.append((sym1, sym2))
                    
                    # Calculate hedge ratio
                    hedge_ratio = np.polyfit(prices[sym2], prices[sym1], 1)[0]
                    
                    self.pairs[(sym1, sym2)] = {
                        'pvalue': pvalue,
                        'hedge_ratio': hedge_ratio,
                        'mean_spread': np.mean(prices[sym1] - hedge_ratio * prices[sym2]),
                        'std_spread': np.std(prices[sym1] - hedge_ratio * prices[sym2])
                    }
        
        return cointegrated
    
    def generate_pairs_signal(self, 
                             pair: Tuple[str, str],
                             current_prices: Dict[str, float]) -> Optional[Dict]:
        """
        Generate trading signal for a cointegrated pair.
        
        Returns:
            Signal dict with direction and z-score
        """
        if pair not in self.pairs:
            return None
        
        pair_data = self.pairs[pair]
        sym1, sym2 = pair
        
        if sym1 not in current_prices or sym2 not in current_prices:
            return None
        
        # Calculate current spread
        spread = current_prices[sym1] - pair_data['hedge_ratio'] * current_prices[sym2]
        
        # Calculate z-score
        zscore = (spread - pair_data['mean_spread']) / pair_data['std_spread']
        
        # Generate signal
        if abs(zscore) < 1.0:
            signal = 'NEUTRAL'
        elif zscore > 2.0:
            signal = 'SHORT_SPREAD'  # Spread too wide, expect mean reversion
        elif zscore < -2.0:
            signal = 'LONG_SPREAD'
        else:
            signal = 'NEUTRAL'
        
        return {
            'pair': pair,
            'signal': signal,
            'zscore': zscore,
            'spread': spread,
            'hedge_ratio': pair_data['hedge_ratio'],
            'sym1_price': current_prices[sym1],
            'sym2_price': current_prices[sym2]
        }


class CrossAssetArbitrageEngine:
    """
    Main arbitrage engine coordinating all strategies.
    """
    
    def __init__(self):
        self.spatial = CrossExchangeArbitrage()
        self.triangular = TriangularArbitrage()
        self.statistical = StatisticalArbitrage()
        
        # Opportunity queue
        self.opportunities: List[ArbitrageOpportunity] = []
        self._lock = threading.RLock()
        
        # Running state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start arbitrage monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Cross-asset arbitrage engine started")
    
    def stop(self):
        """Stop arbitrage monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                # Check spatial arbitrage opportunities
                crypto_pairs = ['BTC/USDT', 'ETH/USDT', 'BTC/USD', 'ETH/USD']
                
                for pair in crypto_pairs:
                    # Would use async properly in production
                    prices = {}  # await self.spatial.fetch_prices(pair)
                    
                    opp = self.spatial.find_spatial_arbitrage(pair)
                    if opp:
                        with self._lock:
                            self.opportunities.append(opp)
                
                # Clean old opportunities
                with self._lock:
                    cutoff = datetime.utcnow() - timedelta(minutes=5)
                    self.opportunities = [
                        o for o in self.opportunities 
                        if o.timestamp > cutoff
                    ]
                
            except Exception as e:
                logger.error(f"Arbitrage monitoring error: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def get_opportunities(self, 
                         min_profit_bps: float = 10.0) -> List[ArbitrageOpportunity]:
        """Get current arbitrage opportunities."""
        with self._lock:
            return [
                o for o in self.opportunities
                if o.expected_profit >= min_profit_bps
            ]
    
    def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Execute arbitrage trade.
        
        This would integrate with execution handlers to place
        simultaneous orders on multiple exchanges.
        """
        logger.info(f"Executing arbitrage: {opportunity.id}")
        
        # Placeholder - would implement actual execution
        # 1. Check balances on both exchanges
        # 2. Place buy order on cheaper exchange
        # 3. Place sell order on expensive exchange
        # 4. Handle transfer if needed
        
        return True
    
    def get_status(self) -> Dict:
        """Get arbitrage engine status."""
        with self._lock:
            return {
                "running": self._running,
                "active_opportunities": len(self.opportunities),
                "opportunities_by_type": defaultdict(int, {
                    o.type.value: 0 for o in self.opportunities
                }),
                "total_profit_bps": sum(o.expected_profit for o in self.opportunities)
            }


# Global engine instance
_arbitrage_engine: Optional[CrossAssetArbitrageEngine] = None


def get_arbitrage_engine() -> CrossAssetArbitrageEngine:
    """Get global arbitrage engine."""
    global _arbitrage_engine
    if _arbitrage_engine is None:
        _arbitrage_engine = CrossAssetArbitrageEngine()
    return _arbitrage_engine


if __name__ == "__main__":
    # Test arbitrage engine
    print("Testing Cross-Asset Arbitrage Engine...")
    
    engine = CrossAssetArbitrageEngine()
    
    # Test triangular calculation
    rates = {
        'BTC/USD': 45000.0,
        'ETH/BTC': 0.065,
        'ETH/USD': 2925.0
    }
    
    triangle = ['BTC/USD', 'ETH/BTC', 'ETH/USD']
    arb = engine.triangular.calculate_triangular_arbitrage(rates, triangle)
    
    if arb:
        print(f"\nTriangular Arbitrage Found:")
        print(f"  Path: {' -> '.join(arb.path)}")
        print(f"  Deviation: {arb.deviation_bps:.2f} bps")
    else:
        print("\nNo triangular arbitrage found")
    
    # Test spatial arbitrage
    engine.spatial.price_cache['BTC/USDT'] = {
        'binance': 45000.0,
        'coinbase': 45050.0
    }
    
    opp = engine.spatial.find_spatial_arbitrage('BTC/USDT')
    if opp:
        print(f"\nSpatial Arbitrage Found:")
        print(f"  Buy: {opp.buy_exchange} @ ${opp.buy_price:,.2f}")
        print(f"  Sell: {opp.sell_exchange} @ ${opp.sell_price:,.2f}")
        print(f"  Profit: {opp.expected_profit:.2f} bps")
    else:
        print("\nNo spatial arbitrage found")
