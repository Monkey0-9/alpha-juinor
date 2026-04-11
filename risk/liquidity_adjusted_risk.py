#!/usr/bin/env python3
"""
LIQUIDITY-ADJUSTED RISK SYSTEM
===============================

Institutional-grade liquidity-based risk management.
Replaces static position sizing with dynamic liquidity analysis.

Features:
- Order book depth analysis
- L-VaR (Liquidity VaR) calculations
- Market impact modeling
- Toxic flow detection
- Dynamic position sizing based on liquidity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy.stats import norm, gamma
import threading
import asyncio
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Real-time order book snapshot"""
    symbol: str
    timestamp: datetime
    
    # Bid side
    bid_prices: List[float] = field(default_factory=list)
    bid_sizes: List[float] = field(default_factory=list)
    
    # Ask side
    ask_prices: List[float] = field(default_factory=list)
    ask_sizes: List[float] = field(default_factory=list)
    
    # Liquidity metrics
    bid_depth: float = 0.0  # Total bid volume
    ask_depth: float = 0.0  # Total ask volume
    spread: float = 0.0
    spread_bps: float = 0.0
    
    # Market quality metrics
    imbalance_ratio: float = 0.0  # Bid/Ask volume imbalance
    concentration_ratio: float = 0.0  # Top 3 levels concentration
    micro_price: float = 0.0  # Volume-weighted mid price


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics"""
    symbol: str
    timestamp: datetime
    
    # Basic liquidity
    daily_volume: float = 0.0
    average_daily_volume: float = 0.0
    volume_participation_rate: float = 0.0
    
    # Liquidity depth
    bid_depth_10: float = 0.0  # Top 10 bid levels
    ask_depth_10: float = 0.0  # Top 10 ask levels
    depth_ratio: float = 0.0
    
    # Market impact
    temporary_impact: float = 0.0  # Temporary price impact
    permanent_impact: float = 0.0  # Permanent price impact
    total_impact: float = 0.0
    
    # Liquidity VaR
    l_var_95: float = 0.0  # 95% Liquidity VaR
    l_var_99: float = 0.0  # 99% Liquidity VaR
    
    # Toxic flow indicators
    flow_toxicity: float = 0.0  # VPIN-like measure
    aggressive_ratio: float = 0.0  # Aggressive vs passive orders
    hidden_liquidity_ratio: float = 0.0  # Estimated hidden liquidity


@dataclass
class LiquidityAdjustedPosition:
    """Position adjusted for liquidity constraints"""
    symbol: str
    desired_quantity: float
    liquidity_adjusted_quantity: float
    max_tradeable_size: float
    
    # Risk metrics
    market_impact_bps: float = 0.0
    liquidity_cost_bps: float = 0.0
    execution_time_estimate: float = 0.0  # Minutes
    
    # Constraints
    adv_limit_pct: float = 0.0
    depth_limit_pct: float = 0.0
    impact_limit_bps: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class LiquidityAnalyzer:
    """
    Advanced liquidity analysis and risk management
    """
    
    def __init__(self):
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.liquidity_metrics: Dict[str, LiquidityMetrics] = {}
        self.historical_volume: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        
        # Market impact models
        self.temporary_impact_model = TemporaryImpactModel()
        self.permanent_impact_model = PermanentImpactModel()
        self.toxic_flow_detector = ToxicFlowDetector()
        
        # Liquidity VaR calculator
        self.l_var_calculator = LiquidityVaRCalculator()
        
        # Threading
        self.analysis_lock = threading.Lock()
        self.is_running = False
        
        # Configuration
        self.max_adv_participation = 0.20  # Max 20% of ADV
        self.max_depth_participation = 0.10  # Max 10% of depth
        self.max_impact_bps = 50.0  # Max 50 bps impact
        
        logger.info("Liquidity Analyzer initialized with advanced models")
    
    def start(self):
        """Start liquidity analysis"""
        self.is_running = True
        threading.Thread(target=self._analysis_loop, daemon=True).start()
        logger.info("Liquidity analysis started")
    
    def stop(self):
        """Stop liquidity analysis"""
        self.is_running = False
        logger.info("Liquidity analysis stopped")
    
    def update_order_book(self, symbol: str, bid_prices: List[float], bid_sizes: List[float],
                         ask_prices: List[float], ask_sizes: List[float]) -> OrderBookSnapshot:
        """Update order book and calculate metrics"""
        with self.analysis_lock:
            try:
                # Create order book snapshot
                snapshot = OrderBookSnapshot(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    bid_prices=bid_prices,
                    bid_sizes=bid_sizes,
                    ask_prices=ask_prices,
                    ask_sizes=ask_sizes
                )
                
                # Calculate basic metrics
                snapshot.bid_depth = sum(bid_sizes)
                snapshot.ask_depth = sum(ask_sizes)
                
                if bid_prices and ask_prices:
                    snapshot.spread = ask_prices[0] - bid_prices[0]
                    snapshot.spread_bps = snapshot.spread / ((bid_prices[0] + ask_prices[0]) / 2) * 10000
                    snapshot.micro_price = self._calculate_micro_price(snapshot)
                
                # Calculate advanced metrics
                snapshot.imbalance_ratio = snapshot.bid_depth / (snapshot.bid_depth + snapshot.ask_depth) if (snapshot.bid_depth + snapshot.ask_depth) > 0 else 0.5
                snapshot.concentration_ratio = self._calculate_concentration_ratio(snapshot)
                
                # Store snapshot
                self.order_books[symbol] = snapshot
                
                # Update liquidity metrics
                self._update_liquidity_metrics(symbol, snapshot)
                
                return snapshot
                
            except Exception as e:
                logger.error(f"Order book update failed for {symbol}: {e}")
                return OrderBookSnapshot(symbol=symbol, timestamp=datetime.utcnow())
    
    def calculate_liquidity_adjusted_position(self, symbol: str, desired_quantity: float,
                                              urgency: str = "normal") -> LiquidityAdjustedPosition:
        """Calculate position size adjusted for liquidity constraints"""
        try:
            # Get current liquidity metrics
            if symbol not in self.liquidity_metrics:
                self._update_liquidity_metrics(symbol, self.order_books.get(symbol))
            
            metrics = self.liquidity_metrics.get(symbol)
            if not metrics:
                return LiquidityAdjustedPosition(
                    symbol=symbol,
                    desired_quantity=desired_quantity,
                    liquidity_adjusted_quantity=0.0,
                    max_tradeable_size=0.0
                )
            
            # Calculate constraints
            adv_limit = metrics.average_daily_volume * self.max_adv_participation
            depth_limit = min(metrics.bid_depth_10, metrics.ask_depth_10) * self.max_depth_participation
            
            # Market impact constraints
            impact_limit = self._calculate_impact_limit(symbol, desired_quantity, urgency)
            
            # Determine maximum tradeable size
            max_tradeable = min(adv_limit, depth_limit, impact_limit)
            
            # Adjust for urgency
            urgency_multiplier = {
                "urgent": 0.5,      # More conservative for urgent trades
                "normal": 1.0,
                "patient": 2.0      # Can be more patient
            }.get(urgency, 1.0)
            
            max_tradeable *= urgency_multiplier
            
            # Calculate market impact
            market_impact = self._calculate_market_impact(symbol, max_tradeable)
            liquidity_cost = self._calculate_liquidity_cost(symbol, max_tradeable)
            
            # Estimate execution time
            execution_time = self._estimate_execution_time(symbol, max_tradeable)
            
            # Final adjusted quantity
            adjusted_quantity = min(desired_quantity, max_tradeable)
            
            return LiquidityAdjustedPosition(
                symbol=symbol,
                desired_quantity=desired_quantity,
                liquidity_adjusted_quantity=adjusted_quantity,
                max_tradeable_size=max_tradeable,
                market_impact_bps=market_impact,
                liquidity_cost_bps=liquidity_cost,
                execution_time_estimate=execution_time,
                adv_limit_pct=(max_tradeable / metrics.average_daily_volume * 100) if metrics.average_daily_volume > 0 else 0,
                depth_limit_pct=(max_tradeable / min(metrics.bid_depth_10, metrics.ask_depth_10) * 100) if min(metrics.bid_depth_10, metrics.ask_depth_10) > 0 else 0,
                impact_limit_bps=self.max_impact_bps
            )
            
        except Exception as e:
            logger.error(f"Liquidity adjustment failed for {symbol}: {e}")
            return LiquidityAdjustedPosition(
                symbol=symbol,
                desired_quantity=desired_quantity,
                liquidity_adjusted_quantity=0.0,
                max_tradeable_size=0.0
            )
    
    def detect_toxic_flow(self, symbol: str, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Detect toxic flow patterns in trading data"""
        return self.toxic_flow_detector.analyze_flow(symbol, trades)
    
    def calculate_l_var(self, symbol: str, position_size: float, confidence: float = 0.99) -> float:
        """Calculate Liquidity VaR"""
        return self.l_var_calculator.calculate_l_var(symbol, position_size, confidence)
    
    def _calculate_micro_price(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate volume-weighted micro price"""
        if not snapshot.bid_prices or not snapshot.ask_prices:
            return 0.0
        
        # Weighted by volume
        total_volume = snapshot.bid_depth + snapshot.ask_depth
        if total_volume == 0:
            return (snapshot.bid_prices[0] + snapshot.ask_prices[0]) / 2
        
        bid_weight = snapshot.bid_depth / total_volume
        ask_weight = snapshot.ask_depth / total_volume
        
        return bid_weight * snapshot.bid_prices[0] + ask_weight * snapshot.ask_prices[0]
    
    def _calculate_concentration_ratio(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate concentration ratio (top 3 levels / total depth)"""
        bid_top3 = sum(snapshot.bid_sizes[:3]) if len(snapshot.bid_sizes) >= 3 else snapshot.bid_depth
        ask_top3 = sum(snapshot.ask_sizes[:3]) if len(snapshot.ask_sizes) >= 3 else snapshot.ask_depth
        
        total_depth = snapshot.bid_depth + snapshot.ask_depth
        if total_depth == 0:
            return 0.0
        
        return (bid_top3 + ask_top3) / total_depth
    
    def _update_liquidity_metrics(self, symbol: str, snapshot: OrderBookSnapshot):
        """Update comprehensive liquidity metrics"""
        try:
            if symbol not in self.liquidity_metrics:
                self.liquidity_metrics[symbol] = LiquidityMetrics(symbol=symbol, timestamp=datetime.utcnow())
            
            metrics = self.liquidity_metrics[symbol]
            metrics.timestamp = datetime.utcnow()
            
            # Update depth metrics
            metrics.bid_depth_10 = sum(snapshot.bid_sizes[:10]) if len(snapshot.bid_sizes) >= 10 else snapshot.bid_depth
            metrics.ask_depth_10 = sum(snapshot.ask_sizes[:10]) if len(snapshot.ask_sizes) >= 10 else snapshot.ask_depth
            metrics.depth_ratio = metrics.bid_depth_10 / (metrics.bid_depth_10 + metrics.ask_depth_10) if (metrics.bid_depth_10 + metrics.ask_depth_10) > 0 else 0.5
            
            # Update volume metrics (simplified - would use real data)
            current_volume = np.random.exponential(1000000)  # Simulate daily volume
            self.historical_volume[symbol].append(current_volume)
            
            metrics.daily_volume = current_volume
            metrics.average_daily_volume = np.mean(list(self.historical_volume[symbol]))
            
            # Calculate market impact
            test_size = 10000  # Test with 10k shares
            metrics.temporary_impact = self.temporary_impact_model.calculate_impact(symbol, test_size, snapshot)
            metrics.permanent_impact = self.permanent_impact_model.calculate_impact(symbol, test_size, snapshot)
            metrics.total_impact = metrics.temporary_impact + metrics.permanent_impact
            
            # Calculate L-VaR
            metrics.l_var_95 = self.calculate_l_var(symbol, test_size, 0.95)
            metrics.l_var_99 = self.calculate_l_var(symbol, test_size, 0.99)
            
            # Update toxic flow metrics
            metrics.flow_toxicity = self.toxic_flow_detector.get_current_toxicity(symbol)
            metrics.aggressive_ratio = np.random.beta(2, 3)  # Simulate aggressive ratio
            metrics.hidden_liquidity_ratio = np.random.uniform(0.1, 0.3)  # Estimate hidden liquidity
            
        except Exception as e:
            logger.error(f"Liquidity metrics update failed for {symbol}: {e}")
    
    def _calculate_impact_limit(self, symbol: str, desired_quantity: float, urgency: str) -> float:
        """Calculate position size limit based on impact constraints"""
        # Get current order book
        snapshot = self.order_books.get(symbol)
        if not snapshot:
            return desired_quantity
        
        # Calculate impact for different sizes
        impact_sizes = []
        test_sizes = np.linspace(1000, desired_quantity * 2, 10)
        
        for test_size in test_sizes:
            impact = self._calculate_market_impact(symbol, test_size)
            impact_sizes.append(impact)
        
        # Find size that meets impact constraint
        for i, impact in enumerate(impact_sizes):
            if impact > self.max_impact_bps:
                return test_sizes[i] * 0.8  # 80% of limit for safety
        
        return desired_quantity
    
    def _calculate_market_impact(self, symbol: str, quantity: float) -> float:
        """Calculate market impact in basis points"""
        snapshot = self.order_books.get(symbol)
        if not snapshot:
            return 0.0
        
        # Use both impact models
        temp_impact = self.temporary_impact_model.calculate_impact(symbol, quantity, snapshot)
        perm_impact = self.permanent_impact_model.calculate_impact(symbol, quantity, snapshot)
        
        return temp_impact + perm_impact
    
    def _calculate_liquidity_cost(self, symbol: str, quantity: float) -> float:
        """Calculate liquidity cost in basis points"""
        snapshot = self.order_books.get(symbol)
        if not snapshot or not snapshot.bid_prices or not snapshot.ask_prices:
            return 0.0
        
        # Estimate execution price based on order book depth
        mid_price = (snapshot.bid_prices[0] + snapshot.ask_prices[0]) / 2
        
        # Simulate walking the book
        remaining_qty = abs(quantity)
        execution_price = mid_price
        side = "bid" if quantity > 0 else "ask"
        
        if side == "bid":
            for i, (price, size) in enumerate(zip(snapshot.bid_prices, snapshot.bid_sizes)):
                if remaining_qty <= 0:
                    break
                fill_qty = min(remaining_qty, size)
                execution_price = execution_price * (1 - fill_qty / abs(quantity)) + price * (fill_qty / abs(quantity))
                remaining_qty -= fill_qty
        else:
            for i, (price, size) in enumerate(zip(snapshot.ask_prices, snapshot.ask_sizes)):
                if remaining_qty <= 0:
                    break
                fill_qty = min(remaining_qty, size)
                execution_price = execution_price * (1 - fill_qty / abs(quantity)) + price * (fill_qty / abs(quantity))
                remaining_qty -= fill_qty
        
        # Calculate cost in bps
        liquidity_cost = abs(execution_price - mid_price) / mid_price * 10000
        return liquidity_cost
    
    def _estimate_execution_time(self, symbol: str, quantity: float) -> float:
        """Estimate execution time in minutes"""
        metrics = self.liquidity_metrics.get(symbol)
        if not metrics:
            return 30.0  # Default 30 minutes
        
        # Base time on volume participation
        participation_rate = quantity / metrics.daily_volume
        base_time = participation_rate * 390  # Trading day in minutes
        
        # Adjust for liquidity depth
        depth_factor = min(metrics.bid_depth_10, metrics.ask_depth_10) / quantity
        adjusted_time = base_time / max(depth_factor, 0.1)
        
        return min(adjusted_time, 240)  # Cap at 4 hours
    
    def _analysis_loop(self):
        """Background analysis loop"""
        while self.is_running:
            try:
                # Update all symbols
                symbols_to_update = list(self.order_books.keys())
                
                for symbol in symbols_to_update:
                    snapshot = self.order_books.get(symbol)
                    if snapshot:
                        self._update_liquidity_metrics(symbol, snapshot)
                
                # Sleep for 1 minute
                asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Liquidity analysis loop error: {e}")
                asyncio.sleep(10)
    
    def get_liquidity_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive liquidity summary"""
        metrics = self.liquidity_metrics.get(symbol)
        snapshot = self.order_books.get(symbol)
        
        if not metrics or not snapshot:
            return {"error": "No liquidity data available"}
        
        return {
            "symbol": symbol,
            "timestamp": metrics.timestamp.isoformat(),
            "basic_metrics": {
                "daily_volume": metrics.daily_volume,
                "average_daily_volume": metrics.average_daily_volume,
                "bid_depth": snapshot.bid_depth,
                "ask_depth": snapshot.ask_depth,
                "spread_bps": snapshot.spread_bps,
                "micro_price": snapshot.micro_price
            },
            "depth_metrics": {
                "bid_depth_10": metrics.bid_depth_10,
                "ask_depth_10": metrics.ask_depth_10,
                "depth_ratio": metrics.depth_ratio,
                "concentration_ratio": snapshot.concentration_ratio
            },
            "impact_metrics": {
                "temporary_impact_bps": metrics.temporary_impact,
                "permanent_impact_bps": metrics.permanent_impact,
                "total_impact_bps": metrics.total_impact
            },
            "risk_metrics": {
                "l_var_95": metrics.l_var_95,
                "l_var_99": metrics.l_var_99,
                "flow_toxicity": metrics.flow_toxicity,
                "aggressive_ratio": metrics.aggressive_ratio,
                "hidden_liquidity_ratio": metrics.hidden_liquidity_ratio
            }
        }


class TemporaryImpactModel:
    """Temporary market impact model (Almgren-Chriss)"""
    
    def calculate_impact(self, symbol: str, quantity: float, snapshot: OrderBookSnapshot) -> float:
        """Calculate temporary impact in bps"""
        # Simplified Almgren-Chriss model
        daily_volume = 1000000  # Would use real ADV
        volume_rate = quantity / daily_volume
        
        # Temporary impact proportional to square root of participation rate
        temporary_impact = 0.1 * np.sqrt(volume_rate) * 10000  # Convert to bps
        
        # Adjust for order book depth
        depth_factor = min(snapshot.bid_depth, snapshot.ask_depth) / quantity
        temporary_impact /= max(depth_factor, 0.1)
        
        return min(temporary_impact, 25.0)  # Cap at 25 bps


class PermanentImpactModel:
    """Permanent market impact model"""
    
    def calculate_impact(self, symbol: str, quantity: float, snapshot: OrderBookSnapshot) -> float:
        """Calculate permanent impact in bps"""
        # Linear impact model
        daily_volume = 1000000  # Would use real ADV
        volume_rate = quantity / daily_volume
        
        # Permanent impact proportional to participation rate
        permanent_impact = 0.05 * volume_rate * 10000  # Convert to bps
        
        return min(permanent_impact, 10.0)  # Cap at 10 bps


class ToxicFlowDetector:
    """Detect toxic flow patterns using VPIN-like measures"""
    
    def __init__(self):
        self.flow_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.vpin_window = 50  # VPIN calculation window
    
    def analyze_flow(self, symbol: str, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze flow for toxicity indicators"""
        if not trades:
            return {"toxicity": 0.0, "vpin": 0.0, "aggressive_ratio": 0.0}
        
        # Calculate VPIN (Volume-Synchronized Probability of Informed Trading)
        volume_buy = sum(t["size"] for t in trades if t["side"] == "buy")
        volume_sell = sum(t["size"] for t in trades if t["side"] == "sell")
        total_volume = volume_buy + volume_sell
        
        if total_volume == 0:
            return {"toxicity": 0.0, "vpin": 0.0, "aggressive_ratio": 0.0}
        
        # VPIN calculation
        abs_volume_imbalance = abs(volume_buy - volume_sell)
        vpin = abs_volume_imbalance / total_volume
        
        # Aggressive ratio (market orders vs limit orders)
        aggressive_trades = sum(1 for t in trades if t.get("aggressive", False))
        aggressive_ratio = aggressive_trades / len(trades)
        
        # Toxicity score (combination of VPIN and aggressive ratio)
        toxicity = (vpin * 0.7 + aggressive_ratio * 0.3)
        
        # Store in history
        self.flow_history[symbol].append({
            "vpin": vpin,
            "aggressive_ratio": aggressive_ratio,
            "toxicity": toxicity,
            "timestamp": datetime.utcnow()
        })
        
        return {
            "toxicity": toxicity,
            "vpin": vpin,
            "aggressive_ratio": aggressive_ratio,
            "volume_imbalance": abs_volume_imbalance / total_volume
        }
    
    def get_current_toxicity(self, symbol: str) -> float:
        """Get current toxicity score"""
        history = self.flow_history.get(symbol)
        if not history:
            return 0.0
        
        return history[-1]["toxicity"]


class LiquidityVaRCalculator:
    """Liquidity Value at Risk calculator"""
    
    def calculate_l_var(self, symbol: str, position_size: float, confidence: float = 0.99) -> float:
        """Calculate Liquidity VaR in bps"""
        # Simplified L-VaR calculation
        # In production, would use bid-ask spread and depth data
        
        # Base L-VaR from spread
        spread_bps = 10.0  # Would use real spread
        spread_var = spread_bps * position_size / 10000
        
        # Add depth adjustment
        depth_multiplier = 1.5  # Adjust for limited depth
        
        # Confidence level adjustment
        confidence_multiplier = norm.ppf(confidence)
        
        l_var = spread_var * depth_multiplier * confidence_multiplier * 10000  # Convert to bps
        
        return min(l_var, 100.0)  # Cap at 100 bps


# Global liquidity analyzer instance
_liquidity_instance = None

def get_liquidity_analyzer() -> LiquidityAnalyzer:
    """Get global liquidity analyzer instance"""
    global _liquidity_instance
    if _liquidity_instance is None:
        _liquidity_instance = LiquidityAnalyzer()
    return _liquidity_instance


if __name__ == "__main__":
    # Test liquidity analyzer
    analyzer = LiquidityAnalyzer()
    analyzer.start()
    
    # Test order book update
    test_order_book = analyzer.update_order_book(
        symbol="AAPL",
        bid_prices=[174.5, 174.4, 174.3, 174.2, 174.1],
        bid_sizes=[1000, 2000, 1500, 3000, 2500],
        ask_prices=[174.6, 174.7, 174.8, 174.9, 175.0],
        ask_sizes=[1200, 1800, 2200, 2800, 3200]
    )
    
    # Test liquidity adjustment
    adjusted_position = analyzer.calculate_liquidity_adjusted_position("AAPL", 50000)
    
    # Get summary
    summary = analyzer.get_liquidity_summary("AAPL")
    print(json.dumps(summary, indent=2, default=str))
