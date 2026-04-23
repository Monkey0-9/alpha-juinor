"""
Async Multi-Source Data Pipeline - Elite Quant Fund System
Features: Kalman Filter, Yang-Zhang Volatility, Amihud Illiquidity
Built to Renaissance Technologies / Jane Street standards
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
import numpy as np
import pandas as pd
from numpy.linalg import inv

from elite_quant_fund.core.types import (
    MarketBar, Quote, Trade, Result, KalmanState, VolatilityState,
    LiquidityMetrics
)

logger = logging.getLogger(__name__)


# ============================================================================
# KALMAN FILTER - Online Price Estimation with Adaptive Noise
# ============================================================================

class AdaptiveKalmanFilter:
    """
    1-D Kalman filter with online EM noise adaptation
    Optimal for filtering high-frequency price data
    """
    
    def __init__(
        self,
        symbol: str,
        initial_price: float,
        initial_Q: float = 0.001,  # Process noise
        initial_R: float = 0.01,   # Measurement noise
        adaptation_window: int = 100
    ):
        self.symbol = symbol
        self.Q = initial_Q
        self.R = initial_R
        self.adaptation_window = adaptation_window
        
        # Innovation history for EM adaptation
        self.innovations: deque = deque(maxlen=adaptation_window)
        self.residuals: deque = deque(maxlen=adaptation_window)
        
        # Initialize state
        self.state = KalmanState(
            symbol=symbol,
            timestamp=datetime.now(),
            x_hat=initial_price,
            P=1.0,
            Q=initial_Q,
            R=initial_R,
            K=0.5
        )
    
    def update(self, measurement: float, timestamp: datetime) -> KalmanState:
        """
        Update filter with new measurement using online EM adaptation
        """
        # Prediction
        state_pred = self.state.predict()
        
        # Innovation
        y = measurement - state_pred.x_hat
        S = state_pred.P + self.R
        
        # Kalman gain
        K = state_pred.P / S
        
        # Update
        x_new = state_pred.x_hat + K * y
        P_new = (1 - K) * state_pred.P
        
        # Store for adaptation
        self.innovations.append(y ** 2)
        residual = measurement - x_new
        self.residuals.append(residual ** 2)
        
        # Online EM adaptation (every 20 observations)
        if len(self.innovations) >= 20 and len(self.innovations) % 20 == 0:
            self._adapt_noise()
        
        self.state = KalmanState(
            symbol=self.symbol,
            timestamp=timestamp,
            x_hat=x_new,
            P=P_new,
            Q=self.Q,
            R=self.R,
            K=K
        )
        
        return self.state
    
    def _adapt_noise(self) -> None:
        """Online EM noise adaptation"""
        if len(self.innovations) < 20:
            return
        
        # Update R (measurement noise) from residuals
        R_new = np.mean(list(self.residuals))
        
        # Update Q (process noise) from innovations
        innovation_var = np.mean(list(self.innovations))
        # Q = max(0, innovation_var - R) with smoothing
        Q_new = max(1e-10, innovation_var - R_new)
        
        # Smooth updates
        self.R = 0.9 * self.R + 0.1 * R_new
        self.Q = 0.9 * self.Q + 0.1 * Q_new
    
    def get_filtered_price(self) -> float:
        """Return current filtered price estimate"""
        return self.state.x_hat
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Return confidence interval around estimate"""
        z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%
        margin = z_score * np.sqrt(self.state.P)
        return (self.state.x_hat - margin, self.state.x_hat + margin)


# ============================================================================
# YANG-ZHANG VOLATILITY - Minimum Variance Estimator
# ============================================================================

class YangZhangVolatility:
    """
    Yang-Zhang minimum variance volatility estimator
    Combines overnight, open-to-close, and intraday information
    More efficient than close-to-close estimator
    """
    
    def __init__(self, symbol: str, window: int = 20):
        self.symbol = symbol
        self.window = window
        
        # Data storage
        self.opens: deque = deque(maxlen=window)
        self.highs: deque = deque(maxlen=window)
        self.lows: deque = deque(maxlen=window)
        self.closes: deque = deque(maxlen=window)
        self.prev_closes: deque = deque(maxlen=window)
        
        self.current_state: Optional[VolatilityState] = None
    
    def update(self, bar: MarketBar) -> VolatilityState:
        """Update volatility estimate with new bar"""
        
        # Store data
        if self.prev_closes:
            self.opens.append(bar.open)
            self.highs.append(bar.high)
            self.lows.append(bar.low)
            self.closes.append(bar.close)
            self.prev_closes.append(self.closes[-2] if len(self.closes) > 1 else bar.open)
        else:
            # First observation
            self.opens.append(bar.open)
            self.highs.append(bar.high)
            self.lows.append(bar.low)
            self.closes.append(bar.close)
            self.prev_closes.append(bar.open)
        
        if len(self.closes) < 2:
            # Return default until we have enough data
            return VolatilityState(
                symbol=self.symbol,
                timestamp=bar.timestamp,
                volatility_annual=0.20,
                overnight_vol=0.01,
                open_close_vol=0.01
            )
        
        # Calculate components
        overnight_vol = self._overnight_volatility()
        open_close_vol = self._open_close_volatility()
        intraday_vol = self._intraday_volatility()
        
        # Yang-Zhang combination (minimum variance)
        # sigma_yz^2 = overnight_vol^2 + k * open_close_vol^2 + (1-k) * intraday_vol^2
        k = 0.34  # Optimal weight from Yang-Zhang paper
        
        total_var = (
            overnight_vol ** 2 +
            k * open_close_vol ** 2 +
            (1 - k) * intraday_vol ** 2
        )
        
        daily_vol = np.sqrt(total_var)
        annual_vol = daily_vol * np.sqrt(252)
        
        self.current_state = VolatilityState(
            symbol=self.symbol,
            timestamp=bar.timestamp,
            volatility_annual=annual_vol,
            overnight_vol=overnight_vol,
            open_close_vol=open_close_vol,
            min_variance_estimator="yang_zhang"
        )
        
        return self.current_state
    
    def _overnight_volatility(self) -> float:
        """Overnight volatility (close-to-open)"""
        if len(self.opens) < 2:
            return 0.01
        
        returns = np.log(np.array(self.opens[1:]) / np.array(list(self.closes)[:-1]))
        return np.std(returns, ddof=1) if len(returns) > 1 else 0.01
    
    def _open_close_volatility(self) -> float:
        """Open-to-close volatility"""
        if len(self.opens) < 2:
            return 0.01
        
        returns = np.log(np.array(self.closes) / np.array(self.opens))
        return np.std(returns, ddof=1) if len(returns) > 1 else 0.01
    
    def _intraday_volatility(self) -> float:
        """Rogers-Satchell intraday volatility"""
        if len(self.highs) < 2:
            return 0.01
        
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        opens = np.array(self.opens)
        closes = np.array(self.closes)
        
        # Rogers-Satchell estimator
        rs = (
            np.log(highs / closes) * np.log(highs / opens) +
            np.log(lows / closes) * np.log(lows / opens)
        )
        
        return np.sqrt(np.mean(rs)) if len(rs) > 0 else 0.01


# ============================================================================
# AMIHUD ILLIQUIDITY RATIO
# ============================================================================

class AmihudIlliquidity:
    """
    Amihud (2002) illiquidity ratio
    ILLIQ = mean(|return| / dollar_volume)
    Higher values = less liquid
    """
    
    def __init__(self, symbol: str, window: int = 20):
        self.symbol = symbol
        self.window = window
        self.returns: deque = deque(maxlen=window)
        self.dollar_volumes: deque = deque(maxlen=window)
    
    def update(self, bar: MarketBar) -> float:
        """Update Amihud ratio"""
        if len(self.returns) > 0:
            ret = abs(bar.returns)
            dollar_vol = bar.volume * bar.vwap if bar.vwap else bar.volume * bar.close
            
            self.returns.append(ret)
            self.dollar_volumes.append(dollar_vol)
        else:
            self.returns.append(0.0)
            self.dollar_volumes.append(bar.volume * bar.close)
        
        if len(self.returns) < 5:
            return 0.0
        
        ratios = [
            r / dv if dv > 0 else 0
            for r, dv in zip(self.returns, self.dollar_volumes)
        ]
        
        return np.mean(ratios)


# ============================================================================
# LIQUIDITY ESTIMATOR
# ============================================================================

class LiquidityEstimator:
    """
    Real-time liquidity estimation combining multiple metrics
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.amihud = AmihudIlliquidity(symbol)
        self.quotes: deque = deque(maxlen=100)
        self.trades: deque = deque(maxlen=100)
        
        # Kyle's lambda estimation
        self.price_changes: deque = deque(maxlen=50)
        self.signed_volumes: deque = deque(maxlen=50)
    
    def update_bar(self, bar: MarketBar) -> LiquidityMetrics:
        """Update liquidity metrics from bar data"""
        amihud_ratio = self.amihud.update(bar)
        
        # Estimate Kyle's lambda from stored trades
        kyle_lambda = self._estimate_kyle_lambda()
        
        # Bid-ask spread from recent quotes
        spread_bps = self._estimate_spread()
        
        # Market depth estimate
        market_depth = self._estimate_depth()
        
        # ADV estimate (simplified)
        adv = bar.volume * 1.5  # Assume today is representative
        
        return LiquidityMetrics(
            symbol=self.symbol,
            timestamp=bar.timestamp,
            bid_ask_spread_bps=spread_bps,
            amihud_ratio=amihud_ratio,
            kyle_lambda=kyle_lambda,
            market_depth_dollars=market_depth,
            adv_20_day=adv
        )
    
    def update_quote(self, quote: Quote) -> None:
        """Store quote for spread estimation"""
        self.quotes.append(quote)
    
    def update_trade(self, trade: Trade) -> None:
        """Store trade for Kyle's lambda estimation"""
        self.trades.append(trade)
        
        if len(self.trades) >= 2:
            # Calculate price change and signed volume
            prev_trade = list(self.trades)[-2]
            price_change = trade.price - prev_trade.price
            
            # Sign volume (simplified - assume trade side if available)
            if trade.side:
                signed_vol = trade.size if trade.side.value == 'buy' else -trade.size
            else:
                # Use tick rule
                signed_vol = trade.size if price_change > 0 else (-trade.size if price_change < 0 else 0)
            
            self.price_changes.append(price_change)
            self.signed_volumes.append(signed_vol)
    
    def _estimate_kyle_lambda(self) -> float:
        """Estimate Kyle's lambda (price impact per unit volume)"""
        if len(self.price_changes) < 10:
            return 0.0
        
        # Kyle's lambda = Cov(delta_p, q) / Var(q)
        p_changes = np.array(self.price_changes)
        volumes = np.array(self.signed_volumes)
        
        if np.var(volumes) == 0:
            return 0.0
        
        return np.cov(p_changes, volumes)[0, 1] / np.var(volumes)
    
    def _estimate_spread(self) -> float:
        """Estimate bid-ask spread in basis points"""
        if not self.quotes:
            return 10.0  # Default 10 bps
        
        recent_quotes = list(self.quotes)[-20:]
        spreads = [q.spread_bps for q in recent_quotes]
        return np.median(spreads) if spreads else 10.0
    
    def _estimate_depth(self) -> float:
        """Estimate market depth in dollars"""
        if not self.quotes:
            return 1_000_000  # Default $1M
        
        recent = list(self.quotes)[-5:]
        avg_bid_size = np.mean([q.bid_size for q in recent])
        avg_ask_size = np.mean([q.ask_size for q in recent])
        avg_mid = np.mean([q.mid for q in recent])
        
        return (avg_bid_size + avg_ask_size) * avg_mid


# ============================================================================
# DATA SOURCE ABSTRACTION
# ============================================================================

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    async def connect(self) -> Result[bool]:
        """Connect to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> Result[bool]:
        """Disconnect from data source"""
        pass
    
    @abstractmethod
    async def subscribe_bars(
        self,
        symbols: List[str],
        callback: Callable[[MarketBar], None]
    ) -> Result[bool]:
        """Subscribe to bar updates"""
        pass
    
    @abstractmethod
    async def subscribe_quotes(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None]
    ) -> Result[bool]:
        """Subscribe to quote updates"""
        pass
    
    @abstractmethod
    async def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1Min'
    ) -> Result[List[MarketBar]]:
        """Get historical bars"""
        pass
    
    @property
    @abstractmethod
    def latency_ms(self) -> float:
        """Typical latency in milliseconds"""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority for source selection (lower = higher priority)"""
        pass


# ============================================================================
# ASYNC DATA PIPELINE MANAGER
# ============================================================================

class DataPipeline:
    """
    Async multi-source data pipeline with automatic failover
    and 6-sigma spike detection
    """
    
    def __init__(
        self,
        symbols: List[str],
        primary_source: DataSource,
        fallback_sources: Optional[List[DataSource]] = None
    ):
        self.symbols = symbols
        self.primary_source = primary_source
        self.fallback_sources = fallback_sources or []
        
        # Sort by priority
        all_sources = [primary_source] + self.fallback_sources
        self.sources = sorted(all_sources, key=lambda s: s.priority)
        
        # Active source
        self.active_source_idx = 0
        
        # Filter states per symbol
        self.kalman_filters: Dict[str, AdaptiveKalmanFilter] = {}
        self.volatility_estimators: Dict[str, YangZhangVolatility] = {}
        self.liquidity_estimators: Dict[str, LiquidityEstimator] = {}
        
        # Bar history for spike detection
        self.bar_history: Dict[str, deque] = {sym: deque(maxlen=100) for sym in symbols}
        
        # Callbacks
        self.bar_callbacks: List[Callable[[MarketBar], None]] = []
        self.quote_callbacks: List[Callable[[Quote], None]] = []
        self.filtered_price_callbacks: List[Callable[[str, float, datetime], None]] = []
        
        # Stats
        self.bars_processed = 0
        self.spikes_detected = 0
        self.failover_count = 0
        
        # Running flag
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_bar_callback(self, callback: Callable[[MarketBar], None]) -> None:
        """Register callback for bar updates"""
        self.bar_callbacks.append(callback)
    
    def register_quote_callback(self, callback: Callable[[Quote], None]) -> None:
        """Register callback for quote updates"""
        self.quote_callbacks.append(callback)
    
    def register_filtered_price_callback(
        self,
        callback: Callable[[str, float, datetime], None]
    ) -> None:
        """Register callback for Kalman-filtered prices"""
        self.filtered_price_callbacks.append(callback)
    
    def _init_filters(self, symbol: str, initial_price: float) -> None:
        """Initialize filters for a symbol"""
        if symbol not in self.kalman_filters:
            self.kalman_filters[symbol] = AdaptiveKalmanFilter(symbol, initial_price)
            self.volatility_estimators[symbol] = YangZhangVolatility(symbol)
            self.liquidity_estimators[symbol] = LiquidityEstimator(symbol)
    
    def _detect_spike(self, bar: MarketBar) -> bool:
        """
        6-sigma spike detection
        Returns True if bar contains anomalous price movement
        """
        history = self.bar_history.get(bar.symbol)
        if not history or len(history) < 20:
            return False
        
        # Calculate statistics from history
        returns = [b.returns for b in history]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return False
        
        # Check if current return is > 6 sigma
        z_score = abs(bar.returns - mean_return) / std_return
        
        return z_score > 6.0
    
    def _process_bar(self, bar: MarketBar) -> Result[MarketBar]:
        """
        Process bar through filters and validation
        """
        # Initialize filters if needed
        self._init_filters(bar.symbol, bar.close)
        
        # Spike detection
        if self._detect_spike(bar):
            self.spikes_detected += 1
            logger.warning(f"6-sigma spike detected in {bar.symbol}: {bar.returns:.4f}")
            # Use filtered price instead
            filtered_price = self.kalman_filters[bar.symbol].get_filtered_price()
            bar = bar.model_copy(update={'close': filtered_price})
        
        # Update Kalman filter
        kalman_state = self.kalman_filters[bar.symbol].update(bar.close, bar.timestamp)
        
        # Update volatility
        vol_state = self.volatility_estimators[bar.symbol].update(bar)
        
        # Update liquidity
        liq_metrics = self.liquidity_estimators[bar.symbol].update_bar(bar)
        
        # Store in history
        self.bar_history[bar.symbol].append(bar)
        
        self.bars_processed += 1
        
        # Notify callbacks
        for callback in self.bar_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")
        
        for callback in self.filtered_price_callbacks:
            try:
                callback(bar.symbol, kalman_state.x_hat, bar.timestamp)
            except Exception as e:
                logger.error(f"Filtered price callback error: {e}")
        
        return Result.ok(bar)
    
    async def start(self) -> Result[bool]:
        """Start the data pipeline"""
        logger.info(f"Starting data pipeline for {len(self.symbols)} symbols")
        
        # Connect to primary source
        result = await self.primary_source.connect()
        if result.is_err:
            logger.error(f"Failed to connect to primary source: {result}")
            return Result.err(f"Connection failed: {result}")
        
        self._running = True
        
        # Subscribe to data
        result = await self.primary_source.subscribe_bars(
            self.symbols,
            lambda bar: self._process_bar(bar)
        )
        
        if result.is_err:
            logger.error(f"Failed to subscribe: {result}")
            return result
        
        logger.info("Data pipeline started successfully")
        return Result.ok(True)
    
    async def stop(self) -> Result[bool]:
        """Stop the data pipeline"""
        logger.info("Stopping data pipeline")
        self._running = False
        
        # Disconnect from all sources
        for source in self.sources:
            await source.disconnect()
        
        logger.info("Data pipeline stopped")
        return Result.ok(True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'bars_processed': self.bars_processed,
            'spikes_detected': self.spikes_detected,
            'failover_count': self.failover_count,
            'active_source': self.sources[self.active_source_idx].__class__.__name__,
            'symbols': len(self.symbols),
            'filters_initialized': len(self.kalman_filters)
        }
    
    def get_kalman_state(self, symbol: str) -> Optional[KalmanState]:
        """Get current Kalman filter state for symbol"""
        if symbol in self.kalman_filters:
            return self.kalman_filters[symbol].state
        return None
    
    def get_volatility(self, symbol: str) -> Optional[VolatilityState]:
        """Get current volatility estimate for symbol"""
        if symbol in self.volatility_estimators:
            return self.volatility_estimators[symbol].current_state
        return None


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'AdaptiveKalmanFilter',
    'YangZhangVolatility',
    'AmihudIlliquidity',
    'LiquidityEstimator',
    'DataSource',
    'DataPipeline',
]
