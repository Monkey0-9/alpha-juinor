"""
Elite Quant Fund Trading System - Main Orchestrator
Integrates: Data Pipeline, Alpha Engine, Risk Engine, Portfolio Optimizer, Execution Engine
Built to Renaissance Technologies / Jane Street standards
"""

import asyncio
import logging
import signal
import sys
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from elite_quant_fund.core.types import (
    MarketBar, AlphaSignal, Portfolio, Position, Order, Fill,
    RiskLimits, RiskBreach, TargetAllocation, SignalBundle,
    Side, OrderType, Venue, Result
)
from elite_quant_fund.data.pipeline import DataPipeline, DataSource
from elite_quant_fund.alpha.engine import AlphaEngine
from elite_quant_fund.risk.engine import RiskEngine
from elite_quant_fund.portfolio.optimizer import PortfolioOptimizer
from elite_quant_fund.execution.algo import ExecutionEngine

logger = logging.getLogger(__name__)


# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """Elite Quant Fund system configuration"""
    
    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX'
    ])
    
    # Risk limits
    risk_limits: RiskLimits = field(default_factory=lambda: RiskLimits(
        max_position_value=1_000_000,
        max_leverage=2.0,
        max_drawdown_pct=0.05,
        max_cvar_95=0.02,
        max_sector_concentration=0.25,
        kelly_fraction=0.3,
        kill_switch_drawdown=0.10
    ))
    
    # Portfolio optimization
    optimization_method: str = 'black_litterman'
    target_volatility: float = 0.10
    rebalance_threshold: float = 0.05  # Rebalance if drift > 5%
    
    # Execution
    default_order_type: OrderType = OrderType.ALMGREN_CHRISS
    max_participation_pct: float = 0.10
    
    # Timing
    bar_interval_seconds: int = 60
    portfolio_update_interval_seconds: int = 300  # 5 minutes
    
    # Data
    data_source: Optional[DataSource] = None


# ============================================================================
# SYSTEM STATE
# ============================================================================

class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


# ============================================================================
# MAIN ELITE QUANT FUND SYSTEM
# ============================================================================

class EliteQuantFund:
    """
    Main Elite Quant Fund Trading System
    
    This system integrates all components:
    - Data Pipeline: Real-time market data with Kalman filtering
    - Alpha Engine: Multi-model signal generation (OU, Factor, ML)
    - Risk Engine: CVaR, Kelly sizing, kill switch
    - Portfolio Optimizer: Black-Litterman, Risk Parity
    - Execution Engine: Almgren-Chriss, VWAP, Smart Order Router
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = SystemState.INITIALIZING
        
        # Initialize components
        logger.info("Initializing Elite Quant Fund System...")
        
        # Data Pipeline
        if config.data_source:
            self.data_pipeline = DataPipeline(
                symbols=config.symbols,
                primary_source=config.data_source
            )
        else:
            self.data_pipeline = None
        
        # Alpha Engine
        self.alpha_engine = AlphaEngine(config.symbols)
        
        # Risk Engine
        self.risk_engine = RiskEngine(config.risk_limits)
        
        # Portfolio Optimizer
        self.portfolio_optimizer = PortfolioOptimizer(
            symbols=config.symbols,
            method=config.optimization_method,
            target_volatility=config.target_volatility
        )
        
        # Execution Engine
        self.execution_engine = ExecutionEngine()
        
        # Portfolio state
        self.portfolio = Portfolio(
            timestamp=datetime.now(),
            positions={},
            cash=10_000_000,  # $10M starting capital
            total_value=10_000_000
        )
        
        # Signal storage
        self.latest_signals: Dict[str, AlphaSignal] = {}
        self.latest_allocation: Optional[TargetAllocation] = None
        
        # Pending orders
        self.pending_orders: Dict[str, Order] = {}
        self.executed_orders: List[Order] = []
        
        # Running flag
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            'bars_processed': 0,
            'signals_generated': 0,
            'orders_placed': 0,
            'orders_filled': 0,
            'total_volume_traded': 0.0,
            'total_pnl': 0.0,
            'start_time': None,
            'risk_breaches': 0
        }
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info("Elite Quant Fund System initialized")
    
    def _register_callbacks(self) -> None:
        """Register component callbacks"""
        
        # Alpha signal callback
        self.alpha_engine.register_signal_callback(self._on_alpha_signal)
        
        # Risk breach callback
        self.risk_engine.register_breach_handler(self._on_risk_breach)
        
        # Execution callback
        self.execution_engine.register_slice_callback(self._on_execution_slice)
    
    def _on_alpha_signal(self, signal: AlphaSignal) -> None:
        """Handle new alpha signal"""
        self.latest_signals[signal.symbol] = signal
        self.stats['signals_generated'] += 1
        
        logger.debug(f"Alpha signal: {signal.symbol} {signal.signal_type.name} "
                    f"strength={signal.strength:.3f}")
    
    def _on_risk_breach(self, breach: RiskBreach) -> None:
        """Handle risk breach"""
        self.stats['risk_breaches'] += 1
        
        logger.warning(f"RISK BREACH: {breach.breach_type.name} - {breach.description}")
        
        if breach.is_critical:
            logger.critical("Critical risk breach - system may need to halt trading")
            # In production, this would trigger emergency procedures
    
    def _on_execution_slice(self, order_id: str, quantity: int, venue: Venue) -> None:
        """Handle execution slice"""
        logger.debug(f"Execution slice: order={order_id}, qty={quantity}, venue={venue.name}")
    
    async def start(self) -> Result[bool]:
        """Start the trading system"""
        
        logger.info("="*70)
        logger.info("ELITE QUANT FUND SYSTEM - STARTING")
        logger.info("="*70)
        logger.info(f"Symbols: {len(self.config.symbols)}")
        logger.info(f"Method: {self.config.optimization_method}")
        logger.info(f"Target Vol: {self.config.target_volatility:.1%}")
        logger.info(f"Kelly Fraction: {self.config.risk_limits.kelly_fraction}")
        logger.info("="*70)
        
        self.state = SystemState.RUNNING
        self._running = True
        self.stats['start_time'] = datetime.now()
        
        # Start data pipeline if available
        if self.data_pipeline:
            result = await self.data_pipeline.start()
            if result.is_err:
                logger.error(f"Failed to start data pipeline: {result}")
                self.state = SystemState.ERROR
                return result
        
        # Initial portfolio optimization
        await self._rebalance_portfolio()
        
        logger.info("System started successfully")
        
        # Start main loop
        await self._main_loop()
        
        return Result.ok(True)
    
    async def stop(self) -> Result[bool]:
        """Stop the trading system"""
        
        logger.info("Stopping Elite Quant Fund System...")
        
        self.state = SystemState.SHUTTING_DOWN
        self._running = False
        self._shutdown_event.set()
        
        # Stop data pipeline
        if self.data_pipeline:
            await self.data_pipeline.stop()
        
        self.state = SystemState.STOPPED
        
        # Print final stats
        self._print_final_stats()
        
        return Result.ok(True)
    
    async def _main_loop(self) -> None:
        """Main trading loop"""
        
        logger.info("Entering main trading loop...")
        
        cycle_count = 0
        last_portfolio_update = datetime.now()
        
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    cycle_count += 1
                    now = datetime.now()
                    
                    # Process any pending data
                    if self.data_pipeline:
                        # Data pipeline runs asynchronously, signals processed via callbacks
                        pass
                    else:
                        # Simulate bar processing for backtesting
                        await self._simulate_market_data()
                    
                    # Periodic portfolio rebalancing
                    time_since_update = (now - last_portfolio_update).total_seconds()
                    if time_since_update >= self.config.portfolio_update_interval_seconds:
                        if self.risk_engine.can_trade():
                            await self._rebalance_portfolio()
                            last_portfolio_update = now
                        else:
                            logger.warning("Kill switch active - skipping rebalance")
                    
                    # Check risk continuously
                    self._update_risk()
                    
                    # Sleep
                    await asyncio.sleep(self.config.bar_interval_seconds)
                
                except asyncio.CancelledError:
                    logger.info("Main loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(10)
        
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
            self.state = SystemState.ERROR
    
    async def _simulate_market_data(self) -> None:
        """
        Simulate market data for backtesting/demo
        In production, this would be replaced by real data feeds
        """
        
        for symbol in self.config.symbols:
            # Generate synthetic bar
            # In production, this comes from data pipeline
            import random
            
            base_price = 100.0
            if symbol in self.portfolio.positions:
                base_price = self.portfolio.positions[symbol].current_price
            
            # Random walk
            drift = 0.0001  # Slight upward drift
            vol = 0.01  # 1% daily vol
            
            ret = np.random.normal(drift, vol)
            price = base_price * (1 + ret)
            
            bar = MarketBar(
                symbol=symbol,
                timestamp=datetime.now(),
                open=price * 0.999,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=int(np.random.uniform(100000, 1000000))
            )
            
            await self._process_bar(bar)
    
    async def _process_bar(self, bar: MarketBar) -> None:
        """Process market bar through the system"""
        
        self.stats['bars_processed'] += 1
        
        # Update portfolio positions with new prices
        if bar.symbol in self.portfolio.positions:
            position = self.portfolio.positions[bar.symbol]
            updated_pos = position.update_price(bar.close)
            self.portfolio.positions[bar.symbol] = updated_pos
            
            # Update total portfolio value
            self._update_portfolio_value()
        
        # Update covariance estimator
        if bar.symbol in self.portfolio.positions:
            ret = bar.returns
            self.portfolio_optimizer.update_returns(bar.symbol, ret)
            self.risk_engine.add_return(bar.symbol, ret)
        
        # Generate alpha signals
        signal_bundle = self.alpha_engine.process_bar(bar)
        
        # Store signals
        for signals in signal_bundle.signals.values():
            for signal in signals:
                self.latest_signals[signal.symbol] = signal
    
    def _update_portfolio_value(self) -> None:
        """Recalculate total portfolio value"""
        
        positions_value = sum(
            p.market_value for p in self.portfolio.positions.values()
        )
        
        total_pnl = sum(
            p.unrealized_pnl + p.realized_pnl 
            for p in self.portfolio.positions.values()
        )
        
        self.portfolio = self.portfolio.model_copy(update={
            'total_value': self.portfolio.cash + positions_value,
            'timestamp': datetime.now()
        })
        
        self.stats['total_pnl'] = total_pnl
    
    def _update_risk(self) -> None:
        """Update risk metrics and check limits"""
        
        breaches = self.risk_engine.update_portfolio(self.portfolio)
        
        if breaches:
            for breach in breaches:
                logger.warning(f"Risk breach: {breach.breach_type.name}")
    
    async def _rebalance_portfolio(self) -> None:
        """Rebalance portfolio to target allocation"""
        
        logger.info("Rebalancing portfolio...")
        
        # Generate target allocation
        allocation = self.portfolio_optimizer.optimize(self.latest_signals)
        
        if allocation is None:
            logger.warning("Portfolio optimization failed - skipping rebalance")
            return
        
        self.latest_allocation = allocation
        
        logger.info(f"Target allocation generated: {allocation.method}")
        logger.info(f"Expected return: {allocation.expected_return:.2%}")
        logger.info(f"Expected vol: {allocation.expected_volatility:.2%}")
        logger.info(f"Sharpe: {allocation.sharpe_ratio:.2f}")
        
        # Calculate rebalancing orders
        orders = self.portfolio_optimizer.rebalance_orders(
            self.portfolio,
            allocation
        )
        
        if not orders:
            logger.info("No rebalancing needed")
            return
        
        # Execute orders
        for symbol, quantity in orders.items():
            await self._place_order(symbol, quantity)
    
    async def _place_order(self, symbol: str, quantity: int) -> Optional[Order]:
        """Place order through risk check and execution"""
        
        side = Side.BUY if quantity > 0 else Side.SELL
        qty = abs(quantity)
        
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type=self.config.default_order_type,
            max_participation_pct=self.config.max_participation_pct
        )
        
        # Pre-trade risk check
        risk_result = self.risk_engine.pre_trade_check(order, self.portfolio)
        
        if risk_result.is_err:
            logger.warning(f"Order blocked by risk: {risk_result}")
            return None
        
        # Calculate optimal position size (Kelly criterion)
        expected_return = 0.05  # 5% expected return
        volatility = 0.20  # 20% volatility
        
        optimal_size = self.risk_engine.calculate_optimal_size(
            symbol, expected_return, volatility, self.portfolio
        )
        
        # Adjust order size
        if qty * 100 > optimal_size:  # Assume $100 price
            logger.warning(f"Order size {qty} exceeds Kelly optimal {optimal_size/100:.0f}")
            # Still proceed, but logged
        
        # Create execution schedule
        volatility = 0.20  # Get from data pipeline
        schedule = self.execution_engine.create_schedule(
            order,
            volatility=volatility
        )
        
        logger.info(f"Order placed: {symbol} {side.name} {qty} shares")
        logger.info(f"Execution schedule: {len(schedule.schedule)} slices, "
                   f"impact={schedule.expected_impact_bps:.2f} bps")
        
        self.pending_orders[order.id] = order
        self.stats['orders_placed'] += 1
        
        return order
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        runtime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        
        return {
            'state': self.state.value,
            'runtime_hours': runtime.total_seconds() / 3600,
            'portfolio_value': self.portfolio.total_value,
            'portfolio_pnl': self.stats['total_pnl'],
            'positions': len(self.portfolio.positions),
            'leverage': self.portfolio.leverage,
            'bars_processed': self.stats['bars_processed'],
            'signals_generated': self.stats['signals_generated'],
            'orders_placed': self.stats['orders_placed'],
            'orders_filled': self.stats['orders_filled'],
            'risk_breaches': self.stats['risk_breaches'],
            'can_trade': self.risk_engine.can_trade(),
            'current_drawdown': self.risk_engine.drawdown.current_drawdown,
            'alpha_model_weights': self.alpha_engine.blender.model_weights,
            'latest_allocation': {
                'method': self.latest_allocation.method if self.latest_allocation else None,
                'sharpe': self.latest_allocation.sharpe_ratio if self.latest_allocation else None
            } if self.latest_allocation else None
        }
    
    def _print_final_stats(self) -> None:
        """Print final system statistics"""
        
        runtime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        
        logger.info("\n" + "="*70)
        logger.info("FINAL SYSTEM STATISTICS")
        logger.info("="*70)
        logger.info(f"Runtime: {runtime.total_seconds()/3600:.2f} hours")
        logger.info(f"Bars Processed: {self.stats['bars_processed']}")
        logger.info(f"Signals Generated: {self.stats['signals_generated']}")
        logger.info(f"Orders Placed: {self.stats['orders_placed']}")
        logger.info(f"Orders Filled: {self.stats['orders_filled']}")
        logger.info(f"Total P&L: ${self.stats['total_pnl']:,.2f}")
        logger.info(f"Risk Breaches: {self.stats['risk_breaches']}")
        logger.info(f"Final Portfolio Value: ${self.portfolio.total_value:,.2f}")
        logger.info(f"Max Drawdown: {self.risk_engine.drawdown.max_drawdown:.2%}")
        logger.info("="*70)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_elite_quant_fund(
    symbols: Optional[List[str]] = None,
    initial_capital: float = 10_000_000,
    target_volatility: float = 0.10,
    kelly_fraction: float = 0.3,
    data_source: Optional[DataSource] = None
) -> EliteQuantFund:
    """
    Factory function to create Elite Quant Fund with sensible defaults
    """
    
    config = SystemConfig(
        symbols=symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        risk_limits=RiskLimits(
            max_position_value=initial_capital * 0.20,
            max_leverage=2.0,
            max_drawdown_pct=0.05,
            max_cvar_95=0.02,
            max_sector_concentration=0.25,
            kelly_fraction=kelly_fraction,
            kill_switch_drawdown=0.10
        ),
        optimization_method='black_litterman',
        target_volatility=target_volatility,
        data_source=data_source
    )
    
    return EliteQuantFund(config)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SystemConfig',
    'SystemState',
    'EliteQuantFund',
    'create_elite_quant_fund',
]
