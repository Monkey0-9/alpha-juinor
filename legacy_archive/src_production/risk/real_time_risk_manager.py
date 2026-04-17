"""
Real-Time Risk Manager - Production Implementation
Provides comprehensive risk management for live trading
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict
import threading
import time

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"

class RiskType(Enum):
    """Risk types"""
    POSITION_LIMIT = "position_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    CONCENTRATION_RISK = "concentration_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    OPERATIONAL_RISK = "operational_risk"
    COUNTERPARTY_RISK = "counterparty_risk"
    MODEL_RISK = "model_risk"
    LEGAL_RISK = "legal_risk"

class CircuitBreakerType(Enum):
    """Circuit breaker types"""
    PRICE_MOVEMENT = "price_movement"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN_LIMIT = "drawdown_limit"
    LOSS_LIMIT = "loss_limit"
    MARGIN_CALL = "margin_call"
    SYSTEM_ERROR = "system_error"

@dataclass
class RiskMetric:
    """Risk metric structure"""
    risk_type: RiskType
    symbol: str
    value: float
    threshold: float
    level: RiskLevel
    timestamp: datetime
    description: str
    action_required: bool
    action_taken: bool = False

@dataclass
class PositionLimit:
    """Position limit structure"""
    symbol: str
    asset_class: str
    max_position: float
    max_notional: float
    max_percentage: float
    current_position: float = 0.0
    current_notional: float = 0.0
    current_percentage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CircuitBreaker:
    """Circuit breaker structure"""
    name: str
    type: CircuitBreakerType
    symbol: Optional[str]
    threshold: float
    current_value: float
    triggered: bool
    triggered_at: Optional[datetime]
    reset_at: Optional[datetime]
    action: str
    auto_reset: bool
    reset_time: timedelta

@dataclass
class RiskAlert:
    """Risk alert structure"""
    alert_id: str
    risk_type: RiskType
    level: RiskLevel
    symbol: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

class RealTimeRiskManager:
    """Production real-time risk manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.position_limits = {}
        self.circuit_breakers = {}
        self.risk_metrics = {}
        self.risk_alerts = {}
        self.portfolio_data = {}
        self.market_data = {}
        self.running = False
        self.lock = threading.RLock()
        self.alert_callbacks = []
        self.circuit_breaker_callbacks = []
        self.risk_history = defaultdict(list)
        self.last_risk_check = datetime.utcnow()
        
        # Initialize risk parameters
        self._initialize_risk_parameters()
        
    def _initialize_risk_parameters(self):
        """Initialize risk parameters from config"""
        # Position limits
        for limit_config in self.config.get('position_limits', []):
            limit = PositionLimit(
                symbol=limit_config['symbol'],
                asset_class=limit_config['asset_class'],
                max_position=limit_config['max_position'],
                max_notional=limit_config['max_notional'],
                max_percentage=limit_config['max_percentage']
            )
            self.position_limits[limit.symbol] = limit
        
        # Circuit breakers
        for cb_config in self.config.get('circuit_breakers', []):
            circuit_breaker = CircuitBreaker(
                name=cb_config['name'],
                type=CircuitBreakerType(cb_config['type']),
                symbol=cb_config.get('symbol'),
                threshold=cb_config['threshold'],
                current_value=0.0,
                triggered=False,
                triggered_at=None,
                reset_at=None,
                action=cb_config['action'],
                auto_reset=cb_config.get('auto_reset', True),
                reset_time=timedelta(minutes=cb_config.get('reset_minutes', 30))
            )
            self.circuit_breakers[circuit_breaker.name] = circuit_breaker
    
    async def start(self):
        """Start risk manager"""
        self.running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_position_limits())
        asyncio.create_task(self._monitor_circuit_breakers())
        asyncio.create_task(self._calculate_portfolio_risk())
        asyncio.create_task(self._check_market_risk())
        asyncio.create_task(self._monitor_operational_risk())
        asyncio.create_task(self._reset_circuit_breakers())
        
        logger.info("Real-time risk manager started")
    
    async def stop(self):
        """Stop risk manager"""
        self.running = False
        logger.info("Real-time risk manager stopped")
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """Update position data"""
        with self.lock:
            if symbol not in self.portfolio_data:
                self.portfolio_data[symbol] = {
                    'quantity': 0.0,
                    'notional': 0.0,
                    'avg_price': 0.0,
                    'unrealized_pl': 0.0,
                    'realized_pl': 0.0
                }
            
            position_data = self.portfolio_data[symbol]
            old_quantity = position_data['quantity']
            
            # Update position
            position_data['quantity'] += quantity
            position_data['notional'] = position_data['quantity'] * price
            
            # Calculate average price
            if old_quantity != 0:
                position_data['avg_price'] = (old_quantity * position_data['avg_price'] + quantity * price) / position_data['quantity']
            else:
                position_data['avg_price'] = price
    
    def update_market_data(self, symbol: str, price: float, volume: int, volatility: float):
        """Update market data"""
        with self.lock:
            self.market_data[symbol] = {
                'price': price,
                'volume': volume,
                'volatility': volatility,
                'timestamp': datetime.utcnow()
            }
    
    async def check_order_risk(self, symbol: str, side: str, quantity: float, price: float) -> Tuple[bool, List[RiskAlert]]:
        """Check if order is within risk limits"""
        alerts = []
        approved = True
        
        with self.lock:
            # Check position limits
            if symbol in self.position_limits:
                limit = self.position_limits[symbol]
                current_position = self.portfolio_data.get(symbol, {}).get('quantity', 0.0)
                
                # Calculate new position
                if side == 'buy':
                    new_position = current_position + quantity
                else:
                    new_position = current_position - quantity
                
                # Check position limit
                if abs(new_position) > limit.max_position:
                    alert = RiskAlert(
                        alert_id=f"pos_limit_{symbol}_{int(time.time())}",
                        risk_type=RiskType.POSITION_LIMIT,
                        level=RiskLevel.HIGH,
                        symbol=symbol,
                        message=f"Position limit exceeded: {new_position} > {limit.max_position}",
                        timestamp=datetime.utcnow()
                    )
                    alerts.append(alert)
                    approved = False
                
                # Check notional limit
                new_notional = abs(new_position * price)
                if new_notional > limit.max_notional:
                    alert = RiskAlert(
                        alert_id=f"notional_limit_{symbol}_{int(time.time())}",
                        risk_type=RiskType.POSITION_LIMIT,
                        level=RiskLevel.HIGH,
                        symbol=symbol,
                        message=f"Notional limit exceeded: ${new_notional:,.2f} > ${limit.max_notional:,.2f}",
                        timestamp=datetime.utcnow()
                    )
                    alerts.append(alert)
                    approved = False
            
            # Check circuit breakers
            for cb_name, cb in self.circuit_breakers.items():
                if cb.triggered and cb.symbol == symbol:
                    alert = RiskAlert(
                        alert_id=f"cb_triggered_{symbol}_{int(time.time())}",
                        risk_type=RiskType.MARKET_RISK,
                        level=RiskLevel.CRITICAL,
                        symbol=symbol,
                        message=f"Circuit breaker triggered: {cb.name}",
                        timestamp=datetime.utcnow()
                    )
                    alerts.append(alert)
                    approved = False
            
            # Store alerts
            for alert in alerts:
                self.risk_alerts[alert.alert_id] = alert
                await self._notify_alert(alert)
        
        return approved, alerts
    
    async def _monitor_position_limits(self):
        """Monitor position limits"""
        while self.running:
            try:
                with self.lock:
                    for symbol, limit in self.position_limits.items():
                        if symbol in self.portfolio_data:
                            position_data = self.portfolio_data[symbol]
                            
                            # Update current values
                            limit.current_position = position_data['quantity']
                            limit.current_notional = position_data['notional']
                            
                            # Calculate percentage
                            total_portfolio_value = sum(
                                pos['notional'] for pos in self.portfolio_data.values()
                            )
                            if total_portfolio_value > 0:
                                limit.current_percentage = position_data['notional'] / total_portfolio_value
                            
                            # Check for breaches
                            if abs(limit.current_position) > limit.max_position * 0.9:
                                await self._create_risk_alert(
                                    RiskType.POSITION_LIMIT,
                                    RiskLevel.HIGH,
                                    symbol,
                                    f"Position limit approaching: {limit.current_position}/{limit.max_position}"
                                )
                            
                            if limit.current_notional > limit.max_notional * 0.9:
                                await self._create_risk_alert(
                                    RiskType.POSITION_LIMIT,
                                    RiskLevel.HIGH,
                                    symbol,
                                    f"Notional limit approaching: ${limit.current_notional:,.2f}/${limit.max_notional:,.2f}"
                                )
                            
                            if limit.current_percentage > limit.max_percentage * 0.9:
                                await self._create_risk_alert(
                                    RiskType.CONCENTRATION_RISK,
                                    RiskLevel.MEDIUM,
                                    symbol,
                                    f"Concentration risk: {limit.current_percentage:.2%}/{limit.max_percentage:.2%}"
                                )
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring position limits: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_circuit_breakers(self):
        """Monitor circuit breakers"""
        while self.running:
            try:
                with self.lock:
                    for cb_name, cb in self.circuit_breakers.items():
                        if cb.type == CircuitBreakerType.PRICE_MOVEMENT and cb.symbol:
                            if cb.symbol in self.market_data:
                                market_data = self.market_data[cb.symbol]
                                # Check for price movement (simplified)
                                # In production, would compare to historical data
                                if market_data['volatility'] > cb.threshold:
                                    await self._trigger_circuit_breaker(cb, market_data['volatility'])
                        
                        elif cb.type == CircuitBreakerType.VOLUME_SPIKE and cb.symbol:
                            if cb.symbol in self.market_data:
                                market_data = self.market_data[cb.symbol]
                                # Check for volume spike
                                if market_data['volume'] > cb.threshold:
                                    await self._trigger_circuit_breaker(cb, market_data['volume'])
                        
                        elif cb.type == CircuitBreakerType.DRAWDOWN_LIMIT:
                            # Check portfolio drawdown
                            total_unrealized_pl = sum(
                                pos.get('unrealized_pl', 0) for pos in self.portfolio_data.values()
                            )
                            if total_unrealized_pl < -cb.threshold:
                                await self._trigger_circuit_breaker(cb, total_unrealized_pl)
                        
                        elif cb.type == CircuitBreakerType.LOSS_LIMIT:
                            # Check total loss
                            total_pl = sum(
                                pos.get('unrealized_pl', 0) + pos.get('realized_pl', 0) 
                                for pos in self.portfolio_data.values()
                            )
                            if total_pl < -cb.threshold:
                                await self._trigger_circuit_breaker(cb, total_pl)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error monitoring circuit breakers: {e}")
                await asyncio.sleep(1)
    
    async def _calculate_portfolio_risk(self):
        """Calculate portfolio-level risk metrics"""
        while self.running:
            try:
                with self.lock:
                    # Calculate portfolio VaR
                    portfolio_var = self._calculate_var()
                    
                    # Calculate concentration risk
                    concentration_risk = self._calculate_concentration_risk()
                    
                    # Calculate leverage
                    leverage = self._calculate_leverage()
                    
                    # Store risk metrics
                    self.risk_metrics['portfolio_var'] = portfolio_var
                    self.risk_metrics['concentration_risk'] = concentration_risk
                    self.risk_metrics['leverage'] = leverage
                    
                    # Check risk thresholds
                    if portfolio_var > self.config.get('max_portfolio_var', 100000):
                        await self._create_risk_alert(
                            RiskType.MARKET_RISK,
                            RiskLevel.HIGH,
                            "PORTFOLIO",
                            f"Portfolio VaR exceeded: ${portfolio_var:,.2f}"
                        )
                    
                    if leverage > self.config.get('max_leverage', 3.0):
                        await self._create_risk_alert(
                            RiskType.LEVERAGE_LIMIT,
                            RiskLevel.HIGH,
                            "PORTFOLIO",
                            f"Leverage exceeded: {leverage:.2f}x"
                        )
                
                await asyncio.sleep(30)  # Calculate every 30 seconds
                
            except Exception as e:
                logger.error(f"Error calculating portfolio risk: {e}")
                await asyncio.sleep(5)
    
    async def _check_market_risk(self):
        """Check market-wide risk factors"""
        while self.running:
            try:
                # Check for market-wide circuit breakers
                # This would typically include:
                # - Market index movements
                # - VIX spikes
                # - Market volatility
                # - Liquidity dry-ups
                
                # Simplified implementation
                market_volatility = np.mean([
                    data['volatility'] for data in self.market_data.values()
                ]) if self.market_data else 0.0
                
                if market_volatility > 0.05:  # 5% volatility threshold
                    await self._create_risk_alert(
                        RiskType.MARKET_RISK,
                        RiskLevel.MEDIUM,
                        "MARKET",
                        f"High market volatility: {market_volatility:.2%}"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error checking market risk: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_operational_risk(self):
        """Monitor operational risk"""
        while self.running:
            try:
                # Check for system errors, connectivity issues, etc.
                # This would typically include:
                # - System health checks
                # - Connection monitoring
                # - Error rate monitoring
                # - Performance monitoring
                
                # Simplified implementation
                error_rate = self._calculate_error_rate()
                
                if error_rate > 0.05:  # 5% error rate threshold
                    await self._create_risk_alert(
                        RiskType.OPERATIONAL_RISK,
                        RiskLevel.HIGH,
                        "SYSTEM",
                        f"High error rate: {error_rate:.2%}"
                    )
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring operational risk: {e}")
                await asyncio.sleep(30)
    
    async def _reset_circuit_breakers(self):
        """Reset circuit breakers when appropriate"""
        while self.running:
            try:
                with self.lock:
                    for cb_name, cb in self.circuit_breakers.items():
                        if cb.triggered and cb.auto_reset:
                            if cb.reset_at and datetime.utcnow() >= cb.reset_at:
                                await self._reset_circuit_breaker(cb)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error resetting circuit breakers: {e}")
                await asyncio.sleep(30)
    
    def _calculate_var(self) -> float:
        """Calculate portfolio Value at Risk"""
        # Simplified VaR calculation
        # In production, would use sophisticated models
        total_notional = sum(
            pos['notional'] for pos in self.portfolio_data.values()
        )
        
        # Assume 2% VaR for demonstration
        return total_notional * 0.02
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk"""
        if not self.portfolio_data:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        total_notional = sum(
            pos['notional'] for pos in self.portfolio_data.values()
        )
        
        if total_notional == 0:
            return 0.0
        
        hhi = sum(
            (pos['notional'] / total_notional) ** 2 
            for pos in self.portfolio_data.values()
        )
        
        return hhi
    
    def _calculate_leverage(self) -> float:
        """Calculate portfolio leverage"""
        total_notional = sum(
            abs(pos['notional']) for pos in self.portfolio_data.values()
        )
        
        # This would need actual equity calculation
        # Simplified implementation
        return total_notional / 1000000 if total_notional > 0 else 0.0  # Assuming $1M equity
    
    def _calculate_error_rate(self) -> float:
        """Calculate system error rate"""
        # Simplified implementation
        # In production, would track actual errors
        return 0.01  # 1% error rate
    
    async def _create_risk_alert(self, risk_type: RiskType, level: RiskLevel, symbol: str, message: str):
        """Create risk alert"""
        alert = RiskAlert(
            alert_id=f"{risk_type.value}_{symbol}_{int(time.time())}",
            risk_type=risk_type,
            level=level,
            symbol=symbol,
            message=message,
            timestamp=datetime.utcnow()
        )
        
        self.risk_alerts[alert.alert_id] = alert
        await self._notify_alert(alert)
    
    async def _trigger_circuit_breaker(self, cb: CircuitBreaker, current_value: float):
        """Trigger circuit breaker"""
        if not cb.triggered:
            cb.triggered = True
            cb.triggered_at = datetime.utcnow()
            cb.current_value = current_value
            
            # Set reset time
            cb.reset_at = datetime.utcnow() + cb.reset_time
            
            # Create alert
            await self._create_risk_alert(
                RiskType.MARKET_RISK,
                RiskLevel.CRITICAL,
                cb.symbol or "SYSTEM",
                f"Circuit breaker triggered: {cb.name} - {cb.action}"
            )
            
            # Notify callbacks
            for callback in self.circuit_breaker_callbacks:
                try:
                    await callback(cb)
                except Exception as e:
                    logger.error(f"Error in circuit breaker callback: {e}")
    
    async def _reset_circuit_breaker(self, cb: CircuitBreaker):
        """Reset circuit breaker"""
        cb.triggered = False
        cb.current_value = 0.0
        cb.triggered_at = None
        cb.reset_at = None
        
        logger.info(f"Circuit breaker reset: {cb.name}")
    
    async def _notify_alert(self, alert: RiskAlert):
        """Notify alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def add_circuit_breaker_callback(self, callback):
        """Add circuit breaker callback"""
        self.circuit_breaker_callbacks.append(callback)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary"""
        with self.lock:
            return {
                'position_limits': {
                    symbol: {
                        'current': limit.current_position,
                        'max': limit.max_position,
                        'utilization': abs(limit.current_position) / limit.max_position if limit.max_position > 0 else 0
                    }
                    for symbol, limit in self.position_limits.items()
                },
                'circuit_breakers': {
                    name: {
                        'triggered': cb.triggered,
                        'triggered_at': cb.triggered_at.isoformat() if cb.triggered_at else None,
                        'action': cb.action
                    }
                    for name, cb in self.circuit_breakers.items()
                },
                'risk_metrics': self.risk_metrics,
                'active_alerts': len([a for a in self.risk_alerts.values() if not a.resolved]),
                'portfolio_value': sum(pos['notional'] for pos in self.portfolio_data.values()),
                'last_check': self.last_risk_check.isoformat()
            }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge alert"""
        with self.lock:
            if alert_id in self.risk_alerts:
                self.risk_alerts[alert_id].acknowledged = True
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert"""
        with self.lock:
            if alert_id in self.risk_alerts:
                self.risk_alerts[alert_id].resolved = True
                return True
            return False
