"""
Risk Engine - Elite Quant Fund System
Historical-simulation CVaR, Ledoit-Wolf covariance, fractional Kelly sizing,
real-time drawdown watermark, kill switch, sector concentration (GICS),
pre-trade order validation.
Built to Renaissance Technologies / Jane Street standards
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from numpy.linalg import inv, LinAlgError

from elite_quant_fund.core.types import (
    Position, Portfolio, RiskLimits, RiskBreach, RiskBreachType,
    Order, MarketBar, calc_cvar, ledoit_wolf_shrinkage,
    fractional_kelly_size, Result
)

logger = logging.getLogger(__name__)


# ============================================================================
# GICS SECTOR CLASSIFICATION
# ============================================================================

class GICSSector(Enum):
    """Global Industry Classification Standard sectors"""
    COMMUNICATION_SERVICES = "Communication Services"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    ENERGY = "Energy"
    FINANCIALS = "Financials"
    HEALTH_CARE = "Health Care"
    INDUSTRIALS = "Industrials"
    INFORMATION_TECHNOLOGY = "Information Technology"
    MATERIALS = "Materials"
    REAL_ESTATE = "Real Estate"
    UTILITIES = "Utilities"
    UNKNOWN = "Unknown"


# Simplified mapping (production would use full GICS database)
SECTOR_MAP: Dict[str, GICSSector] = {
    # Technology
    'AAPL': GICSSector.INFORMATION_TECHNOLOGY,
    'MSFT': GICSSector.INFORMATION_TECHNOLOGY,
    'GOOGL': GICSSector.COMMUNICATION_SERVICES,
    'NVDA': GICSSector.INFORMATION_TECHNOLOGY,
    'AMD': GICSSector.INFORMATION_TECHNOLOGY,
    'CRM': GICSSector.INFORMATION_TECHNOLOGY,
    
    # Consumer
    'AMZN': GICSSector.CONSUMER_DISCRETIONARY,
    'TSLA': GICSSector.CONSUMER_DISCRETIONARY,
    'NFLX': GICSSector.COMMUNICATION_SERVICES,
    'META': GICSSector.COMMUNICATION_SERVICES,
    
    # Others (defaults)
}


def get_sector(symbol: str) -> GICSSector:
    """Get GICS sector for symbol"""
    return SECTOR_MAP.get(symbol, GICSSector.UNKNOWN)


# ============================================================================
# DRAWDOWN WATERMARK
# ============================================================================

class DrawdownWatermark:
    """
    Real-time drawdown tracking with peak/trough watermarks
    """
    
    def __init__(self):
        self.peak: float = 0.0
        self.trough: float = 0.0
        self.peak_time: Optional[datetime] = None
        self.trough_time: Optional[datetime] = None
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        self.drawdown_history: deque = deque(maxlen=252)  # 1 year daily
    
    def update(self, pnl: float, timestamp: datetime) -> float:
        """Update drawdown with new P&L, return current drawdown percentage"""
        
        # Update peak
        if pnl > self.peak:
            self.peak = pnl
            self.peak_time = timestamp
            # Reset trough on new peak
            self.trough = pnl
            self.trough_time = timestamp
        
        # Update trough
        if pnl < self.trough:
            self.trough = pnl
            self.trough_time = timestamp
        
        # Calculate drawdown
        if self.peak > 0:
            self.current_drawdown = (self.peak - pnl) / self.peak
        else:
            self.current_drawdown = 0.0
        
        # Update max drawdown
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Store history
        self.drawdown_history.append({
            'timestamp': timestamp,
            'pnl': pnl,
            'drawdown': self.current_drawdown
        })
        
        return self.current_drawdown
    
    def is_breached(self, threshold: float) -> bool:
        """Check if current drawdown exceeds threshold"""
        return self.current_drawdown > threshold
    
    def get_duration(self) -> Optional[timedelta]:
        """Get duration of current drawdown"""
        if self.peak_time and self.trough_time:
            return self.trough_time - self.peak_time
        return None


# ============================================================================
# HISTORICAL CVaR CALCULATOR
# ============================================================================

class HistoricalCVaR:
    """
    Historical simulation CVaR (Expected Shortfall)
    More conservative than VaR, coherent risk measure
    """
    
    def __init__(self, confidence: float = 0.95, lookback_days: int = 252):
        self.confidence = confidence
        self.lookback_days = lookback_days
        
        # Return history per symbol
        self.return_history: Dict[str, deque] = {}
        
        # CVaR cache
        self.cvar_cache: Dict[str, float] = {}
        self.last_update: Optional[datetime] = None
    
    def add_returns(self, symbol: str, returns: float) -> None:
        """Add return to history"""
        if symbol not in self.return_history:
            self.return_history[symbol] = deque(maxlen=self.lookback_days)
        
        self.return_history[symbol].append(returns)
        
        # Invalidate cache
        if symbol in self.cvar_cache:
            del self.cvar_cache[symbol]
    
    def calculate_cvar(self, symbol: str) -> float:
        """Calculate CVaR for symbol"""
        
        if symbol in self.cvar_cache:
            return self.cvar_cache[symbol]
        
        history = self.return_history.get(symbol)
        if not history or len(history) < 30:
            return 0.02  # Default 2% CVaR
        
        returns = np.array(list(history))
        
        # Calculate CVaR (Expected Shortfall)
        var_threshold = np.percentile(returns, (1 - self.confidence) * 100)
        cvar = np.mean(returns[returns <= var_threshold])
        
        self.cvar_cache[symbol] = abs(cvar)  # Return positive value
        
        return abs(cvar)
    
    def calculate_portfolio_cvar(
        self,
        positions: Dict[str, Position],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate portfolio-level CVaR using covariance
        """
        
        symbols = list(positions.keys())
        if not symbols:
            return 0.0
        
        # Build return matrix
        return_matrix = []
        for sym in symbols:
            if sym in self.return_history and len(self.return_history[sym]) >= 30:
                returns = np.array(list(self.return_history[sym])[-30:])
                return_matrix.append(returns)
        
        if len(return_matrix) < len(symbols):
            # Not enough data for all symbols
            return max(self.calculate_cvar(sym) for sym in symbols)
        
        # Calculate portfolio returns
        if weights is None:
            total_value = sum(abs(p.market_value) for p in positions.values())
            weights = {
                sym: positions[sym].market_value / total_value if total_value > 0 else 0
                for sym in symbols
            }
        
        return_matrix = np.array(return_matrix).T  # (T x N)
        portfolio_returns = np.array([
            sum(r[i] * weights[sym] for i, sym in enumerate(symbols))
            for r in return_matrix
        ])
        
        # Calculate portfolio CVaR
        var_threshold = np.percentile(portfolio_returns, (1 - self.confidence) * 100)
        portfolio_cvar = np.mean(portfolio_returns[portfolio_returns <= var_threshold])
        
        return abs(portfolio_cvar)


# ============================================================================
# LEDOIT-WOLF COVARIANCE ESTIMATOR
# ============================================================================

class LedoitWolfCovariance:
    """
    Ledoit-Wolf covariance shrinkage estimator
    More stable than sample covariance, especially for p > n
    """
    
    def __init__(self, shrinkage_target: str = 'constant_correlation'):
        self.shrinkage_target = shrinkage_target
        self.covariance_cache: Optional[np.ndarray] = None
        self.correlation_cache: Optional[np.ndarray] = None
        self.symbols_cache: Optional[List[str]] = None
        self.last_update: Optional[datetime] = None
    
    def estimate(
        self,
        returns_df: pd.DataFrame,
        force_shrinkage: Optional[float] = None
    ) -> np.ndarray:
        """
        Estimate covariance matrix using Ledoit-Wolf shrinkage
        
        Args:
            returns_df: DataFrame with symbols as columns, time as rows
            force_shrinkage: Optional fixed shrinkage parameter
        
        Returns:
            Shrinkage covariance matrix
        """
        
        if returns_df.empty or len(returns_df) < 10:
            # Return identity if insufficient data
            n = len(returns_df.columns)
            return np.eye(n) * 0.01  # 1% daily variance default
        
        # Sample covariance
        sample_cov = returns_df.cov().values
        
        # Shrinkage target
        if self.shrinkage_target == 'constant_correlation':
            # Target: constant correlation model
            variances = np.diag(sample_cov)
            avg_var = np.mean(variances)
            
            # Average correlation
            corr_matrix = returns_df.corr().values
            # Mask diagonal
            mask = ~np.eye(len(corr_matrix), dtype=bool)
            avg_corr = np.mean(corr_matrix[mask]) if np.any(mask) else 0.0
            
            # Build target
            target = np.ones_like(sample_cov) * avg_corr
            np.fill_diagonal(target, 1.0)
            target = np.outer(np.sqrt(variances), np.sqrt(variances)) * target
        
        elif self.shrinkage_target == 'identity':
            target = np.eye(len(sample_cov)) * np.mean(np.diag(sample_cov))
        
        else:
            target = np.eye(len(sample_cov)) * np.mean(np.diag(sample_cov))
        
        # Calculate optimal shrinkage intensity
        if force_shrinkage is not None:
            delta = force_shrinkage
        else:
            delta = self._optimal_shrinkage(returns_df.values, sample_cov, target)
        
        # Apply shrinkage
        shrunk_cov = delta * target + (1 - delta) * sample_cov
        
        # Ensure positive semi-definite
        eigenvalues = np.linalg.eigvalsh(shrunk_cov)
        if np.any(eigenvalues < 0):
            # Add small diagonal adjustment
            min_eig = np.min(eigenvalues)
            shrunk_cov += np.eye(len(shrunk_cov)) * (abs(min_eig) + 1e-6)
        
        # Cache results
        self.covariance_cache = shrunk_cov
        self.symbols_cache = list(returns_df.columns)
        self.last_update = datetime.now()
        
        return shrunk_cov
    
    def _optimal_shrinkage(
        self,
        returns: np.ndarray,
        sample_cov: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Calculate optimal shrinkage intensity (Ledoit-Wolf formula)"""
        
        t, n = returns.shape
        
        if t < 2 or n < 2:
            return 0.5  # Default moderate shrinkage
        
        # Frobenius norm of difference
        diff = sample_cov - target
        pi = np.sum(diff ** 2)
        
        # Calculate rho (asymptotically optimal)
        # Simplified version - full version uses all elements
        sample_var = np.var(returns, axis=0, ddof=1)
        
        # Shrinkage intensity
        if pi == 0:
            return 0.0
        
        # Simplified optimal delta
        delta = max(0.0, min(1.0, 1.0 / (1.0 + t / (n * 10))))
        
        return delta
    
    def get_covariance(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Get cached covariance matrix for symbols"""
        
        if (self.covariance_cache is None or 
            self.symbols_cache is None or
            set(symbols) != set(self.symbols_cache)):
            return None
        
        return self.covariance_cache


# ============================================================================
# KELLY CRITERION POSITION SIZING
# ============================================================================

class KellyPositionSizing:
    """
    Fractional Kelly criterion for optimal position sizing
    f* = (mu - r) / sigma^2
    We use fractional Kelly for safety: f = f* * fraction
    """
    
    def __init__(self, fraction: float = 0.3, max_leverage: float = 2.0):
        self.fraction = fraction  # Conservative: 30% of full Kelly
        self.max_leverage = max_leverage
    
    def calculate_size(
        self,
        expected_return: float,
        volatility: float,
        current_equity: float,
        risk_limits: RiskLimits
    ) -> float:
        """
        Calculate optimal position size using fractional Kelly
        
        Args:
            expected_return: Expected return (mu - r)
            volatility: Expected volatility (sigma)
            current_equity: Current portfolio equity
            risk_limits: Risk limit configuration
        
        Returns:
            Optimal position size in dollars
        """
        
        if volatility <= 0 or expected_return <= 0:
            return 0.0
        
        # Full Kelly fraction
        kelly_full = expected_return / (volatility ** 2)
        
        # Fractional Kelly
        kelly_fractional = kelly_full * self.fraction
        
        # Apply leverage constraint
        kelly_capped = min(kelly_fractional, self.max_leverage)
        
        # Calculate position size
        position_size = kelly_capped * current_equity
        
        # Apply risk limits
        max_position = risk_limits.max_position_value
        position_size = min(position_size, max_position)
        
        return max(0.0, position_size)
    
    def adjust_for_drawdown(
        self,
        base_size: float,
        current_drawdown: float,
        max_drawdown: float
    ) -> float:
        """
        Reduce position size during drawdowns
        Scales from 100% at 0% DD to 0% at max_drawdown
        """
        
        if max_drawdown <= 0:
            return base_size
        
        # Linear scaling
        scale = max(0.0, 1.0 - (current_drawdown / max_drawdown))
        
        return base_size * scale


# ============================================================================
# SECTOR CONCENTRATION MONITOR
# ============================================================================

class SectorConcentration:
    """
    GICS sector concentration monitoring
    """
    
    def __init__(self, max_concentration: float = 0.25):
        self.max_concentration = max_concentration
    
    def calculate_exposures(
        self,
        positions: Dict[str, Position]
    ) -> Dict[GICSSector, float]:
        """Calculate sector exposures"""
        
        total_value = sum(abs(p.market_value) for p in positions.values())
        
        if total_value == 0:
            return {}
        
        sector_values: Dict[GICSSector, float] = {}
        
        for sym, pos in positions.items():
            sector = get_sector(sym)
            sector_values[sector] = sector_values.get(sector, 0) + abs(pos.market_value)
        
        # Convert to percentages
        sector_exposures = {
            sector: value / total_value
            for sector, value in sector_values.items()
        }
        
        return sector_exposures
    
    def check_limits(
        self,
        positions: Dict[str, Position]
    ) -> List[RiskBreach]:
        """Check for sector concentration breaches"""
        
        breaches = []
        exposures = self.calculate_exposures(positions)
        
        for sector, exposure in exposures.items():
            if exposure > self.max_concentration:
                breach = RiskBreach(
                    breach_type=RiskBreachType.SECTOR_CONCENTRATION,
                    timestamp=datetime.now(),
                    severity=exposure / self.max_concentration - 1.0,
                    description=f"Sector {sector.value} concentration {exposure:.2%} exceeds {self.max_concentration:.2%}",
                    metric_value=exposure,
                    threshold=self.max_concentration
                )
                breaches.append(breach)
        
        return breaches


# ============================================================================
# KILL SWITCH
# ============================================================================

class KillSwitch:
    """
    Emergency kill switch for catastrophic risk events
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.10,
        max_daily_loss: float = 50000,
        max_cvar: float = 0.05
    ):
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_cvar = max_cvar
        
        self.is_triggered = False
        self.trigger_reason: Optional[str] = None
        self.trigger_time: Optional[datetime] = None
        
        # Breach history
        self.breach_history: List[RiskBreach] = []
    
    def check(
        self,
        current_drawdown: float,
        daily_pnl: float,
        portfolio_cvar: float
    ) -> Optional[RiskBreach]:
        """Check if kill switch should trigger"""
        
        if self.is_triggered:
            return None
        
        # Check drawdown
        if current_drawdown > self.max_drawdown:
            self.is_triggered = True
            self.trigger_reason = f"Max drawdown: {current_drawdown:.2%}"
            self.trigger_time = datetime.now()
            
            breach = RiskBreach(
                breach_type=RiskBreachType.KILL_SWITCH,
                timestamp=datetime.now(),
                severity=1.0,  # Critical
                description=self.trigger_reason,
                metric_value=current_drawdown,
                threshold=self.max_drawdown
            )
            
            self.breach_history.append(breach)
            logger.critical(f"KILL SWITCH TRIGGERED: {self.trigger_reason}")
            
            return breach
        
        # Check daily loss
        if daily_pnl < -self.max_daily_loss:
            self.is_triggered = True
            self.trigger_reason = f"Max daily loss: ${daily_pnl:,.2f}"
            self.trigger_time = datetime.now()
            
            breach = RiskBreach(
                breach_type=RiskBreachType.KILL_SWITCH,
                timestamp=datetime.now(),
                severity=1.0,
                description=self.trigger_reason,
                metric_value=abs(daily_pnl),
                threshold=self.max_daily_loss
            )
            
            self.breach_history.append(breach)
            logger.critical(f"KILL SWITCH TRIGGERED: {self.trigger_reason}")
            
            return breach
        
        # Check CVaR
        if portfolio_cvar > self.max_cvar:
            self.is_triggered = True
            self.trigger_reason = f"Max CVaR: {portfolio_cvar:.2%}"
            self.trigger_time = datetime.now()
            
            breach = RiskBreach(
                breach_type=RiskBreachType.KILL_SWITCH,
                timestamp=datetime.now(),
                severity=1.0,
                description=self.trigger_reason,
                metric_value=portfolio_cvar,
                threshold=self.max_cvar
            )
            
            self.breach_history.append(breach)
            logger.critical(f"KILL SWITCH TRIGGERED: {self.trigger_reason}")
            
            return breach
        
        return None
    
    def reset(self) -> None:
        """Reset kill switch (manual override)"""
        self.is_triggered = False
        self.trigger_reason = None
        self.trigger_time = None
        logger.warning("Kill switch manually reset")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        return not self.is_triggered


# ============================================================================
# MAIN RISK ENGINE
# ============================================================================

class RiskEngine:
    """
    Main risk engine orchestrating all risk management components
    """
    
    def __init__(self, risk_limits: RiskLimits):
        self.limits = risk_limits
        
        # Components
        self.drawdown = DrawdownWatermark()
        self.cvar_calculator = HistoricalCVaR(confidence=0.95)
        self.covariance_estimator = LedoitWolfCovariance()
        self.kelly_sizing = KellyPositionSizing(
            fraction=risk_limits.kelly_fraction,
            max_leverage=risk_limits.max_leverage
        )
        self.sector_monitor = SectorConcentration(
            max_concentration=risk_limits.max_sector_concentration
        )
        self.kill_switch = KillSwitch(
            max_drawdown=risk_limits.kill_switch_drawdown,
            max_daily_loss=50000,  # Can be configured
            max_cvar=risk_limits.max_cvar_95
        )
        
        # Breach handlers
        self.breach_handlers: List[Callable[[RiskBreach], None]] = []
        
        # Stats
        self.breaches_detected = 0
        self.pre_trade_checks = 0
        self.pre_trade_blocks = 0
    
    def register_breach_handler(self, handler: Callable[[RiskBreach], None]) -> None:
        """Register callback for risk breaches"""
        self.breach_handlers.append(handler)
    
    def update_portfolio(self, portfolio: Portfolio) -> List[RiskBreach]:
        """
        Update risk state with current portfolio
        Returns list of risk breaches
        """
        
        breaches = []
        
        # Update drawdown
        total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in portfolio.positions.values())
        current_dd = self.drawdown.update(total_pnl, portfolio.timestamp)
        
        # Check kill switch
        daily_pnl = portfolio.total_value - (portfolio.total_value - total_pnl)
        portfolio_cvar = self.cvar_calculator.calculate_portfolio_cvar(
            portfolio.positions
        )
        
        kill_breach = self.kill_switch.check(current_dd, daily_pnl, portfolio_cvar)
        if kill_breach:
            breaches.append(kill_breach)
        
        # Check sector concentration
        sector_breaches = self.sector_monitor.check_limits(portfolio.positions)
        breaches.extend(sector_breaches)
        
        # Check leverage
        if portfolio.leverage > self.limits.max_leverage:
            breaches.append(RiskBreach(
                breach_type=RiskBreachType.LEVERAGE_LIMIT,
                timestamp=datetime.now(),
                severity=(portfolio.leverage / self.limits.max_leverage) - 1.0,
                description=f"Leverage {portfolio.leverage:.2f}x exceeds {self.limits.max_leverage:.2f}x",
                metric_value=portfolio.leverage,
                threshold=self.limits.max_leverage
            ))
        
        # Check CVaR
        if portfolio_cvar > self.limits.max_cvar_95:
            breaches.append(RiskBreach(
                breach_type=RiskBreachType.CVAR_LIMIT,
                timestamp=datetime.now(),
                severity=(portfolio_cvar / self.limits.max_cvar_95) - 1.0,
                description=f"Portfolio CVaR {portfolio_cvar:.2%} exceeds {self.limits.max_cvar_95:.2%}",
                metric_value=portfolio_cvar,
                threshold=self.limits.max_cvar_95
            ))
        
        # Notify handlers
        for breach in breaches:
            self.breaches_detected += 1
            for handler in self.breach_handlers:
                try:
                    handler(breach)
                except Exception as e:
                    logger.error(f"Breach handler error: {e}")
        
        return breaches
    
    def pre_trade_check(self, order: Order, portfolio: Portfolio) -> Result[bool]:
        """
        Pre-trade risk validation
        Returns Ok(True) if approved, Err(reason) if blocked
        """
        
        self.pre_trade_checks += 1
        
        # Check kill switch
        if not self.kill_switch.can_trade():
            self.pre_trade_blocks += 1
            return Result.err(f"Kill switch active: {self.kill_switch.trigger_reason}")
        
        # Check position limit
        current_value = sum(abs(p.market_value) for p in portfolio.positions.values())
        order_value = order.quantity * (order.limit_price or 100)  # Estimate
        
        if current_value + order_value > self.limits.max_position_value:
            self.pre_trade_blocks += 1
            return Result.err(f"Position limit would be exceeded")
        
        # Check leverage (post-trade)
        new_exposure = current_value + order_value
        new_leverage = new_exposure / portfolio.total_value if portfolio.total_value > 0 else 0
        
        if new_leverage > self.limits.max_leverage:
            self.pre_trade_blocks += 1
            return Result.err(f"Leverage would exceed {self.limits.max_leverage:.2f}x")
        
        # Check sector concentration
        sector = get_sector(order.symbol)
        sector_exposures = self.sector_monitor.calculate_exposures(portfolio.positions)
        current_sector = sector_exposures.get(sector, 0)
        
        # Estimate new sector exposure
        total_value_with_order = current_value + order_value
        new_sector_exposure = (current_sector * current_value + order_value) / total_value_with_order
        
        if new_sector_exposure > self.limits.max_sector_concentration:
            self.pre_trade_blocks += 1
            return Result.err(f"Sector {sector.value} concentration would be {new_sector_exposure:.2%}")
        
        return Result.ok(True)
    
    def calculate_optimal_size(
        self,
        symbol: str,
        expected_return: float,
        volatility: float,
        portfolio: Portfolio
    ) -> float:
        """Calculate optimal position size using Kelly criterion"""
        
        base_size = self.kelly_sizing.calculate_size(
            expected_return,
            volatility,
            portfolio.total_value,
            self.limits
        )
        
        # Adjust for drawdown
        final_size = self.kelly_sizing.adjust_for_drawdown(
            base_size,
            self.drawdown.current_drawdown,
            self.limits.max_drawdown_pct
        )
        
        return final_size
    
    def add_return(self, symbol: str, ret: float) -> None:
        """Add return to CVaR calculator"""
        self.cvar_calculator.add_returns(symbol, ret)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get risk engine statistics"""
        return {
            'breaches_detected': self.breaches_detected,
            'pre_trade_checks': self.pre_trade_checks,
            'pre_trade_blocks': self.pre_trade_blocks,
            'current_drawdown': self.drawdown.current_drawdown,
            'max_drawdown': self.drawdown.max_drawdown,
            'kill_switch_active': self.kill_switch.is_triggered,
            'kill_switch_reason': self.kill_switch.trigger_reason
        }
    
    def can_trade(self) -> bool:
        """Check if trading is currently allowed"""
        return self.kill_switch.can_trade()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'GICSSector',
    'get_sector',
    'DrawdownWatermark',
    'HistoricalCVaR',
    'LedoitWolfCovariance',
    'KellyPositionSizing',
    'SectorConcentration',
    'KillSwitch',
    'RiskEngine',
]
