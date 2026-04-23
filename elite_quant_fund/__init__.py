"""
Elite Quant Fund System
World-class quantitative trading system at Renaissance Technologies / Jane Street level

Core Components:
- Data Pipeline: Kalman filtering, Yang-Zhang volatility, Amihud illiquidity
- Alpha Engine: OU stat arb, LightGBM ensemble, IC-weighted blending
- Risk Engine: CVaR, Ledoit-Wolf, Fractional Kelly, kill switch
- Portfolio Optimizer: Black-Litterman, Risk Parity, Min Variance
- Execution Engine: Almgren-Chriss, VWAP, Smart Order Router

Version: 1.0.0
Author: Elite Quant Team
License: Proprietary
"""

__version__ = "1.0.0"
__author__ = "Elite Quant Team"

from elite_quant_fund.core.types import (
    # Monads
    Result,
    # Enums
    Side, OrderType, TimeInForce, Venue, SignalType, RiskBreachType,
    # Market Data
    MarketBar, Quote, Trade,
    # Alpha
    AlphaSignal, SignalBundle,
    # Risk
    Position, RiskBreach, RiskLimits,
    # Portfolio
    Portfolio, TargetAllocation,
    # Execution
    Order, Fill, ExecutionSchedule,
    # Microstructure
    MarketImpactEstimate, LiquidityMetrics,
    # State
    KalmanState, VolatilityState, OUState,
    # Utilities
    calc_cvar, ledoit_wolf_shrinkage, fractional_kelly_size,
)

from elite_quant_fund.system import (
    SystemConfig,
    SystemState,
    EliteQuantFund,
    create_elite_quant_fund,
)

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Core Types
    'Result',
    'Side', 'OrderType', 'TimeInForce', 'Venue', 'SignalType', 'RiskBreachType',
    'MarketBar', 'Quote', 'Trade',
    'AlphaSignal', 'SignalBundle',
    'Position', 'RiskBreach', 'RiskLimits',
    'Portfolio', 'TargetAllocation',
    'Order', 'Fill', 'ExecutionSchedule',
    'MarketImpactEstimate', 'LiquidityMetrics',
    'KalmanState', 'VolatilityState', 'OUState',
    'calc_cvar', 'ledoit_wolf_shrinkage', 'fractional_kelly_size',
    
    # System
    'SystemConfig',
    'SystemState',
    'EliteQuantFund',
    'create_elite_quant_fund',
]
