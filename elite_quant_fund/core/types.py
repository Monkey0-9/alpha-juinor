"""
Core Domain Types - Elite Quant Fund System
Institutional-grade type system with invariant enforcement
Built to Renaissance Technologies / Jane Street standards
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Tuple, Union, Generic, TypeVar, 
    Callable, Any, Literal, Protocol
)
from dataclasses import dataclass
from functools import total_ordering
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict

# ============================================================================
# MONADIC RESULT TYPE - Eliminates exception-driven control flow
# ============================================================================

T = TypeVar('T')
E = TypeVar('E', bound=Exception)

class Result(Generic[T]):
    """Result monad for error handling without exceptions in hot paths"""
    
    def __init__(self, value: Optional[T] = None, error: Optional[str] = None):
        self._value = value
        self._error = error
        self._is_ok = error is None and value is not None
    
    @classmethod
    def ok(cls, value: T) -> Result[T]:
        return cls(value=value)
    
    @classmethod
    def err(cls, error: str) -> Result[T]:
        return cls(error=error)
    
    @property
    def is_ok(self) -> bool:
        return self._is_ok
    
    @property
    def is_err(self) -> bool:
        return not self._is_ok
    
    def unwrap(self) -> T:
        if self.is_err:
            raise ValueError(f"Called unwrap on Err: {self._error}")
        return self._value  # type: ignore
    
    def unwrap_or(self, default: T) -> T:
        return self._value if self.is_ok else default
    
    def map(self, f: Callable[[T], T]) -> Result[T]:
        if self.is_ok:
            return Result.ok(f(self._value))  # type: ignore
        return self
    
    def bind(self, f: Callable[[T], Result[T]]) -> Result[T]:
        if self.is_ok:
            return f(self._value)  # type: ignore
        return self
    
    def __repr__(self) -> str:
        if self.is_ok:
            return f"Ok({self._value})"
        return f"Err({self._error})"


# ============================================================================
# ENUMERATIONS
# ============================================================================

class Side(Enum):
    BUY = auto()
    SELL = auto()
    
    def __repr__(self) -> str:
        return self.name

class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    VWAP = auto()
    TWAP = auto()
    IMPLEMENTATION_SHORTFALL = auto()
    ALMGREN_CHRISS = auto()
    ICEBERG = auto()
    PEGGED = auto()

class TimeInForce(Enum):
    GTC = auto()  # Good Till Cancelled
    DAY = auto()
    IOC = auto()  # Immediate Or Cancel
    FOK = auto()  # Fill Or Kill
    GTX = auto()  # Good Till Crossing

class Venue(Enum):
    NYSE = auto()
    NASDAQ = auto()
    BATS = auto()
    IEX = auto()
    DARK_POOL_SIGMA = auto()
    DARK_POOL_MS = auto()
    DARK_POOL_UBS = auto()

class SignalType(Enum):
    MOMENTUM = auto()
    MEAN_REVERSION = auto()
    STATISTICAL_ARBITRAGE = auto()
    FACTOR = auto()
    MACHINE_LEARNING = auto()
    MACRO = auto()

class RiskBreachType(Enum):
    CVAR_LIMIT = auto()
    POSITION_LIMIT = auto()
    LEVERAGE_LIMIT = auto()
    SECTOR_CONCENTRATION = auto()
    DRAWDOWN_LIMIT = auto()
    KELLY_FRACTION = auto()
    LIQUIDITY_LIMIT = auto()
    KILL_SWITCH = auto()


# ============================================================================
# CORE MARKET DATA TYPES
# ============================================================================

class MarketBar(BaseModel):
    """
    OHLCV bar with invariant enforcement
    Invariants: high >= max(open, close, low), low <= min(open, close)
    """
    model_config = ConfigDict(frozen=True)
    
    symbol: str = Field(..., min_length=1, max_length=10)
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    vwap: Optional[float] = None
    trades: Optional[int] = None
    
    @root_validator(skip_on_failure=True)
    def validate_ohlc_invariants(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        o = values.get('open', 0)
        h = values.get('high', 0)
        l = values.get('low', 0)
        c = values.get('close', 0)
        
        if h < max(o, c, l):
            raise ValueError(f"Invalid OHLC: high({h}) < max(o={o}, c={c}, l={l})")
        if l > min(o, c):
            raise ValueError(f"Invalid OHLC: low({l}) > min(o={o}, c={c})")
        if l > h:
            raise ValueError(f"Invalid OHLC: low({l}) > high({h})")
        
        return values
    
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def returns(self) -> float:
        return (self.close - self.open) / self.open


class Quote(BaseModel):
    """Level 1 quote with spread validation"""
    model_config = ConfigDict(frozen=True)
    
    symbol: str
    timestamp: datetime
    bid: float = Field(..., gt=0)
    ask: float = Field(..., gt=0)
    bid_size: int = Field(..., ge=0)
    ask_size: int = Field(..., ge=0)
    
    @validator('ask')
    def ask_greater_than_bid(cls, v: float, values: Dict[str, Any]) -> float:
        if 'bid' in values and v <= values['bid']:
            raise ValueError(f"Ask({v}) must be > bid({values['bid']})")
        return v
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread_bps(self) -> float:
        return 10000 * self.spread / self.mid


class Trade(BaseModel):
    """Individual trade print"""
    model_config = ConfigDict(frozen=True)
    
    symbol: str
    timestamp: datetime
    price: float = Field(..., gt=0)
    size: int = Field(..., gt=0)
    side: Optional[Side] = None  # None if not reported
    venue: Optional[Venue] = None
    

# ============================================================================
# ALPHA SIGNAL TYPES
# ============================================================================

class AlphaSignal(BaseModel):
    """
    Quantitative alpha signal with strength in [-1, 1]
    Strength represents confidence-weighted direction
    """
    model_config = ConfigDict(frozen=True)
    
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    strength: float = Field(..., ge=-1.0, le=1.0)
    horizon: timedelta  # Expected holding period
    half_life: Optional[timedelta] = None  # For mean-reverting signals
    metadata: Dict[str, float] = Field(default_factory=dict)
    
    @validator('strength')
    def validate_strength(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Signal strength must be in [-1, 1], got {v}")
        return v
    
    @property
    def is_bullish(self) -> bool:
        return self.strength > 0
    
    @property
    def is_bearish(self) -> bool:
        return self.strength < 0
    
    @property
    def confidence(self) -> float:
        """Absolute strength as confidence measure"""
        return abs(self.strength)


class SignalBundle(BaseModel):
    """Collection of alpha signals for multiple symbols"""
    model_config = ConfigDict(frozen=True)
    
    timestamp: datetime
    signals: Dict[str, List[AlphaSignal]] = Field(default_factory=dict)
    
    def get_consensus(self, symbol: str) -> Optional[AlphaSignal]:
        """IC-weighted consensus signal for a symbol"""
        if symbol not in self.signals or not self.signals[symbol]:
            return None
        
        sigs = self.signals[symbol]
        total_weight = sum(abs(s.strength) for s in sigs)
        if total_weight == 0:
            return None
        
        consensus_strength = sum(s.strength for s in sigs) / len(sigs)
        return AlphaSignal(
            symbol=symbol,
            timestamp=self.timestamp,
            signal_type=SignalType.FACTOR,
            strength=np.clip(consensus_strength, -1, 1),
            horizon=timedelta(hours=1),
            metadata={'sources': len(sigs), 'total_weight': total_weight}
        )


# ============================================================================
# RISK TYPES
# ============================================================================

class Position(BaseModel):
    """Position with P&L tracking"""
    model_config = ConfigDict(frozen=True)
    
    symbol: str
    quantity: int
    entry_price: float = Field(..., gt=0)
    entry_time: datetime
    current_price: float = Field(..., gt=0)
    unrealized_pnl: float
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price
    
    def update_price(self, new_price: float) -> Position:
        """Return new position with updated price (immutable)"""
        unrealized = self.quantity * (new_price - self.entry_price)
        return self.model_copy(update={
            'current_price': new_price,
            'unrealized_pnl': unrealized
        })


class RiskBreach(BaseModel):
    """Typed risk breach event for handlers"""
    model_config = ConfigDict(frozen=True)
    
    breach_type: RiskBreachType
    timestamp: datetime
    severity: float = Field(..., ge=0, le=1)  # 0 = info, 1 = critical
    description: str
    metric_value: float
    threshold: float
    symbol: Optional[str] = None
    position: Optional[Position] = None
    
    @property
    def is_critical(self) -> bool:
        return self.severity > 0.8


class RiskLimits(BaseModel):
    """Risk limit configuration"""
    model_config = ConfigDict(frozen=True)
    
    max_position_value: float = Field(default=1_000_000, gt=0)
    max_leverage: float = Field(default=2.0, gt=0)
    max_drawdown_pct: float = Field(default=0.05, gt=0, le=1)
    max_cvar_95: float = Field(default=0.02, gt=0, le=1)
    max_sector_concentration: float = Field(default=0.25, gt=0, le=1)
    kelly_fraction: float = Field(default=0.3, gt=0, le=1)
    kill_switch_drawdown: float = Field(default=0.10, gt=0, le=1)


# ============================================================================
# PORTFOLIO TYPES
# ============================================================================

class Portfolio(BaseModel):
    """Portfolio state with optimization targets"""
    model_config = ConfigDict(frozen=True)
    
    timestamp: datetime
    positions: Dict[str, Position] = Field(default_factory=dict)
    cash: float = Field(default=0.0)
    total_value: float = Field(default=0.0)
    
    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())
    
    @property
    def net_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def leverage(self) -> float:
        if self.total_value == 0:
            return 0.0
        return self.gross_exposure / self.total_value
    
    @property
    def long_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values() if p.quantity > 0)
    
    @property
    def short_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values() if p.quantity < 0)


class TargetAllocation(BaseModel):
    """Target weights from optimizer"""
    model_config = ConfigDict(frozen=True)
    
    timestamp: datetime
    weights: Dict[str, float]  # Symbol -> target weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: Literal['black_litterman', 'risk_parity', 'min_variance', 'max_div']
    
    @property
    def is_fully_invested(self) -> bool:
        return abs(sum(self.weights.values()) - 1.0) < 1e-6
    
    @validator('weights')
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        for sym, w in v.items():
            if not -1 <= w <= 1:
                raise ValueError(f"Weight for {sym} must be in [-1, 1], got {w}")
        return v


# ============================================================================
# ORDER & EXECUTION TYPES
# ============================================================================

class Order(BaseModel):
    """Order with full specification"""
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(default_factory=lambda: f"ord_{datetime.now().timestamp()}")
    symbol: str
    side: Side
    quantity: int = Field(..., gt=0)
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Algo parameters
    vwap_start_time: Optional[datetime] = None
    vwap_end_time: Optional[datetime] = None
    twap_intervals: Optional[int] = None
    urgency: float = Field(default=0.5, ge=0, le=1)
    max_participation_pct: float = Field(default=0.1, ge=0, le=1)


class Fill(BaseModel):
    """Execution fill report"""
    model_config = ConfigDict(frozen=True)
    
    order_id: str
    symbol: str
    timestamp: datetime
    price: float = Field(..., gt=0)
    quantity: int = Field(..., gt=0)
    side: Side
    venue: Venue
    fees: float = Field(default=0.0)
    
    @property
    def value(self) -> float:
        return self.price * self.quantity


class ExecutionSchedule(BaseModel):
    """Almgren-Chriss or VWAP execution schedule"""
    model_config = ConfigDict(frozen=True)
    
    order_id: str
    symbol: str
    total_quantity: int
    side: Side
    start_time: datetime
    end_time: datetime
    schedule: List[Tuple[datetime, int]]  # (time, quantity)
    expected_impact_bps: float
    expected_variance: float
    
    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time
    
    @property
    def num_slices(self) -> int:
        return len(self.schedule)


# ============================================================================
# MARKET IMPACT & MICROSTRUCTURE
# ============================================================================

class MarketImpactEstimate(BaseModel):
    """Market impact estimation using Almgren-Chriss"""
    model_config = ConfigDict(frozen=True)
    
    permanent_impact_bps: float
    temporary_impact_bps: float
    total_impact_bps: float
    decay_time: timedelta
    confidence: float = Field(..., ge=0, le=1)


class LiquidityMetrics(BaseModel):
    """Real-time liquidity estimation"""
    model_config = ConfigDict(frozen=True)
    
    symbol: str
    timestamp: datetime
    bid_ask_spread_bps: float
    amihud_ratio: float  # Illiquidity ratio
    kyle_lambda: float   # Price impact coefficient
    market_depth_dollars: float
    adv_20_day: float   # Average daily volume
    
    @property
    def is_liquid(self) -> bool:
        return self.amihud_ratio < 0.01 and self.bid_ask_spread_bps < 10


# ============================================================================
# KALMAN FILTER STATE
# ============================================================================

class KalmanState(BaseModel):
    """Kalman filter state for online price estimation"""
    model_config = ConfigDict(frozen=True)
    
    symbol: str
    timestamp: datetime
    x_hat: float  # State estimate (true price)
    P: float  # Error covariance
    Q: float  # Process noise (adaptive)
    R: float  # Measurement noise (adaptive)
    K: float  # Kalman gain
    
    def predict(self, dt: float = 1.0) -> KalmanState:
        """Prediction step (state evolves with random walk)"""
        x_pred = self.x_hat
        P_pred = self.P + self.Q
        return self.model_copy(update={'x_hat': x_pred, 'P': P_pred})
    
    def update(self, measurement: float) -> KalmanState:
        """Update step with new observation"""
        y = measurement - self.x_hat  # Innovation
        S = self.P + self.R  # Innovation covariance
        K = self.P / S  # Kalman gain
        
        x_new = self.x_hat + K * y
        P_new = (1 - K) * self.P
        
        return self.model_copy(update={
            'x_hat': x_new,
            'P': P_new,
            'K': K
        })


# ============================================================================
# VOLATILITY ESTIMATOR STATE
# ============================================================================

class VolatilityState(BaseModel):
    """Yang-Zhang minimum variance volatility estimator state"""
    model_config = ConfigDict(frozen=True)
    
    symbol: str
    timestamp: datetime
    volatility_annual: float = Field(..., ge=0)
    overnight_vol: float = Field(..., ge=0)
    open_close_vol: float = Field(..., ge=0)
    min_variance_estimator: str = Field(default="yang_zhang")
    
    @property
    def daily_vol(self) -> float:
        return self.volatility_annual / np.sqrt(252)


# ============================================================================
# OU PROCESS STATE (Mean Reversion)
# ============================================================================

class OUState(BaseModel):
    """Ornstein-Uhlenbeck process state for statistical arbitrage"""
    model_config = ConfigDict(frozen=True)
    
    symbol: str
    timestamp: datetime
    mean: float  # Long-term mean (theta)
    speed: float  # Mean reversion speed (kappa)
    volatility: float  # Volatility (sigma)
    half_life: timedelta
    z_score: float  # Current distance from mean
    
    @property
    def is_mean_reverting(self) -> bool:
        return self.speed > 0 and self.half_life.days < 30
    
    def expected_return(self, horizon: timedelta) -> float:
        """Expected return given OU dynamics"""
        t = horizon.days
        decay = np.exp(-self.speed * t)
        return self.mean + (self.z_score - self.mean) * decay


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calc_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR) at confidence level alpha"""
    var = np.percentile(returns, alpha * 100)
    return np.mean(returns[returns <= var])


def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf covariance shrinkage estimator"""
    t, n = returns.shape
    sample_cov = np.cov(returns, rowvar=False)
    
    # Target: shrink toward constant correlation model
    mean_var = np.mean(np.diag(sample_cov))
    target = np.eye(n) * mean_var
    
    # Shrinkage intensity (simplified - full version uses Frobenius norm)
    shrinkage = min(1.0, t / (t + n * 10))
    
    return shrinkage * target + (1 - shrinkage) * sample_cov


def fractional_kelly_size(edge: float, odds: float, fraction: float = 0.3) -> float:
    """Fractional Kelly criterion for position sizing"""
    if odds <= 0:
        return 0.0
    kelly = edge / odds
    return fraction * kelly


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Monads
    'Result', 'T', 'E',
    # Enums
    'Side', 'OrderType', 'TimeInForce', 'Venue', 'SignalType', 'RiskBreachType',
    # Market Data
    'MarketBar', 'Quote', 'Trade',
    # Alpha
    'AlphaSignal', 'SignalBundle',
    # Risk
    'Position', 'RiskBreach', 'RiskLimits',
    # Portfolio
    'Portfolio', 'TargetAllocation',
    # Execution
    'Order', 'Fill', 'ExecutionSchedule',
    # Microstructure
    'MarketImpactEstimate', 'LiquidityMetrics',
    # State
    'KalmanState', 'VolatilityState', 'OUState',
    # Utilities
    'calc_cvar', 'ledoit_wolf_shrinkage', 'fractional_kelly_size',
]
