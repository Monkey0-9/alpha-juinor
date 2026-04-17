from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import uuid

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class Order(BaseModel):
    """
    Standardized trading order model.
    """
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(..., gt=0.0)
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reason: Optional[str] = None
    metadata: Dict[str, Any] = {}

class Trade(BaseModel):
    """
    Standardized execution fill (trade) model.
    """
    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float = Field(..., gt=0.0)
    price: float = Field(..., gt=0.0)
    commission: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Position(BaseModel):
    """
    Current holding for a specific symbol.
    """
    symbol: str
    quantity: float = 0.0
    average_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0

class PortfolioState(BaseModel):
    """
    Comprehensive portfolio state at a point in time.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cash: float
    equity: float
    positions: Dict[str, Position] = {}
    margin_usage: float = 0.0
    available_funds: float = 0.0
