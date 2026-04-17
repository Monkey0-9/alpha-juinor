from pydantic import BaseModel, ConfigDict, Field, model_validator
from datetime import datetime
from typing import Optional

class MarketBar(BaseModel):
    """
    Standardized market bar (OHLCV) model.
    Single source of truth for all ingestion, research, and backtesting.
    """
    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Ticker symbol")
    timestamp: datetime = Field(..., description="Bar close time (UTC)")
    open: float = Field(..., gt=0.0)
    high: float = Field(..., gt=0.0)
    low: float = Field(..., gt=0.0)
    close: float = Field(..., gt=0.0)
    volume: float = Field(..., ge=0.0)

    @model_validator(mode='after')
    def validate_ohlc(self) -> 'MarketBar':
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= Open ({self.open}) and Close ({self.close})")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= Open ({self.open}) and Close ({self.close})")
        return self
