from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class OHLCVRecord(BaseModel):
    """
    Pydantic schema for a single market data record.
    Useful for API endpoints, real-time ingestion, or strict row-by-row validation.
    """

    timestamp: datetime
    symbol: str
    open: float = Field(..., ge=0)
    high: float = Field(..., ge=0)
    low: float = Field(..., ge=0)
    close: float = Field(..., ge=0)
    volume: float = Field(..., ge=0)
    adj_close: Optional[float] = Field(None, ge=0)
