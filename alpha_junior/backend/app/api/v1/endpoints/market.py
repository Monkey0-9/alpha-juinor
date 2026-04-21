"""
Market Data API Endpoints
Real-time quotes, benchmarks, economic indicators for fund managers
"""

from typing import List, Optional
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.api.deps import (
    get_db, ManagerUser, CurrentUser,
    get_current_user, require_role
)
from app.models.user import UserRole
from app.services.market_data import market_data_service

router = APIRouter(prefix="/market", tags=["Market Data"])


class StockQuoteResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str


class BenchmarkResponse(BaseModel):
    symbol: str
    name: str
    price: float
    daily_return: float
    timestamp: str


class EconomicIndicatorResponse(BaseModel):
    series_id: str
    name: str
    value: float
    date: str
    frequency: str


class CurrencyConversionRequest(BaseModel):
    amount: float
    from_currency: str
    to_currency: str


class CurrencyConversionResponse(BaseModel):
    amount: float
    from_currency: str
    to_currency: str
    rate: float
    converted_amount: float


# Public endpoints (for all authenticated users)
@router.get("/quote/{symbol}", response_model=dict)
async def get_stock_quote(
    symbol: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get real-time stock quote
    """
    quote = await market_data_service.get_stock_quote(symbol.upper())
    
    if not quote:
        raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
    
    return {
        "success": True,
        "data": {
            "symbol": quote.symbol,
            "price": float(quote.price),
            "change": float(quote.change),
            "change_percent": float(quote.change_percent),
            "volume": quote.volume,
            "timestamp": quote.timestamp.isoformat(),
        },
        "error": None,
    }


@router.get("/history/{symbol}", response_model=dict)
async def get_historical_prices(
    symbol: str,
    days: int = 30,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get historical daily prices
    """
    prices = await market_data_service.get_historical_prices(symbol.upper(), days)
    
    if not prices:
        raise HTTPException(status_code=404, detail=f"Historical data not found for {symbol}")
    
    return {
        "success": True,
        "data": [
            {
                "date": p["date"],
                "open": float(p["open"]),
                "high": float(p["high"]),
                "low": float(p["low"]),
                "close": float(p["close"]),
                "volume": p["volume"],
            }
            for p in prices
        ],
        "error": None,
    }


@router.get("/benchmarks", response_model=dict)
async def get_benchmarks(
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get major benchmark index quotes (S&P 500, NASDAQ, Dow, VIX)
    """
    benchmarks = await market_data_service.get_benchmark_quotes()
    
    return {
        "success": True,
        "data": [
            {
                "symbol": b.symbol,
                "name": b.name,
                "price": float(b.price),
                "daily_return": float(b.daily_return),
                "timestamp": b.timestamp.isoformat(),
            }
            for b in benchmarks
        ],
        "error": None,
    }


@router.get("/crypto/{coin_id}", response_model=dict)
async def get_crypto_price(
    coin_id: str,
    currency: str = "usd",
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get cryptocurrency price
    Common coin_ids: bitcoin, ethereum, solana, cardano
    """
    price = await market_data_service.get_crypto_price(coin_id.lower(), currency.lower())
    
    if not price:
        raise HTTPException(status_code=404, detail=f"Price not found for {coin_id}")
    
    return {
        "success": True,
        "data": {
            "coin_id": coin_id,
            "currency": currency.upper(),
            "price": float(price),
        },
        "error": None,
    }


@router.get("/crypto/top", response_model=dict)
async def get_top_cryptocurrencies(
    limit: int = 100,
    currency: str = "usd",
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get top cryptocurrencies by market cap
    """
    data = await market_data_service.get_crypto_market_data(
        vs_currency=currency.lower(),
        per_page=limit
    )
    
    if not data:
        raise HTTPException(status_code=500, detail="Failed to fetch cryptocurrency data")
    
    return {
        "success": True,
        "data": [
            {
                "id": c["id"],
                "symbol": c["symbol"],
                "name": c["name"],
                "price": c["current_price"],
                "market_cap": c["market_cap"],
                "volume_24h": c["total_volume"],
                "change_24h": c["price_change_percentage_24h"],
            }
            for c in data
        ],
        "error": None,
    }


@router.post("/currency/convert", response_model=dict)
async def convert_currency(
    request: CurrencyConversionRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Convert amount between currencies
    """
    converted = await market_data_service.convert_currency(
        amount=Decimal(str(request.amount)),
        from_currency=request.from_currency.upper(),
        to_currency=request.to_currency.upper(),
    )
    
    if not converted:
        raise HTTPException(status_code=400, detail="Currency conversion failed")
    
    # Get rate for reference
    rate = await market_data_service.get_exchange_rate(
        request.from_currency.upper(),
        request.to_currency.upper()
    )
    
    return {
        "success": True,
        "data": {
            "amount": request.amount,
            "from_currency": request.from_currency.upper(),
            "to_currency": request.to_currency.upper(),
            "rate": float(rate) if rate else None,
            "converted_amount": float(converted),
        },
        "error": None,
    }


@router.get("/economic-indicators", response_model=dict)
async def get_economic_indicators(
    current_user: ManagerUser = Depends(require_role([UserRole.MANAGER])),
):
    """
    Get key economic indicators (Fed rate, unemployment, inflation)
    """
    indicators = await market_data_service.get_key_economic_indicators()
    
    return {
        "success": True,
        "data": indicators,
        "error": None,
    }


@router.get("/fred/{series_id}", response_model=dict)
async def get_fred_data(
    series_id: str,
    limit: int = 10,
    current_user: ManagerUser = Depends(require_role([UserRole.MANAGER])),
):
    """
    Get economic data from FRED (Federal Reserve)
    
    Common series IDs:
    - DFF: Federal Funds Rate
    - T10Y2Y: 10Y-2Y Yield Spread
    - UNRATE: Unemployment Rate
    - CPIAUCSL: Consumer Price Index
    - GDP: Gross Domestic Product
    """
    data = await market_data_service.get_fred_series(series_id, limit)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"FRED series {series_id} not found")
    
    return {
        "success": True,
        "data": [
            {
                "series_id": d.series_id,
                "name": d.name,
                "value": float(d.value),
                "date": d.date.isoformat(),
                "frequency": d.frequency,
            }
            for d in data
        ],
        "error": None,
    }


@router.get("/news", response_model=dict)
async def get_financial_news(
    query: str = "finance OR investing OR hedge fund",
    page_size: int = 10,
    current_user: ManagerUser = Depends(require_role([UserRole.MANAGER])),
):
    """
    Get financial news
    """
    news = await market_data_service.get_financial_news(query, page_size=page_size)
    
    if not news:
        raise HTTPException(status_code=500, detail="Failed to fetch news")
    
    return {
        "success": True,
        "data": [
            {
                "title": a["title"],
                "description": a["description"],
                "url": a["url"],
                "published_at": a["publishedAt"],
                "source": a["source"]["name"] if isinstance(a["source"], dict) else a["source"],
            }
            for a in news
        ],
        "error": None,
    }
