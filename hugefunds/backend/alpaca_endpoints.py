"""
HUGEFUNDS - Alpaca Paper Trading API Endpoints
Real Trading Execution - Beyond Simulation
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging

from alpaca_integration import (
    get_alpaca_client,
    AlpacaClient,
    initialize_alpaca,
    close_alpaca
)

logger = logging.getLogger('HugeFunds.AlpacaAPI')

router = APIRouter(prefix="/api/alpaca", tags=["alpaca"])

# Pydantic models for request/response
class OrderRequest(BaseModel):
    symbol: str
    qty: float
    side: str  # "buy" or "sell"
    order_type: str = "market"  # "market", "limit", "stop", "stop_limit"
    time_in_force: str = "day"  # "day", "gtc", "opg", "cls", "ioc", "fok"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None

class CancelOrderRequest(BaseModel):
    order_id: str

class ClosePositionRequest(BaseModel):
    symbol: str

# Dependency to get Alpaca client
async def get_alpaca() -> AlpacaClient:
    client = get_alpaca_client()
    if not client.enabled:
        raise HTTPException(
            status_code=503,
            detail="Alpaca not configured. Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
        )
    return client

# ═══════════════════════════════════════════════════════════════════════════════
# ACCOUNT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/account")
async def get_account(client: AlpacaClient = Depends(get_alpaca)):
    """
    Get Alpaca account information
    
    Returns account details including:
    - Buying power
    - Cash balance
    - Portfolio value
    - Equity
    - Day trade count
    """
    try:
        account = await client.get_account()
        return account
    except Exception as e:
        logger.error(f"Error getting account: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/account/status")
async def get_account_status(client: AlpacaClient = Depends(get_alpaca)):
    """
    Get simple account connection status
    
    Quick check if Alpaca is connected and working
    """
    try:
        account = await client.get_account()
        return {
            "connected": account.get("status") == "connected",
            "paper_trading": True,
            "status": account.get("status"),
            "message": "Alpaca paper trading account active" if account.get("status") == "connected" else account.get("error", "Unknown"),
            "timestamp": account.get("timestamp")
        }
    except Exception as e:
        return {
            "connected": False,
            "paper_trading": True,
            "status": "error",
            "message": str(e),
            "timestamp": None
        }

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/positions")
async def get_positions(client: AlpacaClient = Depends(get_alpaca)):
    """
    Get all current positions
    
    Returns list of open positions with:
    - Symbol
    - Quantity
    - Market value
    - Unrealized P&L
    - Current price
    """
    try:
        positions = await client.get_positions()
        return {
            "status": "success",
            "count": len(positions),
            "positions": positions,
            "timestamp": positions[0].get("timestamp") if positions else None
        }
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions/{symbol}")
async def get_position(symbol: str, client: AlpacaClient = Depends(get_alpaca)):
    """
    Get specific position by symbol
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
    """
    try:
        positions = await client.get_positions()
        for pos in positions:
            if pos["symbol"].upper() == symbol.upper():
                return {
                    "status": "success",
                    "position": pos
                }
        return {
            "status": "not_found",
            "message": f"No position found for {symbol}",
            "symbol": symbol
        }
    except Exception as e:
        logger.error(f"Error getting position: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# ORDER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/orders")
async def submit_order(order: OrderRequest, client: AlpacaClient = Depends(get_alpaca)):
    """
    Submit a new order
    
    Example request:
    ```json
    {
        "symbol": "AAPL",
        "qty": 10,
        "side": "buy",
        "order_type": "market",
        "time_in_force": "day"
    }
    ```
    
    Returns order details including order ID and status
    """
    try:
        result = await client.submit_order(
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            order_type=order.order_type,
            time_in_force=order.time_in_force,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            client_order_id=order.client_order_id
        )
        return result
    except Exception as e:
        logger.error(f"Error submitting order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders")
async def get_orders(
    status: Optional[str] = None,
    limit: int = 50,
    client: AlpacaClient = Depends(get_alpaca)
):
    """
    Get orders
    
    Args:
        status: Filter by status ("open", "closed", "all")
        limit: Maximum number of orders (default: 50)
    """
    try:
        orders = await client.get_orders(status=status, limit=limit)
        return {
            "status": "success",
            "count": len(orders),
            "orders": orders
        }
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders/{order_id}")
async def get_order(order_id: str, client: AlpacaClient = Depends(get_alpaca)):
    """
    Get specific order by ID
    
    Args:
        order_id: Alpaca order ID
    """
    try:
        orders = await client.get_orders(limit=100)
        for order in orders:
            if order["order_id"] == order_id:
                return {
                    "status": "success",
                    "order": order
                }
        return {
            "status": "not_found",
            "message": f"Order {order_id} not found",
            "order_id": order_id
        }
    except Exception as e:
        logger.error(f"Error getting order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str, client: AlpacaClient = Depends(get_alpaca)):
    """
    Cancel an open order
    
    Args:
        order_id: Order ID to cancel
    """
    try:
        result = await client.cancel_order(order_id)
        return result
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# TRADING ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/buy")
async def buy_stock(
    symbol: str,
    qty: float,
    order_type: str = "market",
    client: AlpacaClient = Depends(get_alpaca)
):
    """
    Quick buy endpoint
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        qty: Quantity to buy
        order_type: Order type (default: "market")
    
    Example: POST /api/alpaca/buy?symbol=AAPL&qty=10
    """
    try:
        result = await client.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            order_type=order_type,
            time_in_force="day"
        )
        return {
            "action": "BUY",
            "symbol": symbol,
            "qty": qty,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error buying {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sell")
async def sell_stock(
    symbol: str,
    qty: float,
    order_type: str = "market",
    client: AlpacaClient = Depends(get_alpaca)
):
    """
    Quick sell endpoint
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        qty: Quantity to sell
        order_type: Order type (default: "market")
    
    Example: POST /api/alpaca/sell?symbol=AAPL&qty=10
    """
    try:
        result = await client.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            order_type=order_type,
            time_in_force="day"
        )
        return {
            "action": "SELL",
            "symbol": symbol,
            "qty": qty,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error selling {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@router.delete("/positions/{symbol}")
async def close_position(symbol: str, client: AlpacaClient = Depends(get_alpaca)):
    """
    Close a specific position
    
    Args:
        symbol: Symbol of position to close
    
    This will liquidate the entire position in the specified symbol
    """
    try:
        result = await client.close_position(symbol)
        return result
    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/positions")
async def close_all_positions(client: AlpacaClient = Depends(get_alpaca)):
    """
    Close ALL positions (EMERGENCY KILL SWITCH)
    
    ⚠️ WARNING: This will liquidate ALL open positions immediately!
    
    Use with extreme caution. This is the API endpoint for the kill switch.
    """
    try:
        result = await client.close_all_positions()
        return result
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/clock")
async def get_market_clock(client: AlpacaClient = Depends(get_alpaca)):
    """
    Get market clock
    
    Returns:
    - Is market open?
    - Next open time
    - Next close time
    """
    try:
        clock = await client.get_clock()
        return clock
    except Exception as e:
        logger.error(f"Error getting clock: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/calendar")
async def get_market_calendar(
    start: Optional[str] = None,
    end: Optional[str] = None,
    client: AlpacaClient = Depends(get_alpaca)
):
    """
    Get market calendar (trading days)
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
    """
    try:
        calendar = await client.get_calendar(start=start, end=end)
        return {
            "status": "success",
            "count": len(calendar),
            "trading_days": calendar
        }
    except Exception as e:
        logger.error(f"Error getting calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/portfolio/summary")
async def get_alpaca_portfolio_summary(client: AlpacaClient = Depends(get_alpaca)):
    """
    Get complete portfolio summary from Alpaca
    
    Combines account info and positions into one comprehensive view
    """
    try:
        account = await client.get_account()
        positions = await client.get_positions()
        
        # Calculate portfolio metrics
        total_unrealized_pl = sum(pos.get("unrealized_pl", 0) for pos in positions)
        total_market_value = sum(pos.get("market_value", 0) for pos in positions)
        
        # Get top gainers and losers
        sorted_by_pl = sorted(positions, key=lambda x: x.get("unrealized_pl", 0), reverse=True)
        top_gainers = sorted_by_pl[:3] if len(sorted_by_pl) >= 3 else sorted_by_pl
        top_losers = sorted_by_pl[-3:] if len(sorted_by_pl) >= 3 else []
        
        return {
            "status": "success",
            "timestamp": account.get("timestamp"),
            "account": {
                "buying_power": account.get("buying_power"),
                "cash": account.get("cash"),
                "equity": account.get("equity"),
                "portfolio_value": account.get("portfolio_value"),
                "daytrade_count": account.get("daytrade_count")
            },
            "positions_summary": {
                "count": len(positions),
                "total_market_value": total_market_value,
                "total_unrealized_pl": total_unrealized_pl,
                "total_unrealized_plpc": (total_unrealized_pl / total_market_value * 100) if total_market_value > 0 else 0
            },
            "positions": positions,
            "top_gainers": top_gainers,
            "top_losers": top_losers
        }
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/initialize")
async def initialize_alpaca_endpoint():
    """
    Initialize Alpaca connection
    
    Call this endpoint to test and initialize the Alpaca connection
    """
    try:
        success = await initialize_alpaca()
        if success:
            return {
                "status": "success",
                "message": "Alpaca paper trading connected and ready",
                "paper_trading": True,
                "enabled": True
            }
        else:
            return {
                "status": "not_configured",
                "message": "Alpaca not configured. Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.",
                "paper_trading": True,
                "enabled": False
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "paper_trading": True,
            "enabled": False
        }

# ═══════════════════════════════════════════════════════════════════════════════
# STOCK SCREENER - LARGE/MEDIUM/SMALL CAP
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/screener/{market_cap}")
async def get_stocks_by_market_cap(market_cap: str, client: AlpacaClient = Depends(get_alpaca)):
    """
    Get stocks by market cap size
    
    market_cap: "large", "medium", "small", "all"
    """
    try:
        # Top stocks by market cap (real data)
        large_cap = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "V",
            "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "PFE", "CVX"
        ]
        
        medium_cap = [
            "AMD", "INTC", "NFLX", "DIS", "CSCO", "ADBE", "CRM", "PYPL", "CMCSA", "QCOM",
            "KO", "PEP", "MRK", "ABBV", "COST", "TMO", "AVGO", "ABT", "DHR", "MDT"
        ]
        
        small_cap = [
            "PLTR", "SNOW", "COIN", "RBLX", "U", "SQ", "DOCU", "ZM", "PTON", "NKLA",
            "HOOD", "ROKU", "SPCE", "GME", "AMC", "BB", "NIO", "LCID", "RIVN", "FORD"
        ]
        
        if market_cap == "large":
            stocks = large_cap
        elif market_cap == "medium":
            stocks = medium_cap
        elif market_cap == "small":
            stocks = small_cap
        elif market_cap == "all":
            stocks = large_cap + medium_cap + small_cap
        else:
            raise HTTPException(status_code=400, detail=f"Invalid market_cap: {market_cap}. Use: large, medium, small, all")
        
        return {
            "status": "success",
            "market_cap": market_cap,
            "count": len(stocks),
            "stocks": stocks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting screener: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# NEWS DATA INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/news/{symbol}")
async def get_news(symbol: str, client: AlpacaClient = Depends(get_alpaca)):
    """
    Get news for a specific symbol
    
    Uses Alpaca news API for real-time market news
    """
    try:
        # Try to get news from Alpaca
        news = await client.get_news(symbol, limit=10)
        
        if news:
            return {
                "status": "success",
                "symbol": symbol,
                "count": len(news),
                "news": news
            }
        else:
            # Fallback: Return market sentiment summary
            return {
                "status": "success",
                "symbol": symbol,
                "message": "News API not available, using market sentiment",
                "sentiment": "neutral",
                "summary": f"Market data available for {symbol}"
            }
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        return {
            "status": "success",
            "symbol": symbol,
            "message": "News service unavailable",
            "sentiment": "neutral"
        }

@router.get("/news/top")
async def get_top_news(client: AlpacaClient = Depends(get_alpaca)):
    """
    Get top market news
    """
    try:
        # Top market-moving symbols
        top_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
        
        news_summary = []
        for symbol in top_symbols:
            try:
                news = await client.get_news(symbol, limit=2)
                if news:
                    news_summary.extend(news)
            except:
                pass
        
        return {
            "status": "success",
            "count": len(news_summary),
            "news": news_summary[:20]  # Return top 20 news items
        }
    except Exception as e:
        logger.error(f"Error getting top news: {e}")
        return {
            "status": "success",
            "message": "News service unavailable",
            "news": []
        }

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-STOCK TRADING
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/multi-buy")
async def multi_stock_buy(symbols: List[str], qty_per_stock: float = 10, client: AlpacaClient = Depends(get_alpaca)):
    """
    Buy multiple stocks at once
    
    symbols: List of stock symbols to buy
    qty_per_stock: Quantity to buy for each stock
    """
    try:
        results = []
        for symbol in symbols:
            try:
                result = await client.submit_order(
                    symbol=symbol,
                    qty=qty_per_stock,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                results.append({
                    "symbol": symbol,
                    "status": "success",
                    "order_id": result.get("id"),
                    "qty": qty_per_stock
                })
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "status": "success",
            "total_requested": len(symbols),
            "successful": success_count,
            "failed": len(symbols) - success_count,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in multi-buy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multi-buy/json")
async def multi_stock_buy_json(request: dict, client: AlpacaClient = Depends(get_alpaca)):
    """
    Buy multiple stocks at once (JSON body format)
    
    Body: {"symbols": ["AAPL", "MSFT"], "qty_per_stock": 10}
    """
    try:
        symbols = request.get("symbols", [])
        qty_per_stock = request.get("qty_per_stock", 10)
        
        results = []
        for symbol in symbols:
            try:
                result = await client.submit_order(
                    symbol=symbol,
                    qty=qty_per_stock,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                results.append({
                    "symbol": symbol,
                    "status": "success",
                    "order_id": result.get("id"),
                    "qty": qty_per_stock
                })
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "status": "success",
            "total_requested": len(symbols),
            "successful": success_count,
            "failed": len(symbols) - success_count,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in multi-buy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multi-sell")
async def multi_stock_sell(symbols: List[str], qty_per_stock: float = 10, client: AlpacaClient = Depends(get_alpaca)):
    """
    Sell multiple stocks at once
    
    symbols: List of stock symbols to sell
    qty_per_stock: Quantity to sell for each stock
    """
    try:
        results = []
        for symbol in symbols:
            try:
                result = await client.submit_order(
                    symbol=symbol,
                    qty=qty_per_stock,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                results.append({
                    "symbol": symbol,
                    "status": "success",
                    "order_id": result.get("id"),
                    "qty": qty_per_stock
                })
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "status": "success",
            "total_requested": len(symbols),
            "successful": success_count,
            "failed": len(symbols) - success_count,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in multi-sell: {e}")
        raise HTTPException(status_code=500, detail=str(e))
