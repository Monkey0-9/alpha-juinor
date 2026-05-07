from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import pandas as pd
import numpy as np

from nexus.core.alpha import AlphaEngine
from nexus.core.intelligence import MarketBrain
from nexus.research.backtest import BacktestEngine
from nexus.execution.alpaca import get_client
from nexus.math.risk import RiskEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitor", tags=["monitoring"])

class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str = "1D"
    lookback: int = 120
    entry_size: float = 0.1


@router.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Nexus Monitoring",
        "timestamp": pd.Timestamp.utcnow().isoformat()
    }


@router.get("/brain")
async def brain_snapshot(symbol: str = "SPY", timeframe: str = "1D", lookback: int = 120) -> Dict[str, Any]:
    alpha_engine = AlphaEngine()
    brain = MarketBrain()
    bars = await alpha_engine.fetch_market_data(symbol, timeframe=timeframe, limit=lookback)
    if bars.empty:
        raise HTTPException(status_code=502, detail="Unable to fetch market bars for brain analysis.")

    positions = []
    try:
        positions = await get_client().get_positions()
    except Exception as exc:
        logger.warning(f"Unable to load live positions for brain snapshot: {exc}")

    analysis = brain.analyze_market(bars, positions)
    returns = bars["close"].pct_change().dropna().to_numpy()
    risk_engine = RiskEngine()
    analysis["risk_profile"] = risk_engine.assess_risk(returns)
    analysis["market_depth"] = {
        "universe_size": len(bars),
        "recent_volatility": float(np.std(returns)) if len(returns) > 0 else 0.0,
    }
    return {"status": "success", "analysis": analysis}


@router.post("/backtest")
async def run_backtest(request: BacktestRequest) -> Dict[str, Any]:
    alpha_engine = AlphaEngine()
    bars = await alpha_engine.fetch_market_data(request.symbol, timeframe=request.timeframe, limit=request.lookback)
    if bars.empty or "close" not in bars.columns:
        raise HTTPException(status_code=502, detail="Insufficient historical data for backtest.")

    prices = bars["close"].astype(float)
    signal = pd.Series(0.0, index=prices.index)
    window = min(20, len(prices) - 1)
    for idx in range(window, len(prices)):
        momentum = prices.iloc[idx] / prices.iloc[idx - window] - 1
        signal.iloc[idx] = float(np.tanh(momentum * 10))

    backtester = BacktestEngine()
    result = backtester.run(prices, signal, entry_size=request.entry_size)
    return {
        "status": "success",
        "metrics": result.metrics,
        "trades": result.trades.tail(20).to_dict(orient="records"),
        "equity_curve": result.equity_curve.tail(20).to_dict()
    }
