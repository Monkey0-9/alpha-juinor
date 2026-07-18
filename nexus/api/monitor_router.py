"""
nexus/api/monitor_router.py — Superhuman Brain API Endpoints

Exposes all new intelligence data to the dashboard and external clients:
  GET /api/monitor/health              — service health
  GET /api/monitor/brain               — full market intelligence snapshot
  GET /api/monitor/brain/superhuman    — conviction signals per symbol
  POST /api/monitor/backtest           — symbol backtest
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import logging
import pandas as pd
import numpy as np

from nexus.core.alpha import AlphaEngine
from nexus.core.intelligence import MarketBrain
from nexus.core.superhuman_brain import SuperhumanBrain
from nexus.research.backtest import BacktestEngine
from nexus.execution.alpaca import get_client
from nexus.math.risk import RiskEngine
from nexus.math.indicators import RegimeDetector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitor", tags=["monitoring"])

# Module-level singletons — reused across requests for warm IC/Bayesian state
_market_brain = MarketBrain()
_superhuman_brain = SuperhumanBrain()
_regime_detector = RegimeDetector()


class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str = "1D"
    lookback: int = 120
    entry_size: float = 0.1


@router.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Nexus Monitoring — SuperhumanBrain Edition",
        "version": "3.0.0",
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }


@router.get("/brain")
async def brain_snapshot(
    symbol: str = "SPY",
    timeframe: str = "1D",
    lookback: int = 120,
) -> Dict[str, Any]:
    """
    Full market intelligence snapshot — now includes:
    - Probabilistic regime distribution (BULL/BEAR/SIDEWAYS/TURBULENT)
    - Hurst exponent
    - Forecast confidence
    - Correlation pulse
    - Comprehensive risk profile
    """
    alpha_engine = AlphaEngine()
    bars = await alpha_engine.fetch_market_data(
        symbol, timeframe=timeframe, limit=lookback
    )
    if bars.empty:
        raise HTTPException(
            status_code=502,
            detail="Unable to fetch market bars for brain analysis.",
        )

    positions: List[Dict[str, Any]] = []
    try:
        positions = await get_client().get_positions()
    except Exception as exc:
        logger.warning(f"Unable to load live positions for brain snapshot: {exc}")

    analysis = _market_brain.analyze_market(bars, positions)

    # Risk profile
    returns = bars["close"].pct_change().dropna().to_numpy()
    risk_engine = RiskEngine()
    analysis["risk_profile"] = risk_engine.assess_risk(returns)

    # Market depth with Hurst
    volatility = float(np.std(returns)) if len(returns) > 0 else 0.0
    analysis["market_depth"] = {
        "universe_size": len(bars),
        "recent_volatility": volatility,
    }

    # Ensure regime_probabilities is serializable
    regime_probs = analysis.get("regime_probabilities", {})
    analysis["regime_probabilities"] = {
        k: round(float(v), 4) for k, v in regime_probs.items()
    }

    return {"status": "success", "analysis": analysis}


@router.get("/brain/superhuman")
async def superhuman_snapshot(
    symbols: str = "AAPL,MSFT,NVDA,SPY,QQQ",
    timeframe: str = "15Min",
) -> Dict[str, Any]:
    """
    Superhuman Brain conviction snapshot for a comma-separated list of symbols.

    Returns per-symbol conviction grade, score, Fractal gate status, IR score,
    strategy votes, and the portfolio intelligence report.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided.")

    alpha_engine = AlphaEngine()

    # Fetch raw signals and history for all symbols
    raw_signals = await alpha_engine.get_batch_signals(symbol_list, timeframe=timeframe)

    # Fetch history for fractal gate + strategy scoring
    history: Dict[str, pd.DataFrame] = {}
    for sym in symbol_list:
        df = await alpha_engine.fetch_market_data(sym, timeframe="1D", limit=80)
        if not df.empty:
            history[sym] = df

    # Detect regime probabilities from SPY benchmark
    benchmark = history.get("SPY", pd.DataFrame())
    if benchmark.empty and "SPY" not in symbol_list:
        benchmark = await alpha_engine.fetch_market_data("SPY", timeframe="1D", limit=80)

    regime_probs = _regime_detector.detect_probabilities(benchmark)
    current_regime = max(regime_probs, key=lambda k: regime_probs[k])

    # Run SuperhumanBrain evaluation
    conviction_signals = _superhuman_brain.evaluate_portfolio(
        raw_signals, history, regime_probs, current_regime
    )

    # Portfolio intelligence report
    intel_report = _superhuman_brain.portfolio_intelligence_report(conviction_signals)

    # Serialize conviction signals
    signals_out: Dict[str, Any] = {}
    for sym, cs in conviction_signals.items():
        signals_out[sym] = {
            "score":          round(cs.score, 4),
            "conviction":     round(cs.conviction, 4),
            "conviction_pct": f"{cs.conviction:.0%}",
            "grade":          cs.conviction_grade(),
            "gate_pass":      cs.gate_pass,
            "regime_bias":    cs.regime_bias,
            "ir_score":       round(cs.ir_score, 4),
            "reasoning":      cs.reasoning,
            "strategy_votes": {
                k: round(v, 3)
                for k, v in cs.strategy_votes.items()
            },
        }

    return {
        "status":           "success",
        "regime":           current_regime,
        "regime_probs":     {k: round(float(v), 4) for k, v in regime_probs.items()},
        "signals":          signals_out,
        "intelligence_report": {
            "avg_conviction":      round(intel_report.get("avg_conviction", 0), 4),
            "max_conviction":      round(intel_report.get("max_conviction", 0), 4),
            "avg_signal_strength": round(intel_report.get("avg_signal_strength", 0), 4),
            "gate_pass_rate":      round(intel_report.get("gate_pass_rate", 0), 4),
            "global_ic":           round(intel_report.get("global_ic", 0), 4),
            "a_grade_signals":     intel_report.get("a_grade_signals", 0),
            "total_signals":       intel_report.get("total_signals", 0),
        },
    }


@router.post("/backtest")
async def run_backtest(request: BacktestRequest) -> Dict[str, Any]:
    """Run a symbol backtest using the alpha signal engine."""
    alpha_engine = AlphaEngine()
    bars = await alpha_engine.fetch_market_data(
        request.symbol,
        timeframe=request.timeframe,
        limit=request.lookback,
    )
    if bars.empty or "close" not in bars.columns:
        raise HTTPException(
            status_code=502,
            detail="Insufficient historical data for backtest.",
        )

    prices = bars["close"].astype(float)
    signal = pd.Series(0.0, index=prices.index)
    window = min(20, len(prices) - 1)
    for idx in range(window, len(prices)):
        momentum = prices.iloc[idx] / prices.iloc[idx - window] - 1
        signal.iloc[idx] = float(np.tanh(momentum * 10))

    backtester = BacktestEngine()
    result = backtester.run(prices, signal, entry_size=request.entry_size)
    return {
        "status":       "success",
        "metrics":      result.metrics,
        "trades":       result.trades.tail(20).to_dict(orient="records"),
        "equity_curve": result.equity_curve.tail(20).to_dict(),
    }
