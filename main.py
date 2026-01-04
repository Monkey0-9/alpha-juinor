# main.py
"""
MINI QUANT FUND — INSTITUTIONAL DRIVER (production-grade)

Drop-in main driver that:
- Instantiates provider/datastore/registry if available
- Instantiates execution handler and engine (compatible with institutional execution)
- Runs a monthly-rebalance strategy using alpha + risk components (fallbacks available)
- Persists immutably auditable run artifacts (trades.csv, equity.csv, config.json, meta.json, requirements.txt)
- Integrates with BacktestRegistry when present

Usage:
    python main.py
"""
import sys
import os
import json
import shutil
import tempfile
import hashlib
import stat
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------
# Project imports (best-effort)
# ---------------------------
# Try import user modules; if missing, provide light fallbacks
try:
    from data.provider import YahooDataProvider  # optional
except Exception:
    YahooDataProvider = None

# AlpacaDataProvider removed to enforce Yahoo-only data policy (Institutional Requirement)
AlpacaDataProvider = None

try:
    from brokers.alpaca_broker import AlpacaExecutionHandler
except Exception:
    AlpacaExecutionHandler = None

try:
    from portfolio.ledger import PortfolioLedger, PortfolioEvent, EventType
except Exception:
    PortfolioLedger = None
    PortfolioEvent = None
    EventType = None

try:
    from portfolio.allocator import InstitutionalAllocator
except Exception as e:
    InstitutionalAllocator = None
    # log error?
    pass

try:
    from data.storage import DataStore
except Exception:
    DataStore = None

# Backtest engine & execution - we expect the institutional files present
try:
    from backtest.engine import BacktestEngine
except Exception:
    BacktestEngine = None

try:
    from backtest.execution import RealisticExecutionHandler, Order, OrderType, TradeBlotter
except Exception:
    RealisticExecutionHandler = None
    Order = None
    OrderType = None
    TradeBlotter = None

# Registry (optional)
try:
    from backtest.registry import BacktestRegistry
except Exception:
    BacktestRegistry = None

# Analytics / reporting (optional)
try:
    from engine.analytics import (
        annualized_return,
        annualized_volatility,
        sharpe_ratio,
        max_drawdown,
    )
except Exception:
    annualized_return = None
    annualized_volatility = None
    sharpe_ratio = None
    max_drawdown = None

# Alpha & Risk (optional)
# Alpha & Risk & ML
try:
    from strategies.alpha import (
        CompositeAlpha, 
        TrendAlpha, 
        MeanReversionAlpha, 
        RSIAlpha, 
        MACDAlpha, 
        BollingerBandAlpha
    )
    from strategies.features import FeatureEngineer
    from strategies.ml_alpha import MLAlpha
except Exception:
    CompositeAlpha = TrendAlpha = MeanReversionAlpha = RSIAlpha = MACDAlpha = BollingerBandAlpha = None
    FeatureEngineer = MLAlpha = None

try:
    from risk.engine import RiskManager, RiskRegime, RiskDecision
except Exception:
    RiskManager = None
    RiskRegime = None
    RiskDecision = None
    
try:
    from portfolio.optimizer import MeanVarianceOptimizer
except Exception:
    MeanVarianceOptimizer = None

try:
    from risk.factor_model import StatisticalRiskModel
except Exception:
    StatisticalRiskModel = None

try:
    from strategies.stat_arb import KalmanPairsTrader
except Exception:
    KalmanPairsTrader = None

try:
    from reports.performance_attribution import PerformanceAnalyzer
except Exception:
    PerformanceAnalyzer = None

try:
    from data.validator import DataValidator
except Exception:
    DataValidator = None

# yfinance fallback for history
import yfinance as yf

# ---------------------------
# Configuration (adjust to your fund)
# ---------------------------
TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
START_DATE = "2018-01-01"
END_DATE = None  # set if needed "2024-12-31"
INITIAL_CAPITAL = 1_000_000.0

# "High Accuracy" Settings
ML_CONFIDENCE_THRESHOLD = 0.65  # Confidence required to act on a signal
EQUITY_DRIFT_TOLERANCE = 0.05   # 5% max mismatch before Kill Switch triggers
REBALANCE_PERIOD = "monthly"    # Rebalance frequency

COMMISSION_PCT = 0.0005  # 0.05%
IMPACT_COEFF = 0.15
MAX_PARTICIPATION = 0.10
ADV_LOOKBACK = 20
VOL_LOOKBACK = 21
MIN_VOL_FALLBACK = 0.02

MIN_TRADE_PCT = 0.02  # avoid tiny trades (<2% of equity)
RUN_OUTPUT_ROOT = Path("output/backtests")
OUTPUT_DIR = RUN_OUTPUT_ROOT # Define OUTPUT_DIR for logging setup

# ---------------------------
# Utility: atomic writes & checksums
# ---------------------------
def _atomic_write_text(path: Path, text: str, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".txt")
    os.close(fd)
    try:
        with open(tmp, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def _atomic_write_df(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        with open(tmp, "rb") as f:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_run_dir(base: Path, prefix: str = "run") -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    candidate = base / f"{prefix}_{ts}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base / f"{prefix}_{ts}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

# ---------------------------
# Fallback components (if missing)
# ---------------------------
class _SimpleRegistry:
    """Minimal registry that returns the run_dir path and records metadata.json"""
    def __init__(self, base: Path):
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)

    def register_run(self, payload: Dict) -> str:
        # payload expected to have 'run_id' and 'path'
        # In our usage, we'll just log and return run_id
        run_id = payload.get("run_id", None)
        logger.info(f"[Registry] register_run called for {run_id}")
        return run_id

class _SimpleRiskManager:
    """Rejects/adjusts convictions if volatility > limit (very simple)."""
    def __init__(self, max_leverage: float = 1.0, target_vol_limit: float = 0.12):
        self.max_leverage = max_leverage
        self.target_vol_limit = target_vol_limit

    def enforce_limits(self, conviction_series: pd.Series, price_series: pd.Series, volume_series: pd.Series):
        # Returns (adjusted_conv_series, leverage_series) same index
        # Simple: if recent vol > target, scale conviction down linearly
        ret = price_series.pct_change().dropna()
        vol = float(ret.rolling(min(21, max(2, len(ret)))).std().iloc[-1] * np.sqrt(252)) if len(ret) > 1 else MIN_VOL_FALLBACK
        factor = 1.0
        if vol > self.target_vol_limit:
            factor = max(0.0, 1.0 - (vol - self.target_vol_limit) / (self.target_vol_limit * 2))
        adjusted = conviction_series * factor
        # Adjusted result wrapper for simple compat
        class SimpleRes:
             def __init__(self, adj): 
                 self.adjusted_conviction = adj
                 self.estimated_leverage = float(adj.iloc[-1])
                 self.violations = []
        return SimpleRes(adjusted)

class _SimpleAlpha:
    """Simple composite alpha placeholder: momentum vs mean reversion mix."""
    def __init__(self):
        pass

    def compute(self, price_series: pd.Series) -> pd.Series:
        s = price_series.dropna()
        if len(s) < 50:
            return pd.Series([0.0], index=[s.index[-1]]) if len(s) else pd.Series([0.0])
        mom = float(s.iloc[-1] / s.iloc[-200] - 1) if len(s) > 200 else 0.0
        return pd.Series([1.0 if mom > 0 else 0.0], index=[s.index[-1]])

# ---------------------------
# Helpers for price history
# ---------------------------
def get_price_history(provider, ticker: str, end_dt: datetime, lookback: int = 500) -> pd.Series:
    # Try provider adapters
    try:
        if provider is not None:
            if hasattr(provider, "get_bars"):
                df = provider.get_bars(ticker, end=end_dt, lookback=lookback)
                if isinstance(df, pd.Series):
                    return df.squeeze()
                if "Close" in df.columns:
                    return df["Close"].squeeze()
            if hasattr(provider, "get_history"):
                df = provider.get_history(ticker, end=end_dt, lookback=lookback)
                if isinstance(df, pd.Series):
                    return df.squeeze()
                if "Close" in df.columns:
                    return df["Close"].squeeze()
    except Exception as e:
        logger.warning(f"Provider failed for {ticker}: {e}")

    # fallback yfinance
    try:
        end_str = end_dt.strftime("%Y-%m-%d")
        df = yf.download(ticker, end=end_str, period=f"{lookback}d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df["Close"].squeeze()
    except Exception as e:
        logger.warning(f"yfinance fallback failed for {ticker}: {e}")
        return pd.Series(dtype=float)

def is_first_trading_day_of_month(ts: pd.Timestamp) -> bool:
    # conservative approx: day <=3 considered first trading day window
    return ts.day <= 3

# ---------------------------
# Main orchestration
# ---------------------------
def main(config_override: Optional[Dict] = None):
    """
    Main entry point for institutional fund driver.
    """
    # Config handling
    global START_DATE, END_DATE, TICKERS, INITIAL_CAPITAL
    validation_mode = False
    
    if config_override:
        START_DATE = config_override.get("start_date", START_DATE)
        END_DATE = config_override.get("end_date", END_DATE)
        TICKERS = config_override.get("tickers", TICKERS)
        INITIAL_CAPITAL = config_override.get("initial_capital", INITIAL_CAPITAL)
        validation_mode = config_override.get("validation_mode", False)

    # Setup logging first (to temp location, will be moved to run_dir)
    global logger
    logger = logging.getLogger(__name__)
    
    # Create run directory via registry (immutable)
    run_record = None
    run_id = None
    run_dir = None
    
    # Instantiate registry early (needed before run creation)
    registry = None
    try:
        if BacktestRegistry is not None:
            registry = BacktestRegistry()
    except Exception:
        registry = None
    
    if registry is not None and hasattr(registry, 'create_run'):
        try:
            # Use new registry API
            # Include comprehensive config in run creation
            run_config = {
                "tickers": TICKERS,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "initial_capital": INITIAL_CAPITAL,
                "strategy_name": "mini-quant-fund-institutional",
            }
            if config_override:
                run_config.update(config_override)
                
            run_record = registry.create_run(run_config)
            run_id = run_record["run_id"]
            run_dir = Path(run_record["path"])
            print(f"Registry created run directory: {run_id}")  # Use print before logger configured
        except Exception as e:
            print(f"Registry create_run failed: {e}, falling back to manual")
            run_record = None
    
    # Fallback if registry not available
    if run_dir is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        run_dir = OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    
    # Now configure logging to actual run_dir
    log_file = run_dir / "execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger.info("\n" + "="*48)
    logger.info("  MINI QUANT FUND — INSTITUTIONAL DRIVER (MAIN)")
    logger.info("="*48 + "\n")
    logger.info("Initializing Mini Quant Fund...")
    logger.info(f"Run ID: {run_id}")

    # instantiate optional infra (provider, datastore)
    # Note: registry already instantiated early in main()
    provider = None
    datastore = None

    try:
        if YahooDataProvider is not None:
            provider = YahooDataProvider()
            logger.info("YahooDataProvider instantiated (Primary: FREE Only).")
        else:
             logger.error("YahooDataProvider not found! Critical failure.")
             return
    except Exception as e:
        logger.error(f"Failed to instantiate YahooDataProvider: {e}")
        return

    try:
        if DataStore is not None:
            datastore = DataStore()
            logger.info("DataStore instantiated.")
    except Exception as e:
        logger.warning(f"Failed to instantiate DataStore: {e}")
        datastore = None

    # Execution handler / Engine instantiation (defensive for different ctor names)
    handler = None
    try:
        if RealisticExecutionHandler is not None:
            handler = RealisticExecutionHandler(
                commission_pct=COMMISSION_PCT,
                max_participation_rate=MAX_PARTICIPATION,
                impact_coeff=IMPACT_COEFF,
                adv_lookback=ADV_LOOKBACK,
                vol_lookback=VOL_LOOKBACK,
                min_vol_fallback=MIN_VOL_FALLBACK,
            )
            logger.info("RealisticExecutionHandler instantiated.")
    except Exception as e:
        logger.error(f"Failed instantiate RealisticExecutionHandler: {e}")
        handler = None

    engine = None
    try:
        # try common constructor signatures
        if BacktestEngine is None:
            raise RuntimeError("BacktestEngine class not found.")
        try:
            engine = BacktestEngine(provider=provider, initial_capital=INITIAL_CAPITAL, execution_handler=handler)
        except TypeError:
            try:
                engine = BacktestEngine(data_provider=provider, initial_capital=INITIAL_CAPITAL, execution_handler=handler)
            except TypeError:
                engine = BacktestEngine(provider, INITIAL_CAPITAL, handler)
        logger.info("BacktestEngine instantiated.")
    except Exception as e:
        logger.error(f"Failed to instantiate BacktestEngine: {e}")
        return

    # add tickers if engine supports it
    try:
        if hasattr(engine, "add_tickers"):
            engine.add_tickers(TICKERS)
            logger.info(f"Added tickers {TICKERS} via add_tickers.")
        elif hasattr(engine, "set_universe"):
            engine.set_universe(TICKERS)
            logger.info(f"Set universe {TICKERS} via set_universe.")
    except Exception as e:
        logger.warning(f"Failed to add tickers to engine: {e}")

    # alpha & risk
    if CompositeAlpha is not None:
        try:
            trend_alpha = TrendAlpha(short=50, long=200)
            meanrev_alpha = MeanReversionAlpha(short=5, lookback_std=21)
            rsi_alpha = RSIAlpha(period=14)
            macd_alpha = MACDAlpha()
            bb_alpha = BollingerBandAlpha()
            
            # Combine them with weights
            # e.g., Trend 30%, MeanRev 20%, RSI 20%, MACD 15%, BB 15%
            composite_alpha = CompositeAlpha(
                alphas=[trend_alpha, meanrev_alpha, rsi_alpha, macd_alpha, bb_alpha],
                window=60
            )
            logger.info("CompositeAlpha instantiated with multiple alpha components.")
        except Exception as e:
            logger.warning(f"Failed to instantiate CompositeAlpha with full components, falling back to _SimpleAlpha: {e}")
            composite_alpha = _SimpleAlpha()
    else:
        logger.info("CompositeAlpha not available, falling back to _SimpleAlpha.")
        composite_alpha = _SimpleAlpha()

    if RiskManager is not None:
        try:
            risk_manager = RiskManager(max_leverage=1.0, target_vol_limit=0.12)
            logger.info("RiskManager instantiated.")
        except Exception as e:
            logger.warning(f"Failed to instantiate RiskManager, falling back to _SimpleRiskManager: {e}")
            risk_manager = _SimpleRiskManager(max_leverage=1.0, target_vol_limit=0.12)
    else:
        logger.info("RiskManager not available, falling back to _SimpleRiskManager.")
        risk_manager = _SimpleRiskManager(max_leverage=1.0, target_vol_limit=0.12)

    # ---------------------------
    # ML Initialization & Training
    # ---------------------------
    feature_engineer = None
    ml_model = None
    optimizer = None
    
    if FeatureEngineer is not None and MLAlpha is not None:
        logger.info("[System] Initializing Machine Learning Pipeline...")
        feature_engineer = FeatureEngineer()
        ml_model = MLAlpha(feature_engineer)
        
        # We need to fetch history for ALL tickers to train
        # For simplicity in this "mini" fund, we fetch 2 years history for training
        logger.info(f"[System] Pre-fetching training data for {TICKERS}...")
        training_assets = {}
        for tk in TICKERS:
            # We'll use provider directly
            try:
                # 2 years lookback for training relative to START_DATE
                # Fixed: ensure train_start is always before train_end
                train_end = START_DATE
                train_start_ts = pd.Timestamp(START_DATE) - pd.Timedelta(days=730)
                train_start = train_start_ts.strftime("%Y-%m-%d")
                
                logger.info(f"Fetching training data for {tk} ({train_start} to {train_end})")
                df = provider.fetch_ohlcv(tk, start_date=train_start, end_date=train_end)
                if not df.empty:
                    training_assets[tk] = df
            except Exception as e:
                logger.warning(f"Warning: Failed to fetch training data for {tk}: {e}")
        
        # Train model on concatenated history (Global Model) or Per-Asset?
        # ML4T usually suggests global or sector-specific. We'll do global for robustness.
        if training_assets:
            combined_history = pd.concat(training_assets.values())
            # Ensure index monotonic? Concat stack them.
            # Train!
            try:
                ml_model.train(combined_history)
                logger.info("ML model trained successfully.")
            except Exception as e:
                logger.error(f"ML Training Failed: {e}")
                ml_model = None
        else:
            logger.warning("No training data available. ML disabled.")
            ml_model = None
    else:
        logger.info("FeatureEngineer or MLAlpha not available. ML pipeline skipped.")
    
    if MeanVarianceOptimizer is not None:
        optimizer = MeanVarianceOptimizer()
        logger.info("MeanVarianceOptimizer instantiated.")
    else:
        logger.info("MeanVarianceOptimizer not available.")
        
    risk_model = None
    if StatisticalRiskModel is not None:
        logger.info("[System] Initializing Statistical Risk Model (PCA)...")
        try:
            risk_model = StatisticalRiskModel(n_components=3)
            logger.info("StatisticalRiskModel instantiated.")
        except Exception as e:
            logger.warning(f"Failed to instantiate StatisticalRiskModel: {e}")
    else:
        logger.info("StatisticalRiskModel not available.")

    kalman_strat = None
    if KalmanPairsTrader is not None:
        try:
            kalman_strat = KalmanPairsTrader()
            logger.info("KalmanPairsTrader (Stat Arb) instantiated.")
        except Exception as e:
            logger.warning(f"Failed to instantiate KalmanPairsTrader: {e}")

    # ---------------------------
    # Global Data Cache (Optimization)
    # ---------------------------
    # Fetch all data ONCE to avoid network latency in the loop
    logger.info(f"Pre-fetching full history for backtest (with warmup)...")
    _global_price_cache = {}
    
    warmup_start = (pd.Timestamp(START_DATE) - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    cache_end = datetime.today().strftime("%Y-%m-%d") if END_DATE is None else END_DATE
    
    for tk in TICKERS:
        try:
             # Fetch generous history
             df = provider.fetch_ohlcv(tk, start_date=warmup_start, end_date=cache_end)
             
             # Validate data
             if DataValidator:
                 validation = DataValidator.validate_ohlcv(df, tk)
                 if not validation["valid"]:
                     logger.warning(f"Data validation warnings for {tk}: {validation['issues']}")
                     df = DataValidator.clean_ohlcv(df)
                     logger.info(f"Data cleaned for {tk}")
             
             if not df.empty:
                 _global_price_cache[tk] = df
                 logger.info(f"Cached {len(df)} bars for {tk}")
        except Exception as e:
            logger.error(f"Failed to cache {tk}: {e}")

    # strategy function (monthly rebalance using convictions)
    # strategy_fn (Daily execution capabilty with Monthly Rebalance logic + Stat Arb)
    
    # Instantiate Allocator (Critical for signaling)
    allocator = None
    if InstitutionalAllocator and risk_manager:
        allocator = InstitutionalAllocator(risk_manager)
        logger.info("Institutional Allocator instantiated (Active).")
    else:
        logger.warning(f"Allocator NOT instantiated. Class: {InstitutionalAllocator}")
        
    def strategy_fn(timestamp: pd.Timestamp, prices: Dict[str, float], portfolio) -> List:
        
        # 0. Global Updates (Regime & Kalman State)
        spy_price = prices.get("SPY")
        qqq_price = prices.get("QQQ")
        
        # Update Risk Regime
        if "SPY" in _global_price_cache and risk_manager and hasattr(risk_manager, "update_regime"):
             full_df = _global_price_cache["SPY"]
             # Robust slice locally
             spy_hist = full_df[full_df.index <= timestamp]["Close"]
             risk_manager.update_regime(spy_hist)

        # 0b. Emergency Circuit Breaker Check (Daily)
        # If Risk Manager is Frozen or Market in Crisis -> LIQUIDATE IMMEDIATELY
        is_emergency = False
        emergency_reason = ""
        
        # Check State/Regime
        if risk_manager:
             if hasattr(risk_manager, "state") and risk_manager.state == RiskDecision.FREEZE:
                 is_emergency = True
                 emergency_reason = "Risk Manager FREEZE"
             elif hasattr(risk_manager, "regime") and risk_manager.regime == RiskRegime.BEAR_CRISIS:
                 is_emergency = True
                 emergency_reason = "Market Regime CRISIS"

        if is_emergency:
             # LIQUIDATE ALL POSITIONS
             orders = []
             # Access positions safely
             positions = getattr(portfolio, "positions", {})
             # If portfolio is a Portfolio object, positions might be a property returning a dict copy
             if isinstance(positions, dict):
                 for tk, qty in positions.items():
                     if abs(qty) > 0:
                         # Close it. 
                         orders.append(Order(tk, -qty, OrderType.MARKET, pd.Timestamp(timestamp)))
             
             if orders:
                 logger.warning(f"EMERGENCY LIQUIDATION TRIGGERED: {emergency_reason} - Closing {len(orders)} positions.")
                 return orders
             return [] # Already flat, stay flat

        # Update Kalman State (Daily)
        kalman_signal = False
        kalman_spread_val = 0.0
        kalman_z = 0.0
        
        if kalman_strat and spy_price and qqq_price:
             # SPY (X), QQQ (Y)
             # Update filter
             beta, error, z = kalman_strat.update(spy_price, qqq_price)
             kalman_z = z
             
             # Signal Logic: z > 2.0 or z < -2.0
             if abs(z) > 2.0:
                 kalman_signal = True

        # Check Rebalance Triggers
        is_rebalance_day = is_first_trading_day_of_month(pd.Timestamp(timestamp))
        
        # If neither monthly rebalance nor stat-arb signal, do nothing
        if not is_rebalance_day and not kalman_signal:
            return []
            
        # Kalman Overlay Logic (Tactical) - Generate direct orders if mid-month
        if not is_rebalance_day and kalman_signal:
             orders = []
             equity = getattr(portfolio, "total_equity", INITIAL_CAPITAL)
             alloc_amt = equity * 0.02 # 2% allocation to arb
             
             if kalman_z > 2.0: # Short Spread (Short QQQ, Long SPY)
                 if spy_price > 0 and qqq_price > 0:
                     orders.append(Order("SPY", alloc_amt / spy_price, OrderType.MARKET, pd.Timestamp(timestamp)))
                     orders.append(Order("QQQ", -alloc_amt / qqq_price, OrderType.MARKET, pd.Timestamp(timestamp)))
             elif kalman_z < -2.0: # Long Spread (Long QQQ, Short SPY)
                 if spy_price > 0 and qqq_price > 0:
                     orders.append(Order("QQQ", alloc_amt / qqq_price, OrderType.MARKET, pd.Timestamp(timestamp)))
                     orders.append(Order("SPY", -alloc_amt / spy_price, OrderType.MARKET, pd.Timestamp(timestamp)))
             return orders

        # 1. Feature Generation & Signal (Only run full alpha pipeline on rebalance days)
        signals = {}
        price_history = {}
        vol_history = {}
        
        # Use cached dataframe
        for tk, px in prices.items():
            if tk not in _global_price_cache: 
                continue
                
            full_df = _global_price_cache[tk]
            # Slice history up to timestamp (simulating point-in-time)
            history = full_df[full_df.index <= timestamp]
            if len(history) < 50:
                continue
            
            price_history[tk] = history["Close"]
            vol_history[tk] = history["Volume"]

            # 1.1 Compute Alpha Score
            raw_score = 0.5
            if composite_alpha:
                 try:
                     # compute() returns Series of 0..1 scores
                     alpha_s = composite_alpha.compute(history)
                     if not alpha_s.empty:
                         raw_score = float(alpha_s.iloc[-1])
                 except Exception as e:
                     pass
            else:
                 # Fallback: Simple Momentum
                 mom = (history["Close"].iloc[-1] / history["Close"].iloc[-20] - 1.0)
                 raw_score = 1.0 if mom > 0 else 0.0

            # 1.2 Apply ML Confidence (Gatekeeper)
            if ml_model and ml_model.is_trained:
                conf_score = ml_model.predict_conviction(history)
                
                # High Accuracy Gate: If confidence is too low, neutralize signal to avoid losses
                if conf_score < ML_CONFIDENCE_THRESHOLD:
                     logger.debug(f"Confidence {conf_score:.2f} < {ML_CONFIDENCE_THRESHOLD} for {tk}. Neutralizing.")
                     raw_score = 0.5
                else:
                     # Simpler: Just use ML as a scalar on deviation from neutral
                     deviation = raw_score - 0.5
                     dampened = deviation * conf_score 
                     raw_score = 0.5 + dampened

            signals[tk] = raw_score
            
        # 2. Allocation & Order Generation
        if allocator:
             allocation = allocator.allocate(signals, price_history, vol_history, portfolio, timestamp)
             # Log allocation for audit
             if allocation.orders:
                 logger.info(f"Rebalance Orders: {len(allocation.orders)} | Target Weights: {allocation.target_weights.keys()}")
             elif signals:
                 # Logic for why no orders?
                 logger.info(f"Allocator produced 0 orders from {len(signals)} signals.")
             return allocation.orders
        else:
             logger.warning("Allocator is None in strategy_fn")
        
        return []

    # helper to compute portfolio value (tries engine APIs)
    def total_portfolio_value(portfolio, current_prices: Dict[str, float] = None) -> float:
        # 1. Try manual calculation if prices provided (Most accurate during run)
        if current_prices is not None and portfolio is not None:
             val = getattr(portfolio, "cash", 0.0)
             for tk, qty in getattr(portfolio, "positions", {}).items():
                 px = current_prices.get(tk)
                 if px is not None and not np.isnan(px):
                     val += qty * px
             return float(val)

        try:
            if hasattr(portfolio, "market_value"):
                return float(portfolio.market_value())
            if hasattr(portfolio, "get_equity_value"):
                return float(portfolio.get_equity_value())
            if hasattr(engine, "get_results"):
                df = engine.get_results()
                if df is not None and not df.empty:
                    # last equity
                    s = df.reset_index().iloc[-1]
                    # try common column names
                    if "equity" in df.columns:
                        return float(df["equity"].iloc[-1])
                    if "nav" in df.columns:
                        return float(df["nav"].iloc[-1])
            # fallback
            return float(getattr(portfolio, "cash", INITIAL_CAPITAL))
        except Exception as e:
            logger.warning(f"Failed to compute total portfolio value, falling back to cash: {e}")
            return float(getattr(portfolio, "cash", INITIAL_CAPITAL))

    # Run the engine
    logger.info(f"Starting backtest from {START_DATE} for universe: {TICKERS}")
    try:
        # preferred engine.run signature: start_date, strategy_fn, tickers, end_date
        engine.run(start_date=START_DATE, strategy_fn=strategy_fn, tickers=TICKERS, end_date=END_DATE)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

    # ---------------------------
    # Allocator
    # ---------------------------
    allocator = None
    if InstitutionalAllocator and risk_manager:
        allocator = InstitutionalAllocator(risk_manager)
        logger.info("Institutional Allocator instantiated.")
    else:
        logger.warning(f"Allocator skipped (Available: {InstitutionalAllocator is not None}, RM: {risk_manager is not None})")

    # ---------------------------
    # Performance Analytics
    # ---------------------------
    logger.info("Generating performance analytics...")
    
    try:
        # Get results via standardized API
        try:
            results = engine.get_results()
            blotter = engine.get_blotter()
            trades_df = blotter.trades_df()
        except Exception:
            results = None
            trades_df = pd.DataFrame()
            
        if results is not None and not results.empty and PerformanceAnalyzer:
            equity_curve = results["equity"] if "equity" in results.columns else results.iloc[:, 0]
            
            analyzer = PerformanceAnalyzer(equity_curve, trades_df, risk_free_rate=0.02)
            report_str = analyzer.generate_report()
            logger.info("\n" + report_str)
            
            # Save report
            report_path = run_dir / "performance_report.txt"
            with open(report_path, "w") as f:
                f.write(report_str)
            logger.info(f"Performance report saved to {report_path}")
        else:
            logger.warning("PerformanceAnalyzer or results missing, skipping report.")
    except Exception as e:
        logger.warning(f"Performance analytics failed: {e}")

    # collect results
    try:
        results = engine.get_results()
    except Exception:
        results = None

    if results is None or results.empty:
        print("No results; exiting.")
        return

    # compute analytics (best-effort)
    try:
        equity_curve = results["equity"] if "equity" in results.columns else results.iloc[:, 0]
    except Exception:
        equity_curve = results.iloc[:, 0]

    def _safe_analytics(returns_series: pd.Series):
        try:
            ann_ret = annualized_return(returns_series)
            ann_vol = annualized_volatility(returns_series)
            sr = sharpe_ratio(returns_series)
            mdd = max_drawdown(equity_curve)
            return ann_ret, ann_vol, sr, mdd
        except Exception:
            # fallback simple analytics
            rr = returns_series.mean() * 252
            rv = returns_series.std() * np.sqrt(252)
            sr = rr / (rv + 1e-12)
            mdd = float(np.max(1 - equity_curve / equity_curve.cummax()))
            return rr, rv, sr, mdd

    returns = equity_curve.pct_change().fillna(0)
    ann_ret, ann_vol, sr, mdd = _safe_analytics(returns)

    print("\n--- BACKTEST METRICS ---")
    try:
        print(f"Final Equity: ${float(equity_curve.iloc[-1]):,.2f}")
    except Exception:
        print("Final Equity: (unable to render)")

    print(f"Annualized Return: {ann_ret:.4f}")
    print(f"Annualized Volatility: {ann_vol:.4f}")
    print(f"Sharpe Ratio: {sr:.4f}")
    print(f"Max Drawdown: {mdd:.4%}")

    # ---------------------------
    # Persist run artifacts (institutional-grade)
    # ---------------------------
    try:
        # Collect all data for registry
        blotter = engine.get_blotter()
        trades_df = pd.DataFrame()
        try:
            trades_df = blotter.trades_df()
            
            # REPORT TRADES (Explicit check)
            if not trades_df.empty:
                 print(f"TOTAL TRADES EXECUTED: {len(trades_df)}")
                 if 'pnl' in trades_df.columns:
                     print(f"Gross PnL: ${trades_df['pnl'].sum():,.2f}")
                 if 'commission' in trades_df.columns:
                     print(f"Total Commission: ${trades_df['commission'].sum():,.2f}")
            else:
                 print("TOTAL TRADES EXECUTED: 0")
                 
        except Exception:
            trades_df = pd.DataFrame()
            print("TOTAL TRADES EXECUTED: 0 (Error accessing blotter)")

        # Prepare equity DataFrame
        equity_df = results.reset_index() if hasattr(results, "reset_index") else pd.DataFrame(results)
        if "timestamp" not in equity_df.columns and len(equity_df.columns) > 0:
            equity_df = equity_df.rename(columns={equity_df.columns[0]: "timestamp"})

        # Config dict
        config = {
            "tickers": TICKERS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "initial_capital": INITIAL_CAPITAL,
            "commission_pct": COMMISSION_PCT,
            "impact_coeff": IMPACT_COEFF,
            "max_participation": MAX_PARTICIPATION,
            "strategy_name": "mini-quant-fund-institutional",
        }

        # Data manifest (best-effort)
        data_manifest = {"tickers": TICKERS, "data_source": "yahoo_finance"}
        try:
            if datastore is not None and hasattr(datastore, "get_manifest"):
                data_manifest.update(datastore.get_manifest())
            elif provider is not None and hasattr(provider, "get_manifest"):
                data_manifest.update(provider.get_manifest())
        except Exception:
            pass

        # Extra metadata
        extra_meta = {
            "initial_capital": INITIAL_CAPITAL,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sr,
            "max_drawdown": mdd,
        }

        # Use enhanced registry if available
        if registry is not None and hasattr(registry, 'save_artifacts'):
            try:
                registry.save_artifacts(
                    run_id=run_id,
                    config=config,
                    equity_df=equity_df,
                    trades_df=trades_df,
                    data_manifest=data_manifest,
                    extra_meta=extra_meta,
                )
                logger.info(f"All artifacts saved via registry for run: {run_id}")
                logger.info(f"Run directory: {run_dir}")
                
                # List saved files
                saved_files = list(run_dir.glob("*"))
                logger.info(f"Saved {len(saved_files)} artifacts: {[f.name for f in saved_files]}")
            except Exception as e:
                logger.error(f"Registry save_artifacts failed: {e}")
                # Fall through to manual save below
                raise
        else:
            # Fallback manual save (old method)
            logger.warning("Registry save_artifacts not available, using fallback")
            
            trades_path = run_dir / "trades.csv"
            if trades_df is not None and not trades_df.empty:
                _atomic_write_df(trades_path, trades_df)
            else:
                pd.DataFrame().to_csv(trades_path, index=False)

            equity_path = run_dir / "equity.csv"
            _atomic_write_df(equity_path, equity_df)

            config_path = run_dir / "config.json"
            _atomic_write_text(config_path, json.dumps(config, indent=2))

            data_manifest_path = run_dir / "data_manifest.json"
            _atomic_write_text(data_manifest_path, json.dumps(data_manifest, indent=2))

            req_path = run_dir / "requirements.txt"
            try:
                reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode()
                _atomic_write_text(req_path, reqs)
            except Exception:
                _atomic_write_text(req_path, "")

            # git hash
            try:
                git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                git_hash = None

            # compute checksums
            artifacts = {}
            for p in [trades_path, equity_path, config_path, data_manifest_path, req_path]:
                if p.exists():
                    try:
                        artifacts[p.name] = {"sha256": _sha256_of_file(p), "size": p.stat().st_size}
                    except Exception:
                        artifacts[p.name] = {"sha256": None, "size": None}

            meta = {
                "run_id": run_id,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "git_hash": git_hash,
                "python_version": sys.version,
                "artifacts": artifacts,
            }
            meta.update(extra_meta)
            
            meta_path = run_dir / "meta.json"
            _atomic_write_text(meta_path, json.dumps(meta, indent=2))
            
            logger.info(f"Manual save completed to: {run_dir}")

    except Exception as e:
        logger.error(f"ERROR: Failed to persist run artifacts: {e}")
        raise

    print("\nDone.")

# ---------------------------
# Institutional Safety Helpers
# ---------------------------
def _validate_market_data_freshness(data_map: Dict[str, pd.DataFrame], tickers: List[str]):
    """FAIL FAST: Abort if data is stale or missing."""
    now = datetime.now()
    for tk in tickers:
        if tk not in data_map or data_map[tk].empty:
            raise RuntimeError(f"INSTITUTIONAL ABORT: Missing data for {tk}")
        
        last_dt = data_map[tk].index[-1]
        # In free mode, Yahoo daily bars might be 1-2 days old depending on timezone/market close
        # We allow up to 3 days for weekend/holiday fallback
        if (now - last_dt).days > 3:
            raise RuntimeError(f"INSTITUTIONAL ABORT: Data for {tk} is stale ({last_dt})")
    return True

def _health_check_circuit_breaker(local_equity: float, broker_equity: float):
    """KILL SWITCH: Abort if local book differs significantly from broker."""
    drift = abs(local_equity - broker_equity) / (broker_equity + 1e-9)
    if drift > EQUITY_DRIFT_TOLERANCE:
        raise RuntimeError(f"KILL SWITCH: Equity mismatch detected! Local: ${local_equity:,.2f} | Broker: ${broker_equity:,.2f} | Drift: {drift:.1%}")
    logger.info(f"[Health] Equity alignment OK (Drift: {drift:.2%})")
def run_paper_mode(config: Dict[str, Any]):
    """
    INSTITUTIONAL PAPER TRADING (Free-Only Mode)
    Yahoo -> Signals -> Risk -> Orders -> Alpaca Paper.
    """
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger.info("="*48)
    logger.info("  MINI QUANT FUND — INSTITUTIONAL PAPER MODE")
    logger.info("  (Yahoo Data + Alpaca Execution)")
    logger.info("="*48)
    
    # 1. Initialization & Free-Only Connectivity
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    if not API_KEY or not SECRET_KEY:
        logger.error("Missing Alpaca keys. Aborting.")
        return

    # Alpaca is strictly for EXECUTION
    execution_broker = AlpacaExecutionHandler(API_KEY, SECRET_KEY, BASE_URL)
    # Yahoo is strictly for DATA
    data_provider = YahooDataProvider() 
    
    try:
        account = execution_broker.get_account()
        broker_equity = float(account.get("equity", 0.0))
        broker_cash = float(account.get("cash", 0.0))
        logger.info(f"Connected to Broker. Equity: ${broker_equity:,.2f}")
    except Exception as e:
        logger.error(f"Broker connection failed: {e}")
        return

    # 2. Institutional Signal Pipeline (Offline)
    logger.info("[System] Loading Institutional Pipeline...")
    risk_manager = RiskManager(max_leverage=1.0, target_vol_limit=0.12)
    allocator = InstitutionalAllocator(risk_manager)
    
    # ML Governance: Offline Training only
    ml_model = None
    if FeatureEngineer and MLAlpha:
        logger.info("[System] Initializing ML Alpha (Frozen Mode)...")
        fe = FeatureEngineer()
        ml_model = MLAlpha(fe)
        # In a real institutional setup, weights would be loaded from disk. 
        # Here we train once on Yahoo data to "freeze" the model before the session.
        logger.info("[Download] Fetching Yahoo training data (2Y)...")
        train_start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        train_data = {}
        for tk in TICKERS:
             df = data_provider.fetch_ohlcv(tk, start_date=train_start)
             if not df.empty: train_data[tk] = df
        
        if train_data:
             ml_model.train(pd.concat(train_data.values()))
             logger.info("ML Model Trained & Frozen.")

    # 3. Market Awareness (Yahoo only)
    logger.info("[System] Syncing Market Data from Yahoo...")
    lookback_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    current_market_data = {}
    for tk in TICKERS:
        df = data_provider.fetch_ohlcv(tk, start_date=lookback_start)
        if not df.empty:
            current_market_data[tk] = df
            
    # FAIL FAST: Abort if data gaps detected
    _validate_market_data_freshness(current_market_data, TICKERS)
    
    # 4. Kill Switch: Reconciliation
    # Local portfolio proxy (Book)
    from collections import namedtuple
    Book = namedtuple("Book", ["cash", "positions", "total_equity"])
    
    broker_pos_raw = execution_broker.get_positions()
    broker_positions = {p['symbol']: float(p['qty']) for p in broker_pos_raw}
    
    # Value local book using Yahoo prices
    local_val = broker_cash
    for tk, qty in broker_positions.items():
        if tk in current_market_data:
            local_val += qty * current_market_data[tk]["Close"].iloc[-1]
    
    _health_check_circuit_breaker(local_val, broker_equity)

    # 5. Signal Generation & Risk Filter
    logger.info("[System] Generating Institutional Signals...")
    signals = {}
    price_histories = {}
    vol_histories = {}
    
    # Setup Alpha Components (Institutional standard)
    c_alpha = CompositeAlpha(alphas=[TrendAlpha(), MeanReversionAlpha()], window=60)
    
    for tk in TICKERS:
        hist = current_market_data[tk]
        price_histories[tk] = hist["Close"]
        vol_histories[tk] = hist["Volume"]
        
        # Raw Signal
        score = float(c_alpha.compute(hist).iloc[-1])
        
        # ML Accuracy Filter (High Accuracy Gate)
        if ml_model and ml_model.is_trained:
            conf = ml_model.predict_conviction(hist)
            if conf < ML_CONFIDENCE_THRESHOLD:
                score = 0.5 # Neutralize
            else:
                score = 0.5 + (score - 0.5) * conf
        
        signals[tk] = score

    # 6. Allocation & Realistic Execution Simulation
    # Before sending to Alpaca, we simulate against Yahoo bars to ensure 
    # the orders are realistic (volume/participation).
    logger.info("[System] Calculating Optimal Allocation...")
    p_book = Book(cash=broker_cash, positions=broker_positions, total_equity=broker_equity)
    allocation = allocator.allocate(signals, price_histories, vol_histories, p_book, pd.Timestamp(datetime.now()))
    
    if not allocation.orders:
        logger.info("Rebalance Cycle: NO TRADE REQUIRED (Confidence/Risk conditions not met).")
        return

    # Enforce Execution Realism: Volume/Participation Check
    verified_orders = []
    simulation_handler = RealisticExecutionHandler() # For realism check
    
    for order in allocation.orders:
        tk = order.ticker
        bar_df = current_market_data[tk]
        # Build institutional bar for simulation
        from backtest.execution import BarData
        bar = BarData(
            open=bar_df["Open"].iloc[-1],
            high=bar_df["High"].iloc[-1],
            low=bar_df["Low"].iloc[-1],
            close=bar_df["Close"].iloc[-1],
            volume=bar_df["Volume"].iloc[-1],
            timestamp=bar_df.index[-1],
            ticker=tk
        )
        
        # Check volume constraints
        if bar.volume <= 0:
            logger.warning(f"SKIPPING {tk}: Zero volume bar detected on Yahoo.")
            continue
            
        # Is the order size > max participation? 
        max_q = bar.volume * simulation_handler.max_participation_rate
        if abs(order.quantity) > max_q:
            old_q = order.quantity
            order.quantity = np.sign(order.quantity) * max_q
            logger.info(f"[Realism] Capping {tk} order: {old_q:.1f} -> {order.quantity:.1f} (Participation Limit)")
        
        if abs(order.quantity) >= 1.0: # Ignore tiny dust orders
            verified_orders.append(order)

    # 7. Final Execution (Alpaca)
    if verified_orders:
        logger.info(f"[EXEC] Submitting {len(verified_orders)} verified orders to Alpaca Paper...")
        # Sell first to free cash
        execution_queue = sorted(verified_orders, key=lambda x: x.quantity)
        results = execution_broker.submit_orders(execution_queue)
        for res in results:
            if res.get("status") == "failed":
                logger.error(f"ORDER FAILED: {res.get('error')}")
            else:
                logger.info(f"ORDER OK: {res.get('order', {}).get('symbol')}")
    else:
        logger.info("Rebalance Cycle: Orders filtered by Realism/Liquidity constraints.")

    logger.info("[Complete] Institutional Paper session finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mini Quant Fund Driver")
    parser.add_argument("--mode", default="backtest", choices=["backtest", "paper"], help="Run mode")
    parser.add_argument("--start_date", help="Start date YYYY-MM-DD")
    parser.add_argument("--end_date", help="End date YYYY-MM-DD")
    parser.add_argument("--tickers", help="Comma-sep tickers")
    parser.add_argument("--initial_capital", type=float, help="Initial capital")
    parser.add_argument("--validation", action="store_true", help="Run in validation mode (bypass blocking risk limits)")
    
    args = parser.parse_args()
    
    config = {}
    if args.start_date: config["start_date"] = args.start_date
    if args.end_date: config["end_date"] = args.end_date
    if args.tickers: config["tickers"] = args.tickers.split(",")
    if args.initial_capital: config["initial_capital"] = args.initial_capital
    if args.validation: config["validation_mode"] = True

    if args.mode == "paper":
        run_paper_mode(config)
    else:
        main(config)
