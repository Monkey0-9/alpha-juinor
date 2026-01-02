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
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import logging

import numpy as np
import pandas as pd

# ---------------------------
# Project imports (best-effort)
# ---------------------------
# Try import user modules; if missing, provide light fallbacks
try:
    from data.provider import YahooDataProvider  # optional
except Exception:
    YahooDataProvider = None

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
    from risk.engine import RiskManager
except Exception:
    RiskManager = None
    
try:
    from portfolio.optimizer import MeanVarianceOptimizer
except Exception:
    MeanVarianceOptimizer = None

try:
    from risk.factor_model import StatisticalRiskModel
except Exception:
    StatisticalRiskModel = None

try:
    from analytics.metrics import PerformanceAnalyzer
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

    def enforce_limits(self, conviction_series: pd.Series, price_series: pd.Series):
        # Returns (adjusted_conv_series, leverage_series) same index
        # Simple: if recent vol > target, scale conviction down linearly
        ret = price_series.pct_change().dropna()
        vol = float(ret.rolling(min(21, max(2, len(ret)))).std().iloc[-1] * np.sqrt(252)) if len(ret) > 1 else MIN_VOL_FALLBACK
        factor = 1.0
        if vol > self.target_vol_limit:
            factor = max(0.0, 1.0 - (vol - self.target_vol_limit) / (self.target_vol_limit * 2))
        adjusted = conviction_series * factor
        leverage = pd.Series([self.max_leverage], index=conviction_series.index)
        return adjusted, leverage

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
def main():
    """
    Main entry point for institutional fund driver.
    """
    # Create run directory
    run_id = f"run_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = run_dir / "execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # Use sys.stdout for console output
        ]
    )
    global logger # Make logger global so it can be used in _SimpleRegistry and get_price_history
    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*48)
    logger.info("  MINI QUANT FUND — INSTITUTIONAL DRIVER (MAIN)")
    logger.info("="*48 + "\n")
    logger.info("Initializing Mini Quant Fund...")
    logger.info(f"Run ID: {run_id}")

    # instantiate optional infra
    provider = None
    datastore = None
    registry = None

    try:
        if YahooDataProvider is not None:
            provider = YahooDataProvider()
            logger.info("YahooDataProvider instantiated.")
    except Exception as e:
        logger.warning(f"Failed to instantiate YahooDataProvider: {e}")
        provider = None

    try:
        if DataStore is not None:
            datastore = DataStore()
            logger.info("DataStore instantiated.")
    except Exception as e:
        logger.warning(f"Failed to instantiate DataStore: {e}")
        datastore = None

    try:
        if BacktestRegistry is not None:
            registry = BacktestRegistry()
            logger.info("BacktestRegistry instantiated.")
    except Exception as e:
        logger.warning(f"Failed to instantiate BacktestRegistry, falling back to _SimpleRegistry: {e}")
        # fallback local registry
        registry = _SimpleRegistry(RUN_OUTPUT_ROOT)

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
                weights=[0.30, 0.20, 0.20, 0.15, 0.15]
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
            # We use a helper to get dataframe
            # NOTE: get_price_history returns Series, we need DataFrame for ML
            # We'll use provider directly
            try:
                # 2 years lookback approx
                df = provider.fetch_ohlcv(tk, start_date="2020-01-01", end_date=START_DATE)
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
    def strategy_fn(timestamp: pd.Timestamp, prices: Dict[str, float], portfolio) -> List:
        # Rebalance only on first trading day approx
        if not is_first_trading_day_of_month(pd.Timestamp(timestamp)):
            return []

        # 1. Feature Generation & Signal
        expected_returns = {}
        valid_tickers = []
        
        # Use cached dataframe
        for tk, px in prices.items():
            if tk not in _global_price_cache: 
                continue
                
            full_df = _global_price_cache[tk]
            
            # Slice history up to timestamp (simulating point-in-time)
            # Use searchsorted/slicing for speed or simple boolean indexing
            # boolean indexing is safe
            
            # To avoid "lookahead" in the features, we must strictly use data <= timestamp
            # slice logic:
            history = full_df[full_df.index <= timestamp]
            
            if len(history) < 50:
                continue

            # Use ML if available
            if ml_model and ml_model.is_trained:
                # Predict returns
                score = ml_model.predict_conviction(history)
                ret_proxy = (score - 0.5) * 0.10 
            else:
                # Fallback to Simple Momentum
                ret_proxy = (history["Close"].iloc[-1] / history["Close"].iloc[-20] - 1.0)
            
            expected_returns[tk] = ret_proxy
            valid_tickers.append(tk)

        # 2. Estimate Covariance
        price_series_list = []
        for tk in valid_tickers:
            full_df = _global_price_cache[tk]
            history = full_df[full_df.index <= timestamp]
            if not history.empty:
                s = history["Close"].pct_change()
                s.name = tk
                price_series_list.append(s)
        
        if price_series_list:
            returns_df = pd.concat(price_series_list, axis=1).iloc[-252:].fillna(0) # Use 1y for risk model
            
            if risk_model:
                cov_matrix = risk_model.compute_covariance(returns_df)
            else:
                cov_matrix = returns_df.cov() * 252 # Fallback simple annualized
                
            er_series = pd.Series(expected_returns)
        else:
            return []

        # 3. Optimize
        weights = {}
        if optimizer and len(valid_tickers) > 0:
            weights = optimizer.optimize(er_series, cov_matrix)
        else:
            n = len(valid_tickers)
            if n > 0:
                weights = {t: 1.0/n for t in valid_tickers}

        # 4. Generate Orders
        total_equity = None
        try:
            total_equity = total_portfolio_value(portfolio)
        except Exception:
            # fallback
            total_equity = getattr(portfolio, "cash", INITIAL_CAPITAL)

        orders = []
        for tk, w in weights.items():
            target_value = total_equity * w
            current_qty = getattr(portfolio, "positions", {}).get(tk, 0.0) if portfolio is not None else 0.0
            current_price = prices.get(tk, None)
            
            if current_price is None or current_price <= 0:
                continue
                
            current_val = current_qty * current_price
            diff_val = target_value - current_val
            
            if abs(diff_val) < total_equity * MIN_TRADE_PCT:
                continue
                
            diff_qty = diff_val / current_price
            
            try:
                order = Order(ticker=tk, quantity=diff_qty, order_type=OrderType.MARKET, timestamp=pd.Timestamp(timestamp))
                orders.append(order)
            except Exception as e:
                logger.warning(f"Failed to create order for {tk}: {e}")
                continue
                
        return orders

    # helper to compute portfolio value (tries engine APIs)
    def total_portfolio_value(portfolio) -> float:
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
    # Performance Analytics
    # ---------------------------
    logger.info("Generating performance analytics...")
    
    try:
        # Get equity curve
        equity_df = engine.equity_df() if hasattr(engine, 'equity_df') else None
        trades_df = engine.trades_df() if hasattr(engine, 'trades_df') else None
        
        if equity_df is not None and not equity_df.empty and PerformanceAnalyzer:
            # Assume equity_df has a 'total_value' or 'equity' column
            if 'total_value' in equity_df.columns:
                equity_series = equity_df['total_value']
            elif 'equity' in equity_df.columns:
                equity_series = equity_df['equity']
            else:
                equity_series = equity_df.iloc[:, 0]
            
            analyzer = PerformanceAnalyzer(equity_series, trades_df)
            analyzer.print_summary()
            
            # Save report
            report_path = run_dir / "performance_summary.json"
            analyzer.save_report(str(report_path))
            logger.info(f"Performance report saved to {report_path}")
        elif equity_df is None or equity_df.empty:
            logger.warning("Equity data is empty or not available, skipping PerformanceAnalyzer.")
        else:
            logger.warning("PerformanceAnalyzer not available, skipping performance report generation.")
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
        run_dir = None
        # prefer registry to supply run_dir if it does
        if BacktestRegistry is not None and isinstance(registry, BacktestRegistry):
            # attempt to let registry create a run and give us a path
            try:
                # Many registry APIs accept a config dict and return run record
                rec = registry.create_run({"start_date": START_DATE, "tickers": TICKERS})
                if isinstance(rec, dict) and "path" in rec:
                    run_dir = Path(rec["path"])
                    run_dir.mkdir(parents=True, exist_ok=True)
                elif isinstance(rec, str):
                    run_dir = Path(rec)
                    run_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                run_dir = None

        if run_dir is None:
            run_dir = _safe_run_dir(RUN_OUTPUT_ROOT, "run")

        # trades
        blotter = engine.get_blotter()
        trades_df = pd.DataFrame()
        try:
            trades_df = blotter.trades_df()
        except Exception:
            trades_df = pd.DataFrame()

        trades_path = run_dir / "trades.csv"
        trades_saved = False
        if trades_df is not None and not trades_df.empty:
            _atomic_write_df(trades_path, trades_df)
            trades_saved = True
            try:
                os.chmod(trades_path, stat.S_IRUSR | stat.S_IWUSR)
            except Exception:
                pass

        # equity
        equity_path = run_dir / "equity.csv"
        equity_saved = False
        try:
            df_equity = results.reset_index() if hasattr(results, "reset_index") else pd.DataFrame(results)
            # rename first column to timestamp if unnamed
            if "timestamp" not in df_equity.columns:
                # Make sure index name
                df_equity = df_equity.rename(columns={df_equity.columns[0]: "timestamp"})
            _atomic_write_df(equity_path, df_equity)
            equity_saved = True
            try:
                os.chmod(equity_path, stat.S_IRUSR | stat.S_IWUSR)
            except Exception:
                pass
        except Exception:
            equity_saved = False

        # config.json
        config = {
            "tickers": TICKERS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "initial_capital": INITIAL_CAPITAL,
            "commission_pct": COMMISSION_PCT,
            "impact_coeff": IMPACT_COEFF,
            "max_participation": MAX_PARTICIPATION,
        }
        config_path = run_dir / "config.json"
        _atomic_write_text(config_path, json.dumps(config, indent=2))
        try:
            os.chmod(config_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

        # data_manifest (best-effort)
        data_manifest = {}
        try:
            if datastore is not None and hasattr(datastore, "get_manifest"):
                data_manifest = datastore.get_manifest()
            elif provider is not None and hasattr(provider, "get_manifest"):
                data_manifest = provider.get_manifest()
        except Exception:
            data_manifest = {}
        data_manifest_path = run_dir / "data_manifest.json"
        _atomic_write_text(data_manifest_path, json.dumps(data_manifest or {}, indent=2))

        # requirements
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

        # compute artifacts checksums
        artifacts = {}
        for p in [trades_path if trades_saved else None,
                  equity_path if equity_saved else None,
                  config_path,
                  data_manifest_path,
                  req_path]:
            if p is None:
                continue
            try:
                artifacts[p.name] = {"sha256": _sha256_of_file(p), "size": p.stat().st_size}
            except Exception:
                artifacts[p.name] = {"sha256": None, "size": None}

        meta = {
            "run_id": run_dir.name,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "git_hash": git_hash,
            "initial_capital": INITIAL_CAPITAL,
            "tickers": TICKERS,
            "artifacts": artifacts,
        }
        meta_path = run_dir / "meta.json"
        _atomic_write_text(meta_path, json.dumps(meta, indent=2))

        # registry.register_run best-effort
        try:
            if registry is not None and hasattr(registry, "register_run"):
                registry.register_run({
                    "run_id": run_dir.name,
                    "path": str(run_dir),
                    "meta": meta,
                })
        except Exception as e:
            print("Warning: registry.register_run failed:", e)

        print(f"\nRun artifacts written to: {run_dir}")
        print(f"trades_saved={trades_saved}, equity_saved={equity_saved}")
    except Exception as e:
        print("ERROR: Failed to persist run artifacts:", e)
        raise

    print("\nDone.")

if __name__ == "__main__":
    main()
