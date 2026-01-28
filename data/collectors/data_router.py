# Premium Institutional Data Providers
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data.collectors.alpaca_collector import AlpacaDataProvider
from data.providers.binance import BinanceDataProvider
from data.providers.bloomberg import BloombergDataProvider
from data.providers.coingecko import CoinGeckoDataProvider
from data.providers.finnhub import FinnhubDataProvider
from data.providers.news_provider import NewsDataProvider
from data.providers.polygon import PolygonDataProvider
from data.providers.reuters import ReutersDataProvider

# Alternative Data Providers
from data.providers.sentiment_provider import SentimentDataProvider
from data.providers.twelvedata import TwelveDataDataProvider
from data.providers.yahoo import YahooDataProvider
from database.manager import DatabaseManager
from utils.timeutils import ensure_business_days
from utils.timezone import normalize_index_utc

logger = logging.getLogger(__name__)

# =============================================================================
# INSTITUTIONAL PROVIDER GOVERNANCE MATRIX (DELEGATED)
# =============================================================================
# Now delegated to data.governance.provider_router

from data.governance.provider_router import (
    select_provider as governance_select_provider,
    mark_provider_unavailable,
    PROVIDER_ALPACA, PROVIDER_YAHOO
)
from data.intelligence.confidence_manager import ConfidenceManager

# PHASE 0: CRITICAL GUARDS (Must be enforced)
# Maximum history days allowed in LIVE trading loop
# Multi-year history must NEVER be fetched in live loop
MAX_LIVE_HISTORY_DAYS = 5  # Maximum 5 days of history in live mode

# Minimum data quality threshold
MIN_DATA_QUALITY_THRESHOLD = 0.6


class DataRouter:
    """
    Intelligent data router that selects the best data provider based on ticker type,
    availability, and institutional requirements.

    CRITICAL: Enforces Phase 0 guards:
    - No multi-year history in live loop
    - No retry on entitlement failures (400/403)
    - No symbol-level trading without portfolio competition
    """

    def __init__(self, max_workers: int = 10, enable_cache: bool = True):
        # Initialize providers
        self.providers = {
            "yahoo": YahooDataProvider(enable_cache=enable_cache),
            "binance": BinanceDataProvider(),
            "coingecko": CoinGeckoDataProvider(),
            "bloomberg": BloombergDataProvider(),
            "reuters": ReutersDataProvider(),
            "sentiment": SentimentDataProvider(),
            "news": NewsDataProvider(),
            "polygon": PolygonDataProvider(enable_cache=enable_cache),
            "finnhub": FinnhubDataProvider(),
            "twelvedata": TwelveDataDataProvider(),
            "alpaca": AlpacaDataProvider(),
        }

        # Institutional Provider Capabilities Matrix (Mandated Spec)
        self.PROVIDER_CAPABILITIES = {
            "alpaca": {
                "stocks": True,
                "fx": False,
                "crypto": True,
                "commodities": False,
                "max_history_days": 730,
                "requires_entitlement": True,
            },
            "yahoo": {
                "stocks": True,
                "fx": True,
                "crypto": True,
                "commodities": True,
                "max_history_days": 5000,
                "requires_entitlement": False,
            },
            "polygon": {
                "stocks": True,
                "fx": True,
                "crypto": True,
                "commodities": False,
                "max_history_days": 5000,
                "requires_entitlement": True,
            },
            "binance": {
                "stocks": False,
                "fx": False,
                "crypto": True,
                "commodities": False,
                "max_history_days": 1000,
                "requires_entitlement": False,
            },
            "fred": {
                "stocks": False,
                "fx": False,
                "crypto": False,
                "commodities": False,
                "max_history_days": None,
                "requires_entitlement": False,
                "macro_data": True,
            },
        }

        # Cache for unavailable providers (Circuit Breaker)
        self._unavailable_cache = {}

        # High-Speed Infrastructure
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Confidence & Quality Management (High-Priority Integration)
        from data.intelligence.confidence_manager import ConfidenceManager
        self.confidence_mgr = ConfidenceManager()
        logger.info("[DATA_ROUTER] ConfidenceManager initialized")

    def select_provider(
        self,
        symbol: str,
        history_days: int = 0,
        purpose: str = "history",
        entitled_providers: List[str] = None,
    ) -> str:
        """
        Institutional Entitlement-Aware Provider Routing.

        Delegates to data.governance.provider_router.
        """
        # Data Separation Check
        if purpose == "execution":
             # Execution must use Alpaca (or config execution provider)
             return PROVIDER_ALPACA

        # Use Governance Router
        selected = governance_select_provider(symbol, history_days)

        if not selected:
             logger.warning(
                f"[DATA_ROUTER] NO VALID PROVIDER for {symbol}. "
                f"History: {history_days}D | Purpose: {purpose}"
            )
             return "NO_VALID_PROVIDER"

        return selected

    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and compute quality score.

        Returns dict with 'score' (0-1) and 'reason_codes'.
        """
        if df is None or df.empty:
            return {"score": 0.0, "reason_codes": ["EMPTY_DATASET"]}

        reason_codes = []
        # Validate data quality using Institutional Formula
        # score = 1.0 - (missing_dates_pct * 0.3 + duplicate_pct * 0.2 + zero_negative_flag * 0.2 + extreme_spike_flag * 0.3)

        missing_dates_pct = df.isnull().sum().sum() / max(
            1, (len(df) * len(df.columns))
        )
        duplicate_pct = df.index.duplicated().sum() / max(1, len(df))

        zero_negative_flag = (
            1.0 if (df[["Open", "High", "Low", "Close"]] <= 0).any().any() else 0.0
        )

        extreme_spike_flag = 0.0
        if "Volume" in df.columns:
            vol_mean = df["Volume"].mean()
            vol_std = df["Volume"].std()
            if vol_std > 0:
                if (df["Volume"] > vol_mean + 6 * vol_std).any():
                    extreme_spike_flag = 1.0

        penalty = (
            (missing_dates_pct * 0.3)
            + (duplicate_pct * 0.2)
            + (zero_negative_flag * 0.2)
            + (extreme_spike_flag * 0.05)
        )
        score = max(0.0, 1.0 - penalty)

        if score < 1.0:
            if missing_dates_pct > 0:
                reason_codes.append(f"MISSING:{missing_dates_pct:.2%}")
            if duplicate_pct > 0:
                reason_codes.append(f"DUPS:{duplicate_pct:.2%}")
            if zero_negative_flag > 0:
                reason_codes.append("BAD_PRICES")
            if extreme_spike_flag > 0:
                reason_codes.append("VOL_SPIKE")

        return {"score": score, "reason_codes": reason_codes}

    def _check_entitlement_failure(self, error_msg: str) -> bool:
        """Check if error is an entitlement failure (400/403)."""
        return any(
            code in str(error_msg)
            for code in ["400", "403", "401", "Forbidden", "Unauthorized"]
        )

    def _get_best_provider(self, ticker: str, history_days: int = 0):
        """Get the best available provider for a ticker, checking institutional routing."""
        provider_name = self.select_provider(ticker, history_days)
        if provider_name == "NO_VALID_PROVIDER":
            return None
        return self.providers.get(provider_name)

    @staticmethod
    def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
        """Helper to ensure dataframe index is timezone-aware UTC and has frequency (B-days)."""
        return ensure_business_days(df)

    def get_price_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str = None,
        allow_long_history: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch price history with PHASE 0 guards.

        CRITICAL: This function enforces:
        1. No multi-year history in live loop
        2. No retry on entitlement failures (400/403)
        3. Data quality validation

        Args:
            ticker: Symbol to fetch
            start_date: Start date
            end_date: End date (defaults to now)
            allow_long_history: If True, allows long history (for batch ingestion only)

        Returns:
            DataFrame with price data (empty if validation fails)
        """
        # Calculate history days
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        history_days = (ed - sd).days

        # PHASE 0 GUARD: No multi-year history in live loop
        if not allow_long_history and history_days > MAX_LIVE_HISTORY_DAYS:
            logger.error(
                f"[DATA_ROUTER] GUARD VIOLATION: Requested {history_days} days for {ticker}. "
                f"Live loop max is {MAX_LIVE_HISTORY_DAYS} days. "
                f"Use allow_long_history=True for batch ingestion only."
            )
            return pd.DataFrame()

        # Select provider
        provider = self._get_best_provider(ticker, history_days=history_days)
        if not provider:
            return pd.DataFrame()

        def _validate_schema(df):
            if df is None or df.empty:
                return False
            required = ["Close"]
            return all(col in df.columns for col in required)

        # Retry logic for transient errors
        max_retries = 3
        backoff_factor = 2.0
        initial_delay = 1.0

        for attempt in range(max_retries):
            try:
                df = provider.fetch_ohlcv(ticker, start_date, end_date)
                if _validate_schema(df):
                    df = self._ensure_utc_index(df)
                    logger.debug(
                        f"Fast-path: {ticker} from {provider.__class__.__name__} (attempt {attempt+1})"
                    )

                    # Validate data quality
                    quality = self._validate_data_quality(df)
                    if quality["score"] < MIN_DATA_QUALITY_THRESHOLD:
                        logger.warning(
                            f"[DATA_ROUTER] Low quality data for {ticker}: "
                            f"score={quality['score']:.2f}, reasons={quality['reason_codes']}"
                        )

                    return df
                else:
                    logger.warning(
                        f"[DATA_ROUTER] Empty or invalid schema for {ticker} on attempt {attempt+1}"
                    )

            except Exception as e:
                error_msg = str(e)

                # PHASE 0 GUARD: No retry on entitlement failures
                if self._check_entitlement_failure(error_msg):
                    p_name = next(
                        (k for k, v in self.providers.items() if v == provider),
                        "unknown",
                    )
                    logger.error(
                        f"[DATA_ROUTER] ENTITLEMENT FAILURE for {ticker} via {p_name}: {error_msg}. "
                        f"NOT RETRYING. Symbol rejected for this cycle."
                    )
                    if p_name != "unknown":
                        self._unavailable_cache[p_name] = True
                    return pd.DataFrame()

                # Retry on other errors (5xx, timeouts)
                if attempt < max_retries - 1:
                    sleep_time = initial_delay * (backoff_factor**attempt)
                    logger.warning(
                        f"[DATA_ROUTER] Transient error for {ticker}: {error_msg}. Retrying in {sleep_time}s..."
                    )
                    import time

                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"[DATA_ROUTER] Max retries reached for {ticker}: {error_msg}"
                    )

        # Fallback (MANDATORY: Asset Class Isolation)
        fb_p = self.providers.get("yahoo")
        if fb_p and fb_p != provider:
            try:
                df = fb_p.fetch_ohlcv(ticker, start_date, end_date)
                if not df.empty and "Close" in df.columns:
                    df = self._ensure_utc_index(df)
                    logger.info(f"Fallback successful: {ticker} via YAHOO")
                    return df
            except Exception as e:
                error_msg = str(e)

                # PHASE 0 GUARD: No retry on entitlement failures
                if self._check_entitlement_failure(error_msg):
                    logger.error(
                        f"[DATA_ROUTER] ENTITLEMENT FAILURE on fallback for {ticker}: {error_msg}"
                    )
                    return pd.DataFrame()

                logger.debug(f"Yahoo fallback failed for {ticker}: {e}")

        return pd.DataFrame()

    def get_panel_parallel(
        self, tickers: List[str], start_date: str, end_date: str = None
    ) -> pd.DataFrame:
        """Fetch multiple tickers in parallel using multithreading."""
        tasks = [(tk, start_date, end_date) for tk in tickers]

        def _fetch_wrap(args):
            tk, sd, ed = args
            return tk, self.get_price_history(tk, sd, ed)

        results = list(self.executor.map(_fetch_wrap, tasks))

        data = {}
        for tk, df in results:
            if not df.empty:
                for col in df.columns:
                    data[(tk, col)] = df[col]

        if not data:
            return pd.DataFrame()

        panel = pd.DataFrame(data)
        if not isinstance(panel.columns, pd.MultiIndex):
            panel.columns = pd.MultiIndex.from_tuples(panel.columns)

        return panel

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get latest price for a ticker (Trust Primary)."""
        provider = self._get_best_provider(ticker)
        if not provider:
            return None

        try:
            price = provider.get_latest_quote(ticker)
            if price is not None:
                logger.debug(
                    f"Latest price for {ticker}: {price} from {provider.__class__.__name__}"
                )
                return price
        except Exception as e:
            logger.debug(f"Primary price fetch failed for {ticker}: {e}")

        # Fallback (Yahoo)
        fb_p = self.providers.get("yahoo")
        if fb_p and fb_p != provider:
            try:
                price = fb_p.get_latest_quote(ticker)
                if price is not None:
                    return price
            except Exception:
                pass

        return None

    def get_latest_prices_parallel(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch latest prices for multiple tickers in parallel."""
        results = list(self.executor.map(self.get_latest_price, tickers))
        return {tk: pr for tk, pr in zip(tickers, results) if pr is not None}

    async def get_latest_price_async(self, ticker: str) -> Optional[float]:
        """Async version of get_latest_price."""
        import asyncio

        return await asyncio.to_thread(self.get_latest_price, ticker)

    def get_macro_context(self) -> dict:
        """Get macroeconomic context data."""
        macro_data = {}

        # Try Bloomberg for institutional macro data
        if "bloomberg" in self.providers:
            try:
                macro_data.update(
                    self.providers["bloomberg"].get_institutional_data("SPX", "macro")
                )
            except Exception as e:
                logger.warning(f"Failed to get macro data from Bloomberg: {e}")

        return macro_data

        return macro_data

    def stream_realtime_tick(self, symbol: str, price: float, volume: int, producer):
        """
        Stream a real-time tick to the Kafka Producer.
        Acts as a bridge between DataRouter logic and Streaming Ingestion.
        """
        if not producer:
            return

        try:
            # Use the producer's send_tick method
            producer.send_tick(symbol, price, volume, provider="data_router_bridge")
            logger.debug(f"[STREAM] Sent {symbol} tick to Kafka")
        except Exception as e:
            logger.error(f"[STREAM] Failed to stream tick for {symbol}: {e}")

    def load_market_data(
        self, symbols: List[str], db: DatabaseManager
    ) -> Dict[str, pd.DataFrame]:
        """
        Canonical Market Data Loader (Objective 3).
        Returns last 252 daily bars per symbol.

        Strict Enforcement:
        - Only symbols in ACTIVE state are loaded.
        - If < 252 bars available for an ACTIVE symbol:
            * Mark symbol as DEGRADED.
            * Log governance decision.
            * Exclude from this trading cycle.
        """
        results = {}
        for symbol in symbols:
            # 1. Check Governance State
            with db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT state FROM symbol_governance WHERE symbol = ?", (symbol,)
                )
                row = cursor.fetchone()
            state = row[0] if row else "QUARANTINED"

            if state != "ACTIVE":
                logger.info(f"[DATA_LOADER] Skipping {symbol}: State={state}")
                continue

            # 2. Fetch last 252 bars from DB
            history = db.get_daily_prices(symbol, limit=252)
            if len(history) < 252:
                logger.warning(
                    f"[DATA_LOADER] Insufficient data for {symbol}: {len(history)} < 252. Downgrading."
                )
                # Atomic Governance Downgrade
                with db.get_connection() as conn:
                    conn.execute(
                        "UPDATE symbol_governance SET state = 'DEGRADED', reason = 'Insufficient 252-bar lookback' WHERE symbol = ?",
                        (symbol,),
                    )
                db.log_governance_decision(symbol, "DOWNGRADE", "INSUFFICIENT_LOOKBACK")
                continue

            # Already a DataFrame (Objective 3)
            df = ensure_business_days(history)

            # Map columns to OHLCV (Case-sensitive alignment)
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            results[symbol] = df

        logger.info(f"[DATA_LOADER] Loaded 252 bars for {len(results)} symbols")
        return results


# =============================================================================
# Data Purpose Provider Map (ABSOLUTE)
# =============================================================================
# This map enforces strict data separation as per Phase 1

DATA_PURPOSE_PROVIDER_MAP = {
    "5y_history": {
        "providers": ["yahoo", "polygon"],
        "max_history_days": 5000,
        "allow_long_history": True,
    },
    "fx_history": {
        "providers": ["yahoo"],
        "max_history_days": 5000,
        "allow_long_history": True,
    },
    "commodities_history": {
        "providers": ["yahoo"],
        "max_history_days": 5000,
        "allow_long_history": True,
    },
    "crypto_history": {
        "providers": ["yahoo", "binance"],
        "max_history_days": 1000,
        "allow_long_history": True,
    },
    "live_quotes": {
        "providers": ["alpaca"],
        "max_history_days": 1,
        "allow_long_history": False,
    },
    "execution": {
        "providers": ["alpaca"],
        "max_history_days": 0,
        "allow_long_history": False,
    },
    "macro_data": {
        "providers": ["fred"],
        "max_history_days": None,
        "allow_long_history": True,
    },
}


def get_provider_for_purpose(purpose: str, symbol: str = None) -> str:
    """
    Get the approved provider for a specific data purpose.

    Enforces ABSOLUTE data separation.
    """
    if purpose not in DATA_PURPOSE_PROVIDER_MAP:
        return "NO_VALID_PROVIDER"

    config = DATA_PURPOSE_PROVIDER_MAP[purpose]
    providers = config["providers"]

    # Return first available provider
    for provider_name in providers:
        # Check if provider is enabled
        env_key = f"{provider_name.upper()}_ENABLED"
        if os.getenv(env_key, "true").lower() == "false":
            continue

        return provider_name

    return "NO_VALID_PROVIDER"
