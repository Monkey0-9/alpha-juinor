import logging
import warnings
from typing import Dict, Any, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.arima.model import ARIMA
from .base_alpha import BaseAlpha
from utils.metrics import metrics

logger = logging.getLogger(__name__)

# Minimum ARIMA sample size (3x parameters for (p, d, q))
MIN_ARIMA_SAMPLE = 30  # For (1,1,1) order: 3 * (1+1+1) = 9, safety margin to 30


class ModelNotFitError(Exception):
    """Raised when ARIMA cannot be fit due to insufficient data or convergence issues."""
    pass


class StatisticalAlpha(BaseAlpha):
    """
    Statistical modeling-based alpha using advanced time series techniques with hardening.

    Features:
    - ARIMA safe predict with EWMA fallback
    - Per-symbol failure tracking
    - Auto-degradation after 3 consecutive failures
    - Statsmodels warning suppression
    """

    def __init__(self):
        super().__init__()
        self.arima_order = (1, 1, 1)
        self.arima_fallbacks = 0

        # Per-symbol failure tracking for degradation
        self._symbol_failures = defaultdict(int)
        self._degraded_symbols = set()

    def _check_symbol_degraded(self, symbol: str) -> bool:
        """Check if symbol is degraded due to repeated ARIMA failures."""
        return symbol in self._degraded_symbols

    def _record_arima_failure(self, symbol: str):
        """Record ARIMA failure and check for degradation threshold."""
        self._symbol_failures[symbol] += 1

        if self._symbol_failures[symbol] >= 3:
            if symbol not in self._degraded_symbols:
                self._degraded_symbols.add(symbol)
                logger.warning(
                    f"[ARIMA] Symbol {symbol} DEGRADED after {self._symbol_failures[symbol]} "
                    f"consecutive failures. Forcing EWMA fallback."
                )
                # TODO: Update lifecycle state in database to DEGRADED_ARIMA

    def _reset_symbol_failures(self, symbol: str):
        """Reset failure counter on successful ARIMA fit."""
        if self._symbol_failures[symbol] > 0:
            self._symbol_failures[symbol] = 0

    def arima_safe_predict(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Institutional Grade ARIMA Wrapper with hardening.

        Args:
            data: Price DataFrame
            symbol: Symbol name for failure tracking

        Returns:
            Dict with signal, method, and optional reason

        Raises:
            ModelNotFitError: If data insufficient or model fails to fit
        """
        # Check if already degraded
        if self._check_symbol_degraded(symbol):
            return {"signal": self._calculate_ewma_fallback(data), "method": "EWMA", "reason": "DEGRADED_ARIMA"}

        returns = data['Close'].pct_change(fill_method=None).dropna().tail(100)

        # Pre-fit check: minimum sample size
        if len(returns) < MIN_ARIMA_SAMPLE:
            raise ModelNotFitError(f"Insufficient data for ARIMA: {len(returns)} < {MIN_ARIMA_SAMPLE}")

        try:
            # ARIMA fitting with exception handling
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=UserWarning)
                warnings.filterwarnings("error", category=RuntimeWarning)

                try:
                    model = ARIMA(returns, order=self.arima_order)
                    model_fit = model.fit(disp=False, maxiter=50)
                except (UserWarning, RuntimeWarning) as w:
                    # Warnings converted to exceptions
                    raise ModelNotFitError(f"ARIMA fit warning: {w}")

            # Check convergence
            if not getattr(model_fit, 'mle_retvals', {}).get('converged', True):
                raise ModelNotFitError("ARIMA non-convergence")

            forecast = model_fit.forecast(steps=1).iloc[0]

            # Success - reset failure counter
            self._reset_symbol_failures(symbol)

            return {
                "signal": float(np.tanh(forecast * 10)),
                "method": "ARIMA",
                "arima_failures": self._symbol_failures[symbol]
            }

        except (ModelNotFitError, Exception) as e:
            # Record failure
            self._record_arima_failure(symbol)
            self.arima_fallbacks += 1
            metrics.arima_fallbacks += 1

            logger.debug(
                f"[ARIMA] Fallback to EWMA for {symbol} "
                f"(failure {self._symbol_failures[symbol]}/3): {e}"
            )

            # Fall back to EWMA
            return {
                "signal": self._calculate_ewma_fallback(data),
                "method": "EWMA",
                "reason": str(e),
                "arima_failures": self._symbol_failures[symbol]
            }

    def generate_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Generate statistical alpha signal with ARIMA hardening."""
        symbol = kwargs.get("symbol", "UNKNOWN")

        try:
            arima_res = self.arima_safe_predict(data, symbol=symbol)
            return {
                "signal": arima_res["signal"],
                "confidence": 0.6 if arima_res["method"] == "ARIMA" else 0.3,
                "metadata": {
                    "method": arima_res["method"],
                    "arima_failures": arima_res.get("arima_failures", 0),
                    "degraded": symbol in self._degraded_symbols
                }
            }
        except Exception as e:
            logger.error(f"Statistical alpha failed for {symbol}: {e}")
            return {
                "signal": 0.0,
                "confidence": 0.0,
                "metadata": {"error": str(e), "symbol": symbol}
            }

    def _calculate_ewma_fallback(self, data: pd.DataFrame) -> float:
        """Exponentially weighted moving average fallback."""
        returns = data['Close'].pct_change(fill_method=None).dropna()
        if returns.empty:
            return 0.0
        return float(np.tanh(returns.ewm(span=20).mean().iloc[-1] * 5))
