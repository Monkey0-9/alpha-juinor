"""
Statistical Alpha Family - Statistical arbitrage and modeling signals.

Uses advanced statistical techniques like GARCH volatility modeling,
cointegration analysis, statistical arbitrage, and time series analysis.
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.arima.model import ARIMA
from .base_alpha import BaseAlpha

logger = logging.getLogger(__name__)

class StatisticalAlpha(BaseAlpha):
    """
    Statistical modeling-based alpha using advanced time series techniques.

    Generates signals from:
    - GARCH volatility modeling
    - Cointegration relationships
    - Statistical arbitrage pairs
    - ARIMA forecasting
    - Mean reversion with statistical bounds
    """

    def __init__(self):
        super().__init__()
        self.garch_lookback = 60      # Days for GARCH estimation
        self.cointegration_window = 100  # Window for cointegration tests
        self.arima_order = (1, 1, 1)  # ARIMA parameters
        self.z_score_threshold = 2.0  # Standard deviations for signals

    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate statistical modeling-based trading signal.

        Args:
            data: OHLCV data
            regime_context: Current market regime info

        Returns:
            Dict with signal, confidence, and metadata
        """
        if data is None or data.empty or "Close" not in data.columns:
            return {'signal': 0.0, 'confidence': 0.0, 'metadata': {'error': 'Invalid data or missing Close column'}}

        try:
            # Calculate statistical indicators
            garch_volatility = self._estimate_garch_volatility(data)
            cointegration_signal = self._calculate_cointegration_signal(data)
            arima_forecast = self._calculate_arima_forecast(data)
            statistical_arbitrage = self._calculate_statistical_arbitrage(data)

            # Composite statistical score
            statistical_score = self._composite_statistical_score(
                garch_volatility, cointegration_signal, arima_forecast, statistical_arbitrage
            )

            # Generate signal with statistical bounds
            signal = self._statistical_to_signal(statistical_score, data, regime_context)

            # Calculate confidence based on statistical significance
            confidence = self._calculate_statistical_confidence(data, statistical_score)

            return {
                'signal': signal,
                'confidence': confidence,
                'metadata': {
                    'garch_volatility': garch_volatility,
                    'cointegration_signal': cointegration_signal,
                    'arima_forecast': arima_forecast,
                    'statistical_arbitrage': statistical_arbitrage,
                    'statistical_score': statistical_score,
                    'regime_adjusted': regime_context is not None
                }
            }

        except Exception as e:
            logger.warning(f"Statistical alpha failed: {e}")
            return {'signal': 0.0, 'confidence': 0.0, 'metadata': {'error': str(e)}}

    def _estimate_garch_volatility(self, data: pd.DataFrame) -> float:
        """
        Estimate conditional volatility using GARCH-like model.
        Simplified implementation - real version would use arch library.
        """
        returns = data['Close'].pct_change().dropna()

        if len(returns) < 20:
            return data['Close'].pct_change().std() * np.sqrt(252)  # Fallback

        # Simplified GARCH(1,1) estimation
        omega = 0.000001  # Constant term
        alpha = 0.1       # ARCH term
        beta = 0.85       # GARCH term

        # Initialize volatility
        sigma2 = np.zeros(len(returns))
        sigma2[0] = returns.var()

        # Recursively estimate conditional variance
        for t in range(1, len(returns)):
            sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]

        # Current volatility (annualized)
        current_vol = np.sqrt(sigma2[-1]) * np.sqrt(252)

        return current_vol

    def _calculate_cointegration_signal(self, data: pd.DataFrame) -> float:
        """
        Calculate cointegration-based signal.
        Tests for mean-reverting relationships between price series.
        """
        if len(data) < self.cointegration_window:
            return 0.0

        # Use price vs moving average as cointegration proxy
        price = data['Close']
        ma = data['Close'].rolling(50).mean()

        # Test cointegration between price and MA
        try:
            _, p_value, _ = coint(price.tail(self.cointegration_window),
                                ma.tail(self.cointegration_window))

            # Cointegration signal: deviation from equilibrium
            spread = price - ma
            z_score = (spread - spread.mean()) / spread.std()

            # Signal when significantly deviated from mean
            if p_value < 0.05:  # Cointegrated
                signal = -z_score.iloc[-1] / self.z_score_threshold
                signal = np.clip(signal, -1, 1)
            else:
                signal = 0.0

        except Exception:
            # Fallback: simple mean reversion signal
            returns = data['Close'].pct_change(20)
            signal = -returns.iloc[-1] if not returns.empty else 0.0

        return signal

    def _calculate_arima_forecast(self, data: pd.DataFrame) -> float:
        """
        Calculate ARIMA-based forecast signal.
        """
        if len(data) < 30:
            return 0.0

        try:
            # Fit ARIMA model to returns
            returns = data['Close'].pct_change().dropna().tail(50)
            model = ARIMA(returns, order=self.arima_order)
            model_fit = model.fit()

            # Forecast next return
            forecast = model_fit.forecast(steps=1)[0]

            # Signal based on forecast direction and magnitude
            signal = np.tanh(forecast * 10)  # Scale forecast to signal

        except Exception:
            # Fallback: momentum signal
            momentum = data['Close'].pct_change(10).iloc[-1]
            signal = np.tanh(momentum * 5)

        return signal

    def _calculate_statistical_arbitrage(self, data: pd.DataFrame) -> float:
        """
        Calculate statistical arbitrage signal using pairs trading logic.
        """
        if len(data) < 40:
            return 0.0

        # Use price vs volume-weighted average price as pairs proxy
        price = data['Close']
        vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

        # Calculate spread
        spread = price - vwap

        # Z-score of spread
        spread_mean = spread.rolling(30).mean()
        spread_std = spread.rolling(30).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-8)

        # Statistical arbitrage signal
        signal = -z_score.iloc[-1] / self.z_score_threshold
        signal = np.clip(signal, -1, 1)

        return signal

    def _composite_statistical_score(self, garch_vol: float, coint_signal: float,
                                   arima_signal: float, arb_signal: float) -> float:
        """Create composite statistical score."""

        # Adjust weights based on market conditions
        vol_regime = "HIGH" if garch_vol > 0.3 else "NORMAL"

        if vol_regime == "HIGH":
            # In high vol, favor mean-reversion strategies
            weights = {
                'garch': 0.1,
                'cointegration': 0.4,
                'arima': 0.2,
                'arbitrage': 0.3
            }
        else:
            # In normal vol, balance all signals
            weights = {
                'garch': 0.2,
                'cointegration': 0.3,
                'arima': 0.3,
                'arbitrage': 0.2
            }

        composite_score = (
            weights['garch'] * (1.0 - garch_vol / 0.5) +  # Lower vol = higher score
            weights['cointegration'] * coint_signal +
            weights['arima'] * arima_signal +
            weights['arbitrage'] * arb_signal
        )

        return composite_score

    def _statistical_to_signal(self, statistical_score: float, data: pd.DataFrame,
                             regime_context: Dict[str, Any] = None) -> float:
        """Convert statistical score to trading signal."""

        # Base signal from statistical score
        signal = np.tanh(statistical_score * 3)

        # Volatility adjustment
        current_vol = data['Close'].pct_change().rolling(20).std().iloc[-1]
        if current_vol > 0.03:  # High vol environment
            signal *= 1.2  # Statistical models work better in volatile markets

        # Regime adjustment
        if regime_context:
            regime_tag = regime_context.get('regime_tag', 'NORMAL')
            if regime_tag in ['CRISIS', 'HIGH_VOL']:
                # Statistical arbitrage thrives in volatile markets
                signal *= 1.3
            elif regime_tag == 'LOW_VOL':
                # Less edge in quiet markets
                signal *= 0.8

        return self.normalize_signal(signal)

    def _calculate_statistical_confidence(self, data: pd.DataFrame, statistical_score: float) -> float:
        """Calculate confidence in statistical signal based on model stability."""

        # Confidence based on data length, signal consistency, and model fit
        data_length_factor = min(len(data) / self.cointegration_window, 1.0)
        signal_strength = abs(statistical_score)
        data_stability = 1.0 - data['Close'].pct_change().std()

        confidence_factors = [
            data_length_factor,
            min(signal_strength * 2, 1.0),
            data_stability
        ]

        confidence = np.mean(confidence_factors)
        return confidence
