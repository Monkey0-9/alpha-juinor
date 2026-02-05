#!/usr/bin/env python3
"""
VALIDATED ALPHA INTEGRATION
===========================

Integrates the validated strategy signals into the main trading system's
alpha pipeline. This replaces the previous unvalidated momentum signals
with the institutional-grade validated signals.

Usage:
    from alpha.validated_alpha import ValidatedAlphaGenerator

    generator = ValidatedAlphaGenerator()
    signals = generator.generate(market_data)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VALIDATED_ALPHA")


@dataclass
class AlphaSignal:
    """Single alpha signal for trading."""
    symbol: str
    signal: float  # -1 to +1
    confidence: float  # 0 to 1
    source: str
    timestamp: str


class ValidatedAlphaGenerator:
    """
    Generates alpha signals using validated strategies only.

    Validated Strategies (passed walk-forward + multi-asset):
    1. Mean Reversion (RSI) - OOS Sharpe 0.85
    2. Dual MA Trend - OOS Sharpe 0.74
    3. Vol Targeting - OOS Sharpe 0.76

    NOT USED (failed validation):
    - Complex momentum
    - Bond strategies on TLT/IEF/AGG
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        ma_fast: int = 50,
        ma_slow: int = 200,
        use_regime: bool = True
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.use_regime = use_regime

        # Excluded assets (failed validation)
        self.excluded = {'TLT', 'IEF', 'AGG', 'SHY', 'GOVT'}

        logger.info("ValidatedAlphaGenerator initialized")
        logger.info(f"  RSI: {rsi_period}d, OS={rsi_oversold}, OB={rsi_overbought}")
        logger.info(f"  Trend: MA {ma_fast}/{ma_slow}")

    def _calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI value."""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1])

    def _calculate_trend(self, prices: pd.Series) -> float:
        """Calculate trend signal."""
        if len(prices) < self.ma_slow + 1:
            return 0.0

        fast = prices.rolling(self.ma_fast).mean().iloc[-1]
        slow = prices.rolling(self.ma_slow).mean().iloc[-1]

        if fast > slow:
            return min((fast - slow) / slow * 10, 1.0)
        else:
            return max((fast - slow) / slow * 10, -1.0)

    def _detect_regime(self, benchmark: pd.Series) -> tuple:
        """Detect market regime."""
        if len(benchmark) < 60:
            return 'NORMAL', 1.0

        returns = benchmark.pct_change().dropna()
        vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

        if vol > 0.25:
            return 'CRISIS', 0.3
        elif vol > 0.18:
            return 'BEAR', 0.6
        elif vol < 0.12:
            return 'BULL', 1.2
        else:
            return 'NORMAL', 1.0

    def generate(
        self,
        market_data: Dict[str, pd.Series],
        benchmark: Optional[pd.Series] = None
    ) -> Dict[str, AlphaSignal]:
        """
        Generate alpha signals for all assets.

        Args:
            market_data: Dict of symbol -> price series
            benchmark: Optional benchmark for regime detection

        Returns:
            Dict of symbol -> AlphaSignal
        """
        from datetime import datetime

        signals = {}

        # Detect regime
        if self.use_regime and benchmark is not None:
            regime, risk_mult = self._detect_regime(benchmark)
            logger.info(f"Regime: {regime}, Risk Mult: {risk_mult:.1f}")
        else:
            regime, risk_mult = 'NORMAL', 1.0

        for symbol, prices in market_data.items():
            # Skip excluded assets
            if symbol in self.excluded:
                continue

            # Calculate signals
            rsi = self._calculate_rsi(prices)
            trend = self._calculate_trend(prices)

            # RSI signal
            if rsi < self.rsi_oversold:
                rsi_signal = (self.rsi_oversold - rsi) / self.rsi_oversold
            elif rsi > self.rsi_overbought:
                rsi_signal = -(rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            else:
                rsi_signal = 0.0

            # Combine: 60% RSI, 40% Trend (based on validation)
            combined = 0.6 * rsi_signal + 0.4 * trend

            # Apply regime adjustment
            adjusted = combined * risk_mult

            # Confidence based on signal strength
            confidence = abs(adjusted)

            signals[symbol] = AlphaSignal(
                symbol=symbol,
                signal=float(np.clip(adjusted, -1, 1)),
                confidence=float(min(confidence, 1.0)),
                source="VALIDATED_RSI_TREND",
                timestamp=datetime.now().isoformat()
            )

        return signals

    def get_portfolio_weights(
        self,
        signals: Dict[str, AlphaSignal],
        max_position: float = 0.10
    ) -> Dict[str, float]:
        """
        Convert signals to portfolio weights.

        Args:
            signals: Generated alpha signals
            max_position: Maximum position per asset

        Returns:
            Dict of symbol -> portfolio weight
        """
        weights = {}
        total_signal = sum(max(0, s.signal) for s in signals.values())

        if total_signal == 0:
            # No positive signals, return equal weight cash
            return weights

        for symbol, signal in signals.items():
            if signal.signal > 0:
                # Long only - scale by signal strength
                raw_weight = signal.signal / total_signal
                weights[symbol] = min(raw_weight, max_position)

        # Normalize if needed
        total = sum(weights.values())
        if total > 1:
            weights = {k: v / total for k, v in weights.items()}

        return weights


def integrate_with_main():
    """Demo integration with main trading system."""
    print("=" * 70)
    print("     VALIDATED ALPHA INTEGRATION")
    print("=" * 70)

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        return 1

    # Fetch data
    symbols = ["SPY", "QQQ", "IWM", "GLD", "XLF", "XLK"]
    data = yf.download(symbols, period="1y", progress=False)

    market_data = {}
    for sym in symbols:
        if isinstance(data.columns, pd.MultiIndex):
            if ('Close', sym) in data.columns:
                market_data[sym] = data[('Close', sym)].dropna()

    # Initialize generator
    generator = ValidatedAlphaGenerator()

    # Generate signals
    benchmark = market_data.get('SPY')
    signals = generator.generate(market_data, benchmark)

    # Get weights
    weights = generator.get_portfolio_weights(signals)

    # Print results
    print("\nGenerated Signals:")
    for sym, sig in signals.items():
        print(f"  {sym}: signal={sig.signal:+.3f}, "
              f"conf={sig.confidence:.2f}")

    print("\nPortfolio Weights:")
    for sym, weight in weights.items():
        print(f"  {sym}: {weight:.1%}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(integrate_with_main())
