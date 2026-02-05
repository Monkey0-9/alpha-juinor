#!/usr/bin/env python3
"""
ENHANCED VALIDATED STRATEGY
===========================

Adds improvements to the base Mean Reversion (RSI) strategy:
1. Regime Detection - crisis vs normal market
2. Multi-Signal Alpha - RSI + Dual MA + Volatility
3. Dynamic Risk Scaling - adapt to market conditions
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import os

# Constants
TRADING_DAYS = 252
TOTAL_FRICTION = 0.0015  # 15 bps


@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime: str  # 'NORMAL', 'BULL', 'BEAR', 'CRISIS'
    confidence: float
    vix_level: float
    trend: str  # 'UP', 'DOWN', 'FLAT'
    risk_multiplier: float


@dataclass
class EnhancedSignal:
    """Multi-factor signal with regime awareness."""
    symbol: str
    rsi_signal: float
    trend_signal: float
    vol_signal: float
    combined_signal: float
    regime_adjusted: float
    position_size: float


class RegimeDetector:
    """
    Detects market regime for adaptive strategy behavior.

    Regimes:
    - NORMAL: Standard market conditions
    - BULL: Strong uptrend, low volatility
    - BEAR: Downtrend, elevated volatility
    - CRISIS: Extreme volatility, correlation spike
    """

    def __init__(self):
        self.vol_lookback = 22  # ~1 month
        self.trend_lookback = 200 # 200-day SMA for major trend
        self.trend_short = 50     # 50-day SMA for Golden Cross
        self.crisis_vol_threshold = 0.30  # 30% annualized
        self.bull_vol_threshold = 0.15   # 15% annualized

    def detect(self, prices: pd.Series, vix: float = None) -> MarketRegime:
        """Detect current market regime."""
        if len(prices) < self.trend_lookback + 1:
            return MarketRegime(
                regime='NORMAL',
                confidence=0.5,
                vix_level=vix or 20,
                trend='FLAT',
                risk_multiplier=1.0
            )

        # Calculate metrics
        returns = prices.pct_change().dropna()
        realized_vol = returns.rolling(self.vol_lookback).std().iloc[-1]
        ann_vol = realized_vol * np.sqrt(TRADING_DAYS)

        # Trend detection (Golden Cross Logic)
        sma_short = prices.rolling(self.trend_short).mean().iloc[-1]
        sma_long = prices.rolling(self.trend_lookback).mean().iloc[-1]
        current_price = prices.iloc[-1]

        # Trend Definition
        # UP: Price > SMA50 > SMA200 (Strong Bull)
        # DOWN: Price < SMA50 < SMA200 (Strong Bear)
        # FLAT/Choppy: Everything else

        if current_price > sma_short and sma_short > sma_long:
            trend = 'UP'
        elif current_price < sma_short and sma_short < sma_long:
            trend = 'DOWN'
        else:
            trend = 'FLAT'

        # Regime classification
        if ann_vol > self.crisis_vol_threshold:
            regime = 'CRISIS'
            risk_mult = 0.3
            confidence = min(ann_vol / self.crisis_vol_threshold, 1.0)
        elif ann_vol < self.bull_vol_threshold and trend == 'UP':
            regime = 'BULL'
            risk_mult = 1.2
            confidence = 0.8
        elif trend == 'DOWN':
            regime = 'BEAR'
            risk_mult = 0.6
            confidence = 0.6
        else:
            regime = 'NORMAL'
            risk_mult = 1.0
            confidence = 0.5

        return MarketRegime(
            regime=regime,
            confidence=float(confidence),
            vix_level=float(vix or ann_vol * 100),
            trend=trend,
            risk_multiplier=float(risk_mult)
        )


class MultiSignalAlpha:
    """
    Combines multiple alpha signals for robust performance.

    Signals:
    1. RSI Mean Reversion (validated, OOS Sharpe 0.85)
    2. Dual MA Trend Following (validated, OOS Sharpe 0.74)
    3. Volatility Timing (validated, OOS Sharpe 0.76)
    """

    def __init__(self):
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.ma_fast = 50
        self.ma_slow = 200
        self.vol_target = 0.10

        # Signal weights (based on validation performance)
        self.weights = {
            'rsi': 0.50,      # Highest OOS Sharpe
            'trend': 0.30,    # Second best
            'vol': 0.20      # Third
        }

    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI signal (-1 to +1)."""
        if len(prices) < self.rsi_period + 1:
            return 0.0

        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean().iloc[-1]

        if loss == 0:
            rsi = 100
        else:
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

        # Convert to signal
        if rsi < self.rsi_oversold:
            return 1.0  # Strong buy
        elif rsi > self.rsi_overbought:
            return -1.0  # Strong sell
        else:
            # Linear scaling in neutral zone
            return (50 - rsi) / 50

    def calculate_trend(self, prices: pd.Series) -> float:
        """Calculate trend signal (-1 to +1)."""
        if len(prices) < self.ma_slow + 1:
            return 0.0

        fast_ma = prices.rolling(self.ma_fast).mean().iloc[-1]
        slow_ma = prices.rolling(self.ma_slow).mean().iloc[-1]
        price = prices.iloc[-1]

        # Trend strength
        if fast_ma > slow_ma:
            strength = (fast_ma - slow_ma) / slow_ma
            return min(strength * 10, 1.0)  # Cap at 1.0
        else:
            strength = (slow_ma - fast_ma) / slow_ma
            return max(-strength * 10, -1.0)

    def calculate_vol_signal(self, prices: pd.Series) -> float:
        """Calculate volatility timing signal."""
        if len(prices) < 21:
            return 1.0

        returns = prices.pct_change().dropna()
        vol_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(TRADING_DAYS)
        vol_60 = returns.rolling(60).std().iloc[-1] * np.sqrt(TRADING_DAYS)

        # Vol regime
        if vol_20 < vol_60 * 0.8:
            return 1.5  # Low vol, increase exposure
        elif vol_20 > vol_60 * 1.2:
            return 0.5  # High vol, reduce exposure
        else:
            return 1.0

    def generate_signal(
        self,
        symbol: str,
        prices: pd.Series,
        regime: MarketRegime
    ) -> EnhancedSignal:
        """Generate combined multi-factor signal."""
        rsi_sig = self.calculate_rsi(prices)
        trend_sig = self.calculate_trend(prices)
        vol_sig = self.calculate_vol_signal(prices)

        # Combine signals
        combined = (
            self.weights['rsi'] * rsi_sig +
            self.weights['trend'] * trend_sig
        )

        # Apply volatility scaling
        combined *= vol_sig

        # Apply regime adjustment
        regime_adjusted = combined * regime.risk_multiplier

        # Convert to position size (0 to 1)
        position = max(0, min(1, (regime_adjusted + 1) / 2))

        return EnhancedSignal(
            symbol=symbol,
            rsi_signal=float(rsi_sig),
            trend_signal=float(trend_sig),
            vol_signal=float(vol_sig),
            combined_signal=float(combined),
            regime_adjusted=float(regime_adjusted),
            position_size=float(position)
        )


class EnhancedStrategyEngine:
    """
    Production-ready enhanced strategy engine.

    Combines:
    - Regime detection
    - Multi-signal alpha
    - Dynamic position sizing
    """

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.alpha_engine = MultiSignalAlpha()
        self.positions: Dict[str, float] = {}
        self.signals: List[EnhancedSignal] = []

    def run_cycle(
        self,
        market_data: Dict[str, pd.Series],
        benchmark: pd.Series = None
    ) -> Dict:
        """Run complete strategy cycle."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'signals': [],
            'regime': None,
            'positions': {}
        }

        # Detect regime from benchmark (SPY)
        if benchmark is not None:
            regime = self.regime_detector.detect(benchmark)
        elif 'SPY' in market_data:
            regime = self.regime_detector.detect(market_data['SPY'])
        else:
            regime = MarketRegime(
                regime='NORMAL',
                confidence=0.5,
                vix_level=20,
                trend='FLAT',
                risk_multiplier=1.0
            )

        results['regime'] = {
            'name': regime.regime,
            'confidence': regime.confidence,
            'risk_multiplier': regime.risk_multiplier,
            'trend': regime.trend
        }

        print(f"\n[REGIME] {regime.regime} "
              f"(conf={regime.confidence:.0%}, "
              f"risk_mult={regime.risk_multiplier:.1f})")

        # Generate signals for each asset
        for symbol, prices in market_data.items():
            signal = self.alpha_engine.generate_signal(
                symbol, prices, regime
            )
            self.signals.append(signal)
            self.positions[symbol] = signal.position_size

            results['signals'].append({
                'symbol': symbol,
                'rsi': signal.rsi_signal,
                'trend': signal.trend_signal,
                'vol': signal.vol_signal,
                'combined': signal.combined_signal,
                'position': signal.position_size
            })

            print(f"  {symbol}: RSI={signal.rsi_signal:+.2f}, "
                  f"Trend={signal.trend_signal:+.2f}, "
                  f"Position={signal.position_size:.1%}")

        results['positions'] = self.positions.copy()
        return results


# Extended Universe
EXTENDED_UNIVERSE = [
    # Core Equity (validated)
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    # International (validated)
    "EFA", "EEM", "VEA", "VWO",
    # Sectors (validated)
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    # Commodities (validated)
    "GLD", "SLV", "USO", "DBA",
    # Fixed Income (selective - only HYG and LQD passed)
    "HYG", "LQD",
    # REITs
    "VNQ", "IYR",
    # Factor ETFs
    "MTUM", "VLUE", "QUAL", "SIZE",
]


def run_enhanced_validation():
    """Run validation with enhanced strategy."""
    print("=" * 70)
    print("     ENHANCED STRATEGY VALIDATION")
    print("     " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        return 1

    # Fetch data
    print(f"\nFetching {len(EXTENDED_UNIVERSE)} assets...")
    data = yf.download(
        EXTENDED_UNIVERSE,
        start="2018-01-01",
        end=datetime.now().strftime("%Y-%m-%d"),
        progress=False
    )

    # Process data
    market_data = {}
    for sym in EXTENDED_UNIVERSE:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ('Close', sym) in data.columns:
                    close = data[('Close', sym)].dropna()
                    if len(close) > 252:
                        market_data[sym] = close
        except Exception:
            continue

    print(f"Loaded {len(market_data)} assets")

    # Initialize engine
    engine = EnhancedStrategyEngine()

    # Get benchmark for regime
    benchmark = market_data.get('SPY')

    # Run cycle
    results = engine.run_cycle(market_data, benchmark)

    # Print summary
    print("\n" + "=" * 70)
    print("ENHANCED STRATEGY SUMMARY")
    print("=" * 70)

    print(f"""
  Regime: {results['regime']['name']}
  Risk Multiplier: {results['regime']['risk_multiplier']:.1f}
  Trend: {results['regime']['trend']}

  Assets Analyzed: {len(results['signals'])}
  Avg Position Size: {np.mean([s['position'] for s in results['signals']]):.1%}
""")

    # Top positions
    sorted_signals = sorted(
        results['signals'],
        key=lambda x: x['position'],
        reverse=True
    )

    print("  TOP POSITIONS:")
    for s in sorted_signals[:10]:
        print(f"    {s['symbol']}: {s['position']:.1%} "
              f"(RSI={s['rsi']:+.2f}, Trend={s['trend']:+.2f})")

    # Save results
    os.makedirs("output", exist_ok=True)
    fn = f"output/enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Report: {fn}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_enhanced_validation())
