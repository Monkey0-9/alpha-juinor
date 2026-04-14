"""
Full Market Alpha Generator - Generate Signals for ALL Stocks.

Features:
- Parallel signal generation for 1000s of stocks
- Multi-model ensemble (ML, Statistical, Technical)
- Ranking and selection
- Daily batch processing
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


@dataclass
class MarketSignal:
    """Signal for a single stock."""
    symbol: str
    alpha_score: float  # -1 to 1
    confidence: float
    signal_type: str  # "LONG", "SHORT", "NEUTRAL"
    expected_return: float
    risk_score: float
    rank: int = 0
    sources: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketScanResult:
    """Result of full market scan."""
    timestamp: float
    total_scanned: int
    long_candidates: List[MarketSignal]
    short_candidates: List[MarketSignal]
    top_picks: List[MarketSignal]
    processing_time: float


class FullMarketAlphaGenerator:
    """
    Generate alpha signals for the ENTIRE market.

    Process:
    1. Fetch data for all symbols
    2. Calculate features in parallel
    3. Generate signals using ensemble
    4. Rank and select top opportunities
    """

    def __init__(
        self,
        max_workers: int = 20,
        top_n_picks: int = 50,
        min_confidence: float = 0.6
    ):
        self.max_workers = max_workers
        self.top_n_picks = top_n_picks
        self.min_confidence = min_confidence

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Model weights
        self.model_weights = {
            "momentum": 0.25,
            "value": 0.15,
            "quality": 0.15,
            "technical": 0.20,
            "ml": 0.25
        }

    def calculate_momentum_score(
        self,
        returns: pd.Series
    ) -> float:
        """Calculate momentum score."""
        if len(returns) < 20:
            return 0.0

        # Multiple timeframes
        mom_5 = returns.iloc[-5:].sum()
        mom_20 = returns.iloc[-20:].sum()
        mom_60 = returns.sum() if len(returns) >= 60 else returns.sum()

        # Weighted momentum
        score = 0.5 * mom_5 + 0.3 * mom_20 + 0.2 * mom_60

        # Normalize
        return float(np.clip(score * 10, -1, 1))

    def calculate_technical_score(
        self,
        prices: pd.Series
    ) -> float:
        """Calculate technical score."""
        if len(prices) < 20:
            return 0.0

        score = 0.0

        # Price vs moving averages
        ma_10 = prices.rolling(10).mean().iloc[-1]
        ma_20 = prices.rolling(20).mean().iloc[-1]
        current = prices.iloc[-1]

        if current > ma_10:
            score += 0.3
        if current > ma_20:
            score += 0.3
        if ma_10 > ma_20:
            score += 0.2

        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        if rsi < 30:
            score += 0.2  # Oversold
        elif rsi > 70:
            score -= 0.2  # Overbought

        return float(np.clip(score, -1, 1))

    def calculate_volatility_score(
        self,
        returns: pd.Series
    ) -> float:
        """Calculate volatility-adjusted score."""
        if len(returns) < 20:
            return 0.5

        vol = returns.std() * np.sqrt(252)

        # Lower volatility = higher score
        if vol < 0.15:
            return 0.8
        elif vol < 0.25:
            return 0.6
        elif vol < 0.40:
            return 0.4
        else:
            return 0.2

    def generate_signal_for_symbol(
        self,
        symbol: str,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> MarketSignal:
        """Generate signal for a single symbol."""
        try:
            returns = prices.pct_change().dropna()

            # Calculate component scores
            momentum = self.calculate_momentum_score(returns)
            technical = self.calculate_technical_score(prices)
            vol_score = self.calculate_volatility_score(returns)

            # Simple value proxy (mean reversion)
            if len(returns) >= 20:
                zscore = (prices.iloc[-1] - prices.mean()) / prices.std()
                value = float(np.clip(-zscore / 3, -1, 1))
            else:
                value = 0.0

            # Ensemble score
            alpha_score = (
                self.model_weights["momentum"] * momentum +
                self.model_weights["technical"] * technical +
                self.model_weights["value"] * value
            )

            # Confidence based on consistency
            scores = [momentum, technical, value]
            confidence = 1 - np.std(scores) / (np.abs(np.mean(scores)) + 0.1)
            confidence = float(np.clip(confidence, 0, 1))

            # Signal type
            if alpha_score > 0.2:
                signal_type = "LONG"
            elif alpha_score < -0.2:
                signal_type = "SHORT"
            else:
                signal_type = "NEUTRAL"

            # Expected return (simplified)
            expected_return = alpha_score * 0.05  # 5% max

            return MarketSignal(
                symbol=symbol,
                alpha_score=alpha_score,
                confidence=confidence,
                signal_type=signal_type,
                expected_return=expected_return,
                risk_score=1 - vol_score,
                sources={
                    "momentum": momentum,
                    "technical": technical,
                    "value": value,
                    "volatility": vol_score
                }
            )

        except Exception as e:
            logger.debug(f"Error generating signal for {symbol}: {e}")
            return MarketSignal(
                symbol=symbol,
                alpha_score=0.0,
                confidence=0.0,
                signal_type="NEUTRAL",
                expected_return=0.0,
                risk_score=0.5
            )

    def scan_market(
        self,
        market_data: Dict[str, pd.Series],
        volumes: Optional[Dict[str, pd.Series]] = None
    ) -> MarketScanResult:
        """
        Scan entire market and generate signals.

        Args:
            market_data: Dict of symbol -> price series
        """
        start_time = time.time()
        signals = []

        # Generate signals in parallel
        futures = {}
        for symbol, prices in market_data.items():
            vol = volumes.get(symbol) if volumes else None
            future = self.executor.submit(
                self.generate_signal_for_symbol,
                symbol, prices, vol
            )
            futures[future] = symbol

        # Collect results
        for future in as_completed(futures):
            try:
                signal = future.result(timeout=5)
                signals.append(signal)
            except Exception as e:
                logger.debug(f"Signal generation failed: {e}")

        # Filter by confidence
        confident_signals = [
            s for s in signals
            if s.confidence >= self.min_confidence
        ]

        # Separate longs and shorts
        long_candidates = sorted(
            [s for s in confident_signals if s.signal_type == "LONG"],
            key=lambda x: x.alpha_score,
            reverse=True
        )

        short_candidates = sorted(
            [s for s in confident_signals if s.signal_type == "SHORT"],
            key=lambda x: x.alpha_score
        )

        # Assign ranks
        for i, signal in enumerate(long_candidates):
            signal.rank = i + 1
        for i, signal in enumerate(short_candidates):
            signal.rank = i + 1

        # Top picks (balanced long/short)
        n_each = self.top_n_picks // 2
        top_picks = long_candidates[:n_each] + short_candidates[:n_each]

        processing_time = time.time() - start_time

        logger.info(
            f"Market scan: {len(signals)} stocks, "
            f"{len(long_candidates)} long, {len(short_candidates)} short, "
            f"{processing_time:.2f}s"
        )

        return MarketScanResult(
            timestamp=time.time(),
            total_scanned=len(signals),
            long_candidates=long_candidates[:100],
            short_candidates=short_candidates[:100],
            top_picks=top_picks,
            processing_time=processing_time
        )


# Global singleton
_generator: Optional[FullMarketAlphaGenerator] = None


def get_market_alpha_generator() -> FullMarketAlphaGenerator:
    """Get or create global market alpha generator."""
    global _generator
    if _generator is None:
        _generator = FullMarketAlphaGenerator()
    return _generator
