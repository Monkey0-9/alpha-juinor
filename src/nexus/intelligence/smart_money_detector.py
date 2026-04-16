"""
Smart Money Detector - Track Institutional Flow
=================================================

Detects when institutions (smart money) are accumulating or distributing.

Features:
1. Volume Profile Analysis
2. Accumulation/Distribution
3. Dark Pool Activity Detection
4. Block Trade Analysis
5. Institutional Footprint

When smart money moves, we follow.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

getcontext().prec = 50


@dataclass
class SMCOpportunity:
    """Smart Money Concept Opportunity Structure."""

    type: Literal[
        "BullishOB", "BearishOB", "LiquidityGap", "BullishHunt", "BearishHunt"
    ]
    confidence: float  # 0.0-1.0
    timeframe: str  # "15m", "1h", "4h", "1d"
    level: float  # Price level
    strength: int  # 1-5
    symbol: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SmartMoneySignal:
    """Smart money activity signal."""

    symbol: str
    timestamp: datetime

    # Activity type
    activity: str  # ACCUMULATION, DISTRIBUTION, NEUTRAL
    strength: float  # 0 to 1

    # Evidence
    volume_anomaly: float
    price_volume_divergence: bool
    block_trade_detected: bool
    dark_pool_print: bool

    # Interpretation
    direction: str  # BULLISH, BEARISH
    confidence: float

    # Trade suggestion
    suggested_action: str
    entry_zone_low: Decimal
    entry_zone_high: Decimal
    stop_loss: Decimal
    target: Decimal

    # Reasoning
    reasoning: List[str]


class SmartMoneyDetector:
    """
    Detects institutional (smart money) activity.

    Smart money leaves footprints:
    - Unusual volume patterns
    - Price-volume divergences
    - Block trades
    - Accumulation patterns

    We detect and follow.
    """

    def __init__(self):
        """Initialize the detector."""
        self.detections = 0

        logger.info(
            "[SMARTMONEY] Smart Money Detector initialized - " "FOLLOWING THE WHALES"
        )

    def analyze(
        self, symbol: str, prices: pd.Series, volumes: pd.Series
    ) -> Optional[SmartMoneySignal]:
        """Analyze for smart money activity."""
        if len(prices) < 50 or len(volumes) < 50:
            return None

        p = prices.values
        v = volumes.values
        current = Decimal(str(p[-1]))

        reasoning = []
        signals = []

        # 1. Volume Anomaly Detection
        vol_mean = np.mean(v[-50:])
        vol_std = np.std(v[-50:])
        vol_z = (v[-1] - vol_mean) / (vol_std + 1e-10)

        volume_anomaly = float(vol_z)

        # VAST UPGRADE: Stricter Volume Threshold (2.0 -> 2.5)
        # VAST UPGRADE: Stricter Volume Threshold (2.0 -> 2.5)
        if vol_z > 2.5:
            reasoning.append(f"Institutional Volume Spike: {vol_z:.1f}σ")
            signals.append(("VOLUME", min(1.0, vol_z / 4)))  # Capped contribution
        elif vol_z < 0.5 and abs(p[-1] - p[-2]) / p[-2] > 0.02:
            # Price moved 2% on low volume -> FAKE MOVE (Trap)
            reasoning.append("Low Volume Price Move (Possible Trap)")
            signals.append(("TRAP", 1.0))  # Strong warning

        # 2. On-Balance Volume (OBV) trend
        delta = np.sign(np.diff(p))
        obv = np.cumsum(delta * v[1:])
        obv_trend = obv[-1] - obv[-20] if len(obv) >= 20 else 0

        price_trend = p[-1] - p[-20] if len(p) >= 20 else 0

        # Price-volume divergence (smart money signal)
        divergence = False
        if price_trend < 0 and obv_trend > 0:
            divergence = True
            reasoning.append("Bullish divergence: Price down, OBV up")
            signals.append(("BULLISH_DIV", 0.8))
        elif price_trend > 0 and obv_trend < 0:
            divergence = True
            reasoning.append("Bearish divergence: Price up, OBV down")
            signals.append(("BEARISH_DIV", 0.8))

        # 3. Accumulation/Distribution
        # Money Flow Multiplier (Unused)
        # mf_mult = ((p[-1] - np.min(p[-1:])) - (np.max(p[-1:]) - p[-1])) / (
        #     np.max(p[-1:]) - np.min(p[-1:]) + 1e-10
        # )
        # mf_volume = mf_mult * v[-1] # Unused

        # Accumulation pattern: low volume on dips, high on rallies
        up_days_vol = np.mean([v[i] for i in range(-20, 0) if p[i] > p[i - 1]])
        down_days_vol = np.mean([v[i] for i in range(-20, 0) if p[i] < p[i - 1]])

        if up_days_vol > down_days_vol * 1.3:
            reasoning.append("Accumulation: Higher volume on up days")
            signals.append(("ACCUMULATION", 0.7))
        elif down_days_vol > up_days_vol * 1.3:
            reasoning.append("Distribution: Higher volume on down days")
            signals.append(("DISTRIBUTION", 0.7))

        # 4. Block Trade Detection (proxy)
        # Large single-candle moves with high volume
        daily_range = (np.max(p[-1:]) - np.min(p[-1:])) / np.mean(p[-1:])
        if daily_range > 0.03 and vol_z > 2:
            reasoning.append("Block trade signature detected")
            signals.append(("BLOCK", 0.6))

        # 5. Dark Pool Print (proxy)
        # Price moves without much public volume often indicate dark pool
        # We detect this as unusual price efficiency
        price_move = abs(p[-1] / p[-2] - 1) if len(p) >= 2 else 0
        vol_ratio = v[-1] / vol_mean

        if price_move > 0.02 and vol_ratio < 0.8:
            reasoning.append("Possible dark pool activity")
            signals.append(("DARK_POOL", 0.5))

        # Aggregate signals
        if not signals:
            return None

        # Determine direction
        bullish_score = sum(
            s[1] for s in signals if s[0] in ["ACCUMULATION", "BULLISH_DIV", "BLOCK"]
        )
        bearish_score = sum(
            s[1] for s in signals if s[0] in ["DISTRIBUTION", "BEARISH_DIV"]
        )

        if bullish_score > bearish_score and bullish_score > 0.5:
            direction = "BULLISH"
            activity = "ACCUMULATION"
            action = "BUY"
            stop = current * Decimal("0.95")
            target = current * Decimal("1.15")
        elif bearish_score > bullish_score and bearish_score > 0.5:
            direction = "BEARISH"
            activity = "DISTRIBUTION"
            action = "SELL"
            stop = current * Decimal("1.05")
            target = current * Decimal("0.88")
        else:
            return None

        strength = max(bullish_score, bearish_score) / 2
        confidence = 0.60 + strength * 0.25

        self.detections += 1

        return SmartMoneySignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            activity=activity,
            strength=min(1.0, strength),
            volume_anomaly=volume_anomaly,
            price_volume_divergence=divergence,
            block_trade_detected=any(s[0] == "BLOCK" for s in signals),
            dark_pool_print=any(s[0] == "DARK_POOL" for s in signals),
            direction=direction,
            confidence=min(0.90, confidence),
            suggested_action=action,
            entry_zone_low=(current * Decimal("0.99")).quantize(Decimal("0.01")),
            entry_zone_high=(current * Decimal("1.01")).quantize(Decimal("0.01")),
            stop_loss=stop.quantize(Decimal("0.01")),
            target=target.quantize(Decimal("0.01")),
            reasoning=reasoning,
        )

    def detect_smc_opportunities(
        self,
        symbol: str,
        prices: pd.Series,
        volumes: pd.Series,
        high: pd.Series,
        low: pd.Series,
        open_px: pd.Series,
    ) -> List[SMCOpportunity]:
        """
        Detect specific SMC patterns: Order Blocks, Liquidity Gaps, Hunts.
        Returns structured SMCOpportunity objects.
        """
        opportunities = []
        if len(prices) < 50:
            return opportunities

        # Convert to numpy for performance
        c = prices.values
        h = high.values
        l = low.values
        o = open_px.values
        v = volumes.values

        # 1. Fair Value Gaps (FVG)
        # Bullish FVG: High[i-2] < Low[i]
        # Bearish FVG: Low[i-2] > High[i]

        # Check last 5 candles for fresh FVGs
        for i in range(len(c) - 1, len(c) - 6, -1):
            if i < 2:
                continue

            # Bullish FVG
            if l[i] > h[i - 2]:
                gap_size = l[i] - h[i - 2]
                avg_range = np.mean(h[i - 5 : i] - l[i - 5 : i])
                if gap_size > avg_range * 0.5:  # Significant gap
                    opportunities.append(
                        SMCOpportunity(
                            type="LiquidityGap",
                            confidence=0.8,
                            timeframe="1d",  # Assuming daily for now, or need to pass timeframe
                            level=float(h[i - 2]),  # Support level is top of candle 1
                            strength=3,
                            symbol=symbol,
                        )
                    )

            # Bearish FVG
            if h[i] < l[i - 2]:
                gap_size = l[i - 2] - h[i]
                avg_range = np.mean(h[i - 5 : i] - l[i - 5 : i])
                if gap_size > avg_range * 0.5:
                    opportunities.append(
                        SMCOpportunity(
                            type="LiquidityGap",
                            confidence=0.8,
                            timeframe="1d",
                            level=float(
                                l[i - 2]
                            ),  # Resistance level is bottom of candle 1
                            strength=3,
                            symbol=symbol,
                        )
                    )

        # 2. Order Blocks (OB)
        # Simplified: Last opposing candle before strong displacement
        # Look for strong moves in last 10 candles
        for i in range(len(c) - 1, len(c) - 10, -1):
            if i < 3:
                continue

            # Identify big move (Displacement)
            body = abs(c[i] - o[i])
            avg_body = np.mean([abs(c[k] - o[k]) for k in range(i - 5, i)])

            if body > avg_body * 2.0:  # Strong displacement candle
                # Bullish Displacement (Green candle) -> Look for Bearish OB before it
                if c[i] > o[i]:
                    # Previous candle should be red (down)
                    if c[i - 1] < o[i - 1]:
                        opportunities.append(
                            SMCOpportunity(
                                type="BullishOB",
                                confidence=0.75,
                                timeframe="1d",
                                level=float(h[i - 1]),  # Entry at high of OB
                                strength=4,
                                symbol=symbol,
                            )
                        )

                # Bearish Displacement (Red candle) -> Look for Bullish OB before it
                elif c[i] < o[i]:
                    # Previous candle should be green (up)
                    if c[i - 1] > o[i - 1]:
                        opportunities.append(
                            SMCOpportunity(
                                type="BearishOB",
                                confidence=0.75,
                                timeframe="1d",
                                level=float(l[i - 1]),  # Entry at low of OB
                                strength=4,
                                symbol=symbol,
                            )
                        )

        # 3. Liquidity Hunt (Sweep)
        # Price takes out a recent swing high/low and closes back inside
        # Look at last candle
        curr_h = h[-1]
        curr_l = l[-1]
        curr_c = c[-1]

        # Find recent swing high/low (last 20 candles excluding current)
        recent_high = np.max(h[-20:-1])
        recent_low = np.min(l[-20:-1])

        # Swept High (Bearish Hunt)
        if curr_h > recent_high and curr_c < recent_high:
            opportunities.append(
                SMCOpportunity(
                    type="BearishHunt",
                    confidence=0.85,
                    timeframe="1d",
                    level=float(recent_high),
                    strength=5,
                    symbol=symbol,
                )
            )

        # Swept Low (Bullish Hunt)
        if curr_l < recent_low and curr_c > recent_low:
            opportunities.append(
                SMCOpportunity(
                    type="BullishHunt",
                    confidence=0.85,
                    timeframe="1d",
                    level=float(recent_low),
                    strength=5,
                    symbol=symbol,
                )
            )
        return opportunities

    def scan_market(self, market_data: pd.DataFrame) -> List[SmartMoneySignal]:
        """Scan entire market for smart money activity."""
        signals = []

        if isinstance(market_data.columns, pd.MultiIndex):
            symbols = list(market_data.columns.get_level_values(0).unique())
        else:
            return signals

        for symbol in symbols:
            try:
                # Need OHLCV for SMC
                df = market_data[symbol]
                prices = df["Close"].dropna()
                volumes = df.get("Volume", pd.Series()).dropna()

                # Legacy Analysis
                signal = self.analyze(symbol, prices, volumes)
                if signal and signal.confidence >= 0.70:
                    signals.append(signal)

                # New SMC Analysis (Logging findings for now, or storing in separate cache?)
                # For Phase 3 transition, we start using it.
                # However, this method returns List[SmartMoneySignal].
                # To support transition, we stick to legacy return for now,
                # but we could wrap SMC into logic later.
                # For now, just printing/logging SMC could be useful verification.

            except Exception:
                continue

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)

        return signals


class InstitutionalFlowTracker:
    """
    Tracks institutional order flow.

    Aggregates multiple flow indicators to determine
    institutional positioning.
    """

    def __init__(self):
        """Initialize the tracker."""
        self.smart_money = SmartMoneyDetector()

        logger.info("[FLOW] Institutional Flow Tracker initialized")

    def get_flow_score(
        self, symbol: str, prices: pd.Series, volumes: pd.Series
    ) -> Dict[str, Any]:
        """Get institutional flow score for a symbol."""
        if len(prices) < 50 or len(volumes) < 50:
            return {"flow_score": 0, "direction": "NEUTRAL"}

        p = prices.values
        v = volumes.values

        # Multiple flow indicators

        # 1. Money Flow Index (MFI)
        typical_price = p
        raw_mf = typical_price * v

        positive_mf = sum(raw_mf[i] for i in range(-14, 0) if p[i] > p[i - 1])
        negative_mf = sum(raw_mf[i] for i in range(-14, 0) if p[i] < p[i - 1])

        mfi = 100 - 100 / (1 + positive_mf / (negative_mf + 1e-10))

        # 2. Chaikin Money Flow
        clv = ((p - np.min(p)) - (np.max(p) - p)) / (np.max(p) - np.min(p) + 1e-10)
        cmf = np.sum(clv[-20:] * v[-20:]) / np.sum(v[-20:])

        # 3. Force Index
        force = np.diff(p) * v[1:]
        force_ema = np.mean(force[-13:])

        # Aggregate flow score
        flow_score = 0

        if mfi > 60:
            flow_score += 0.3
        elif mfi < 40:
            flow_score -= 0.3

        if cmf > 0.1:
            flow_score += 0.3
        elif cmf < -0.1:
            flow_score -= 0.3

        if force_ema > 0:
            flow_score += 0.2
        elif force_ema < 0:
            flow_score -= 0.2

        # OBV trend
        delta = np.sign(np.diff(p))
        obv = np.cumsum(delta * v[1:])
        obv_slope = (obv[-1] - obv[-20]) / 20 if len(obv) >= 20 else 0

        if obv_slope > 0:
            flow_score += 0.2
        elif obv_slope < 0:
            flow_score -= 0.2

        # Determine direction
        if flow_score > 0.3:
            direction = "BULLISH"
        elif flow_score < -0.3:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "flow_score": flow_score,
            "direction": direction,
            "mfi": mfi,
            "cmf": cmf,
            "obv_slope": obv_slope,
            "force_index": force_ema,
        }

    def get_top_institutional_picks(
        self, market_data: pd.DataFrame, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get stocks with strongest institutional buying."""
        results = []

        if isinstance(market_data.columns, pd.MultiIndex):
            symbols = list(market_data.columns.get_level_values(0).unique())
        else:
            return results

        for symbol in symbols:
            try:
                prices = market_data[symbol]["Close"].dropna()
                volumes = market_data[symbol].get("Volume", pd.Series()).dropna()

                flow = self.get_flow_score(symbol, prices, volumes)
                flow["symbol"] = symbol
                results.append(flow)

            except Exception:
                continue

        # Sort by absolute flow score
        results.sort(key=lambda x: abs(x["flow_score"]), reverse=True)

        # Filter for significant flow
        significant = [r for r in results if abs(r["flow_score"]) > 0.4]

        return significant[:top_n]


# Singletons
_detector: Optional[SmartMoneyDetector] = None
_tracker: Optional[InstitutionalFlowTracker] = None


def get_smart_money_detector() -> SmartMoneyDetector:
    """Get or create the Smart Money Detector."""
    global _detector
    if _detector is None:
        _detector = SmartMoneyDetector()
    return _detector


def get_flow_tracker() -> InstitutionalFlowTracker:
    """Get or create the Flow Tracker."""
    global _tracker
    if _tracker is None:
        _tracker = InstitutionalFlowTracker()
    return _tracker
