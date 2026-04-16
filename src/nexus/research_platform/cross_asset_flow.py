"""
Cross-Asset Flow Mapping - Contagion Detection
=================================================

Extend flow analysis beyond single stocks.

Features:
1. ETF → constituent flow patterns
2. FX → commodity correlations
3. Sector rotation flow patterns
4. Risk-on/Risk-off flow indicators
5. Cross-market contagion detection

Flow moves markets. Track it across assets.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import numpy as np
import pandas as pd
import threading

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset classes."""
    EQUITY = "EQUITY"
    FIXED_INCOME = "FIXED_INCOME"
    COMMODITY = "COMMODITY"
    FX = "FX"
    VOLATILITY = "VOLATILITY"
    CRYPTO = "CRYPTO"


class FlowRelationType(Enum):
    """Type of flow relationship."""
    LEADING = "LEADING"       # Source leads target
    LAGGING = "LAGGING"       # Source lags target
    CORRELATED = "CORRELATED" # Move together
    INVERSE = "INVERSE"       # Move opposite
    CAUSAL = "CAUSAL"         # Granger-causal


@dataclass
class FlowRelationship:
    """A relationship between two asset flows."""
    source: str        # e.g., "XLF" (financials ETF)
    target: str        # e.g., "JPM" (JP Morgan)

    # Relationship type
    relation_type: FlowRelationType

    # Strength
    correlation: float
    lead_lag_days: int  # Positive = source leads

    # Statistical significance
    p_value: float
    is_significant: bool

    # Direction
    flow_direction: str  # SAME, OPPOSITE

    # Metadata
    source_class: AssetClass
    target_class: AssetClass
    last_updated: datetime


@dataclass
class CrossAssetSignal:
    """Signal derived from cross-asset flow."""
    timestamp: datetime
    target_symbol: str

    # Signal
    signal_direction: int  # 1 = bullish, -1 = bearish
    confidence: float

    # Source information
    source_flows: List[str]
    relationship_used: FlowRelationType

    # Explanation
    reasoning: str


# Pre-defined cross-asset relationships
ETF_RELATIONSHIPS = {
    # Tech sector
    "QQQ": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "AMD"],
    "SMH": ["NVDA", "AMD", "INTC", "TSM", "ASML", "QCOM"],

    # Financials
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK"],
    "KBE": ["JPM", "BAC", "WFC", "C", "USB", "PNC"],

    # Energy
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
    "OIH": ["SLB", "HAL", "BKR", "NOV"],

    # Commodities
    "GLD": ["NEM", "GOLD", "AEM", "FNV"],
    "SLV": ["AG", "PAAS", "HL"],

    # Health
    "XLV": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY"],
    "XBI": ["REGN", "VRTX", "GILD", "BIIB", "AMGN"],

    # Consumer
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM"],

    # Broad market
    "SPY": ["QQQ", "IWM", "DIA"],
    "IWM": ["PINS", "SNAP", "ETSY", "ROKU"],
}

# Risk-on/Risk-off pairs
RISK_PAIRS = [
    ("SPY", "TLT", "INVERSE"),    # Stocks vs bonds
    ("HYG", "TLT", "INVERSE"),    # High yield vs treasuries
    ("EEM", "UUP", "INVERSE"),    # EM vs dollar
    ("GLD", "UUP", "INVERSE"),    # Gold vs dollar
    ("VXX", "SPY", "INVERSE"),    # Vol vs stocks
]


class CrossAssetFlowMapper:
    """
    Maps and tracks flow relationships across assets.

    Detects contagion patterns:
    - ETF flow predicts constituent moves
    - FX impacts commodities
    - Sector rotation patterns
    """

    def __init__(self):
        """Initialize the mapper."""
        self.relationships: Dict[str, List[FlowRelationship]] = defaultdict(list)
        self.flow_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.signals_generated: List[CrossAssetSignal] = []

        # Initialize known relationships
        self._init_known_relationships()

        self._lock = threading.Lock()

        logger.info(
            f"[CROSS-ASSET] Flow Mapper initialized | "
            f"Tracking {len(ETF_RELATIONSHIPS)} ETF relationships"
        )

    def _init_known_relationships(self):
        """Initialize known ETF → constituent relationships."""
        for etf, constituents in ETF_RELATIONSHIPS.items():
            for constituent in constituents:
                self.relationships[etf].append(FlowRelationship(
                    source=etf,
                    target=constituent,
                    relation_type=FlowRelationType.LEADING,
                    correlation=0.7,  # Default assumption
                    lead_lag_days=1,
                    p_value=0.01,
                    is_significant=True,
                    flow_direction="SAME",
                    source_class=AssetClass.EQUITY,
                    target_class=AssetClass.EQUITY,
                    last_updated=datetime.utcnow()
                ))

    def record_flow(
        self,
        symbol: str,
        flow_value: float,  # Positive = buying, negative = selling
        timestamp: Optional[datetime] = None
    ):
        """Record a flow observation for a symbol."""
        ts = timestamp or datetime.utcnow()

        with self._lock:
            self.flow_history[symbol].append((ts, flow_value))

            # Keep last 252 observations
            if len(self.flow_history[symbol]) > 252:
                self.flow_history[symbol] = self.flow_history[symbol][-252:]

    def get_constituent_signals(
        self,
        etf_symbol: str,
        etf_flow: float
    ) -> List[CrossAssetSignal]:
        """
        Get signals for constituents based on ETF flow.

        If large buying in XLF → expect buying in JPM, BAC, etc.
        """
        signals = []

        with self._lock:
            if etf_symbol not in self.relationships:
                return signals

            for rel in self.relationships[etf_symbol]:
                if rel.relation_type != FlowRelationType.LEADING:
                    continue

                # Determine signal direction
                if rel.flow_direction == "SAME":
                    direction = 1 if etf_flow > 0 else -1
                else:
                    direction = -1 if etf_flow > 0 else 1

                # Confidence based on correlation and flow magnitude
                confidence = min(0.9, abs(rel.correlation) * abs(etf_flow) / 100)

                # Only generate signal if flow is significant
                if abs(etf_flow) > 0.1:  # Threshold
                    signal = CrossAssetSignal(
                        timestamp=datetime.utcnow(),
                        target_symbol=rel.target,
                        signal_direction=direction,
                        confidence=confidence,
                        source_flows=[etf_symbol],
                        relationship_used=rel.relation_type,
                        reasoning=f"ETF {etf_symbol} flow {etf_flow:+.1f} → {rel.target}"
                    )
                    signals.append(signal)
                    self.signals_generated.append(signal)

        return signals

    def detect_sector_rotation(
        self,
        sector_flows: Dict[str, float]  # Sector symbol → net flow
    ) -> Dict[str, Any]:
        """
        Detect sector rotation patterns.

        Money flowing OUT of X and INTO Y.
        """
        if not sector_flows:
            return {"rotation_detected": False}

        # Find inflows and outflows
        inflows = {s: f for s, f in sector_flows.items() if f > 0}
        outflows = {s: f for s, f in sector_flows.items() if f < 0}

        if not inflows or not outflows:
            return {"rotation_detected": False}

        # Top inflow and outflow
        top_inflow = max(inflows.items(), key=lambda x: x[1])
        top_outflow = min(outflows.items(), key=lambda x: x[1])

        # Rotation strength
        rotation_strength = (top_inflow[1] - top_outflow[1]) / 2

        return {
            "rotation_detected": rotation_strength > 1.0,
            "rotation_strength": rotation_strength,
            "money_leaving": top_outflow[0],
            "money_entering": top_inflow[0],
            "sectors": sector_flows,
            "signals": {
                "avoid": top_outflow[0],
                "favor": top_inflow[0]
            }
        }

    def detect_risk_regime(
        self,
        spy_flow: float,
        tlt_flow: float,
        vxx_flow: Optional[float] = None,
        gld_flow: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect risk-on/risk-off regime from flow patterns.
        """
        risk_score = 0  # Positive = risk-on, negative = risk-off
        signals = []

        # SPY vs TLT
        if spy_flow > 0 and tlt_flow < 0:
            risk_score += 2
            signals.append("SPY inflow + TLT outflow = RISK-ON")
        elif spy_flow < 0 and tlt_flow > 0:
            risk_score -= 2
            signals.append("SPY outflow + TLT inflow = RISK-OFF")

        # VXX flows
        if vxx_flow is not None:
            if vxx_flow > 0.5:
                risk_score -= 1
                signals.append("VXX buying = hedging = RISK-OFF")
            elif vxx_flow < -0.5:
                risk_score += 1
                signals.append("VXX selling = complacency = RISK-ON")

        # Gold flows
        if gld_flow is not None:
            if gld_flow > 0.5:
                risk_score -= 1
                signals.append("GLD buying = safe haven = RISK-OFF")

        # Determine regime
        if risk_score >= 2:
            regime = "RISK_ON"
            confidence = min(0.9, risk_score / 4)
        elif risk_score <= -2:
            regime = "RISK_OFF"
            confidence = min(0.9, abs(risk_score) / 4)
        else:
            regime = "NEUTRAL"
            confidence = 0.5

        return {
            "regime": regime,
            "risk_score": risk_score,
            "confidence": confidence,
            "signals": signals,
            "recommendation": {
                "RISK_ON": "Favor beta, growth, cyclicals",
                "RISK_OFF": "Favor quality, defensives, cash",
                "NEUTRAL": "Maintain balanced allocation"
            }.get(regime, "")
        }

    def calculate_flow_correlation(
        self,
        symbol1: str,
        symbol2: str,
        lookback_days: int = 60
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate flow correlation between two symbols.
        """
        with self._lock:
            if symbol1 not in self.flow_history or symbol2 not in self.flow_history:
                return None

            flows1 = self.flow_history[symbol1][-lookback_days:]
            flows2 = self.flow_history[symbol2][-lookback_days:]

        if len(flows1) < 10 or len(flows2) < 10:
            return None

        # Align by date
        # Simplified: assume same dates
        values1 = [f[1] for f in flows1]
        values2 = [f[1] for f in flows2[:len(values1)]]

        if len(values1) != len(values2):
            min_len = min(len(values1), len(values2))
            values1 = values1[:min_len]
            values2 = values2[:min_len]

        # Calculate correlation
        corr = np.corrcoef(values1, values2)[0, 1]

        # Lead-lag analysis (simple)
        best_lag = 0
        best_corr = corr

        for lag in range(1, 6):
            if lag < len(values1):
                lagged_corr = np.corrcoef(values1[:-lag], values2[lag:])[0, 1]
                if abs(lagged_corr) > abs(best_corr):
                    best_corr = lagged_corr
                    best_lag = lag

        return {
            "symbol1": symbol1,
            "symbol2": symbol2,
            "correlation": float(corr),
            "best_lag": best_lag,
            "best_correlation": float(best_corr),
            "relationship": "LEADING" if best_lag > 0 else "SIMULTANEOUS",
            "flow_direction": "SAME" if corr > 0 else "OPPOSITE"
        }

    def get_trading_signals(
        self,
        current_flows: Dict[str, float]
    ) -> List[CrossAssetSignal]:
        """
        Generate trading signals from current flows.
        """
        signals = []

        # ETF → Constituent signals
        for symbol, flow in current_flows.items():
            if symbol in ETF_RELATIONSHIPS:
                constituent_signals = self.get_constituent_signals(symbol, flow)
                signals.extend(constituent_signals)

        return signals

    def get_flow_summary(self) -> Dict[str, Any]:
        """Get summary of cross-asset flow patterns."""
        with self._lock:
            return {
                "symbols_tracked": len(self.flow_history),
                "relationships_mapped": sum(len(r) for r in self.relationships.values()),
                "signals_generated": len(self.signals_generated),
                "recent_signals": [
                    {
                        "target": s.target_symbol,
                        "direction": s.signal_direction,
                        "confidence": s.confidence,
                        "source": s.source_flows
                    }
                    for s in self.signals_generated[-10:]
                ]
            }


# Singleton
_mapper: Optional[CrossAssetFlowMapper] = None


def get_cross_asset_mapper() -> CrossAssetFlowMapper:
    """Get or create the Cross-Asset Flow Mapper."""
    global _mapper
    if _mapper is None:
        _mapper = CrossAssetFlowMapper()
    return _mapper
