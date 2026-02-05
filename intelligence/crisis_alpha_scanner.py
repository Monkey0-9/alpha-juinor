"""
Crisis Alpha Scanner - Real-Time Crisis Detection
====================================================

Don't wait for crisis. DETECT it early and PROFIT from it.

This module:
1. Scans for early-warning signatures of crises
2. VIX term structure spikes
3. Credit spread blowouts
4. FX volatility anomalies
5. Activates crisis playbook strategies

Turn crisis into alpha.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import threading

logger = logging.getLogger(__name__)


class CrisisType(Enum):
    """Types of market crises."""
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    CREDIT_CRUNCH = "CREDIT_CRUNCH"
    LIQUIDITY_DROUGHT = "LIQUIDITY_DROUGHT"
    FLASH_CRASH = "FLASH_CRASH"
    CORRELATION_BREAKDOWN = "CORRELATION_BREAKDOWN"
    FLIGHT_TO_QUALITY = "FLIGHT_TO_QUALITY"
    SECTOR_COLLAPSE = "SECTOR_COLLAPSE"
    CURRENCY_CRISIS = "CURRENCY_CRISIS"


class AlertLevel(Enum):
    """Alert severity levels."""
    GREEN = "GREEN"      # Normal conditions
    YELLOW = "YELLOW"    # Elevated risk
    ORANGE = "ORANGE"    # High risk
    RED = "RED"          # Crisis imminent/active


@dataclass
class CrisisIndicator:
    """A single crisis indicator reading."""
    name: str
    value: float
    threshold_yellow: float
    threshold_orange: float
    threshold_red: float

    # Historical context
    historical_mean: float
    historical_std: float
    z_score: float

    # Current alert level
    alert_level: AlertLevel

    # Trend
    trend_1h: float  # % change
    trend_24h: float

    # Last updated
    timestamp: datetime


@dataclass
class CrisisPlaybook:
    """A pre-defined response to a crisis type."""
    crisis_type: CrisisType
    name: str
    description: str

    # Position adjustments
    equity_exposure_pct: float  # Target equity exposure
    hedge_ratio: float          # Hedge ratio (0 = no hedge, 1 = fully hedged)
    cash_target_pct: float      # Target cash allocation

    # Strategies to activate
    activate_strategies: List[str]
    deactivate_strategies: List[str]

    # Instruments
    long_instruments: List[str]   # e.g., ["VIX", "TLT", "GLD"]
    short_instruments: List[str]  # e.g., ["SPY", "HYG"]

    # Risk adjustments
    position_size_multiplier: float  # 0.5 = half size
    stop_loss_tighter: float         # Tighten stops by this %


# Pre-defined crisis playbooks
CRISIS_PLAYBOOKS = {
    CrisisType.VOLATILITY_SPIKE: CrisisPlaybook(
        crisis_type=CrisisType.VOLATILITY_SPIKE,
        name="Volatility Spike Playbook",
        description="VIX > 30, volatility explosion",
        equity_exposure_pct=0.30,
        hedge_ratio=0.70,
        cash_target_pct=0.40,
        activate_strategies=["volatility_trend", "momentum_vol", "tail_hedging"],
        deactivate_strategies=["mean_reversion", "pairs_trading"],
        long_instruments=["VXX", "UVXY", "TLT", "GLD"],
        short_instruments=["HYG", "JNK"],
        position_size_multiplier=0.50,
        stop_loss_tighter=0.02
    ),

    CrisisType.CREDIT_CRUNCH: CrisisPlaybook(
        crisis_type=CrisisType.CREDIT_CRUNCH,
        name="Credit Crunch Playbook",
        description="Credit spreads blowing out",
        equity_exposure_pct=0.20,
        hedge_ratio=0.80,
        cash_target_pct=0.50,
        activate_strategies=["flight_to_quality", "treasury_momentum"],
        deactivate_strategies=["credit_long", "high_beta"],
        long_instruments=["TLT", "IEF", "GLD", "UUP"],
        short_instruments=["HYG", "JNK", "BKLN", "XLF"],
        position_size_multiplier=0.40,
        stop_loss_tighter=0.03
    ),

    CrisisType.LIQUIDITY_DROUGHT: CrisisPlaybook(
        crisis_type=CrisisType.LIQUIDITY_DROUGHT,
        name="Liquidity Drought Playbook",
        description="Market liquidity evaporating",
        equity_exposure_pct=0.25,
        hedge_ratio=0.75,
        cash_target_pct=0.60,
        activate_strategies=["large_cap_only", "liquid_etfs"],
        deactivate_strategies=["small_cap", "illiquid_names", "options_complex"],
        long_instruments=["SPY", "TLT", "GLD"],
        short_instruments=[],
        position_size_multiplier=0.30,
        stop_loss_tighter=0.05
    ),

    CrisisType.FLASH_CRASH: CrisisPlaybook(
        crisis_type=CrisisType.FLASH_CRASH,
        name="Flash Crash Playbook",
        description="Rapid market dislocation",
        equity_exposure_pct=0.10,
        hedge_ratio=0.90,
        cash_target_pct=0.70,
        activate_strategies=["dip_buyer"],  # Buy the crash
        deactivate_strategies=["momentum", "trend_following"],
        long_instruments=["SPY"],  # Buy quality at discount
        short_instruments=["VXX"],  # Vol will mean revert
        position_size_multiplier=0.20,
        stop_loss_tighter=0.10
    ),

    CrisisType.FLIGHT_TO_QUALITY: CrisisPlaybook(
        crisis_type=CrisisType.FLIGHT_TO_QUALITY,
        name="Flight to Quality Playbook",
        description="Risk-off, safe haven flows",
        equity_exposure_pct=0.40,
        hedge_ratio=0.50,
        cash_target_pct=0.30,
        activate_strategies=["quality_factor", "low_vol_factor", "dividend_growth"],
        deactivate_strategies=["high_beta", "small_cap", "emerging"],
        long_instruments=["TLT", "GLD", "JY=F", "CHF=F"],
        short_instruments=["EEM", "IWM", "XLF"],
        position_size_multiplier=0.60,
        stop_loss_tighter=0.02
    )
}


class CrisisAlphaScanner:
    """
    Real-time crisis detection and response.

    Scans multiple indicators for early warning signs.
    When crisis detected, activates appropriate playbook.
    """

    def __init__(self):
        """Initialize the scanner."""
        self.indicators: Dict[str, CrisisIndicator] = {}
        self.alert_history: List[Tuple[datetime, AlertLevel, str]] = []
        self.active_playbooks: List[CrisisPlaybook] = []

        # Current state
        self.current_alert_level = AlertLevel.GREEN
        self.crisis_probability = 0.0

        # Historical indicator data for z-score calculation
        self.indicator_history: Dict[str, List[float]] = {}

        self._lock = threading.Lock()

        logger.info(
            "[CRISIS] Crisis Alpha Scanner initialized - "
            "WATCHING FOR EARLY WARNINGS"
        )

    def update_indicator(
        self,
        name: str,
        value: float,
        threshold_yellow: float,
        threshold_orange: float,
        threshold_red: float
    ):
        """Update a crisis indicator."""
        with self._lock:
            # Store history
            if name not in self.indicator_history:
                self.indicator_history[name] = []

            self.indicator_history[name].append(value)

            # Keep last 252 values (1 year daily)
            if len(self.indicator_history[name]) > 252:
                self.indicator_history[name] = self.indicator_history[name][-252:]

            # Calculate statistics
            history = self.indicator_history[name]
            hist_mean = np.mean(history)
            hist_std = np.std(history) if len(history) > 1 else 1
            z_score = (value - hist_mean) / hist_std if hist_std > 0 else 0

            # Determine alert level
            if value >= threshold_red:
                level = AlertLevel.RED
            elif value >= threshold_orange:
                level = AlertLevel.ORANGE
            elif value >= threshold_yellow:
                level = AlertLevel.YELLOW
            else:
                level = AlertLevel.GREEN

            # Calculate trend
            trend_1h = 0
            trend_24h = 0
            if len(history) > 1:
                trend_24h = (value - history[-2]) / history[-2] if history[-2] != 0 else 0

            self.indicators[name] = CrisisIndicator(
                name=name,
                value=value,
                threshold_yellow=threshold_yellow,
                threshold_orange=threshold_orange,
                threshold_red=threshold_red,
                historical_mean=hist_mean,
                historical_std=hist_std,
                z_score=z_score,
                alert_level=level,
                trend_1h=trend_1h,
                trend_24h=trend_24h,
                timestamp=datetime.utcnow()
            )

            # Log significant changes
            if level in [AlertLevel.RED, AlertLevel.ORANGE]:
                logger.warning(
                    f"[CRISIS] {level.value} ALERT: {name} = {value:.2f} | "
                    f"Z-score: {z_score:.2f}"
                )

    def scan_market(
        self,
        vix: Optional[float] = None,
        vix3m: Optional[float] = None,
        hy_spread: Optional[float] = None,
        ig_spread: Optional[float] = None,
        fx_vol: Optional[float] = None,
        sp500_1d_return: Optional[float] = None,
        volume_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Scan market conditions for crisis signatures.

        Args:
            vix: VIX index
            vix3m: 3-month VIX
            hy_spread: High yield spread
            ig_spread: Investment grade spread
            fx_vol: FX volatility index (e.g., CVIX)
            sp500_1d_return: S&P 500 1-day return
            volume_ratio: Volume vs average
        """
        detected_crises = []

        # 1. VIX Analysis
        if vix is not None:
            self.update_indicator("VIX", vix, 20, 30, 40)

            if vix >= 35:
                detected_crises.append(CrisisType.VOLATILITY_SPIKE)

        # 2. VIX Term Structure (Contango vs Backwardation)
        if vix is not None and vix3m is not None:
            term_spread = vix - vix3m
            self.update_indicator("VIX_TERM_SPREAD", term_spread, 2, 5, 10)

            if term_spread > 5:  # Severe backwardation
                detected_crises.append(CrisisType.VOLATILITY_SPIKE)

        # 3. Credit Spreads
        if hy_spread is not None:
            self.update_indicator("HY_SPREAD", hy_spread, 4.0, 6.0, 8.0)

            if hy_spread >= 6.0:
                detected_crises.append(CrisisType.CREDIT_CRUNCH)

        if hy_spread is not None and ig_spread is not None:
            credit_diff = hy_spread - ig_spread
            self.update_indicator("HY_IG_DIFF", credit_diff, 3.0, 4.5, 6.0)

        # 4. FX Volatility
        if fx_vol is not None:
            self.update_indicator("FX_VOL", fx_vol, 10, 15, 20)

            if fx_vol >= 18:
                detected_crises.append(CrisisType.CURRENCY_CRISIS)

        # 5. Flash Crash Detection
        if sp500_1d_return is not None:
            self.update_indicator("SP500_1D", sp500_1d_return * 100, -2, -4, -7)

            if sp500_1d_return <= -0.05:  # -5% or worse
                detected_crises.append(CrisisType.FLASH_CRASH)

        # 6. Liquidity
        if volume_ratio is not None and sp500_1d_return is not None:
            # High volume + big down move = panic selling
            if volume_ratio > 2.0 and sp500_1d_return < -0.02:
                detected_crises.append(CrisisType.LIQUIDITY_DROUGHT)

        # Update overall alert level
        self._update_alert_level()

        # Calculate crisis probability
        self._calculate_crisis_probability()

        # Activate playbooks
        playbooks = self._select_playbooks(detected_crises)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_level": self.current_alert_level.value,
            "crisis_probability": self.crisis_probability,
            "detected_crises": [c.value for c in detected_crises],
            "active_playbooks": [p.name for p in playbooks],
            "indicators": {
                name: {
                    "value": ind.value,
                    "z_score": ind.z_score,
                    "alert": ind.alert_level.value
                }
                for name, ind in self.indicators.items()
            }
        }

    def _update_alert_level(self):
        """Update overall alert level from indicators."""
        with self._lock:
            if not self.indicators:
                self.current_alert_level = AlertLevel.GREEN
                return

            # Count alerts at each level
            red_count = sum(1 for i in self.indicators.values() if i.alert_level == AlertLevel.RED)
            orange_count = sum(1 for i in self.indicators.values() if i.alert_level == AlertLevel.ORANGE)

            if red_count >= 1:
                self.current_alert_level = AlertLevel.RED
            elif orange_count >= 2 or red_count >= 1:
                self.current_alert_level = AlertLevel.ORANGE
            elif orange_count >= 1:
                self.current_alert_level = AlertLevel.YELLOW
            else:
                self.current_alert_level = AlertLevel.GREEN

            # Record history
            self.alert_history.append((
                datetime.utcnow(),
                self.current_alert_level,
                "Indicators updated"
            ))

    def _calculate_crisis_probability(self):
        """Calculate probability of crisis from indicators."""
        with self._lock:
            if not self.indicators:
                self.crisis_probability = 0.05
                return

            # Use z-scores to estimate probability
            z_scores = [ind.z_score for ind in self.indicators.values()]
            max_z = max(z_scores) if z_scores else 0
            avg_z = np.mean(z_scores) if z_scores else 0

            # Simple probability model
            # Higher z-scores = higher crisis probability
            prob = 1 / (1 + np.exp(-0.5 * (avg_z - 1.5)))  # Sigmoid centered at z=1.5
            prob = min(0.95, max(0.05, prob))  # Bound between 5% and 95%

            self.crisis_probability = prob

    def _select_playbooks(
        self,
        detected_crises: List[CrisisType]
    ) -> List[CrisisPlaybook]:
        """Select appropriate playbooks for detected crises."""
        playbooks = []

        for crisis in detected_crises:
            if crisis in CRISIS_PLAYBOOKS:
                playbooks.append(CRISIS_PLAYBOOKS[crisis])

        with self._lock:
            self.active_playbooks = playbooks

        return playbooks

    def get_active_playbooks(self) -> List[CrisisPlaybook]:
        """Get currently active playbooks."""
        with self._lock:
            return self.active_playbooks.copy()

    def get_trading_adjustments(self) -> Dict[str, Any]:
        """Get trading adjustments based on active playbooks."""
        with self._lock:
            if not self.active_playbooks:
                return {
                    "equity_exposure": 1.0,
                    "position_size_mult": 1.0,
                    "activate": [],
                    "deactivate": [],
                    "longs": [],
                    "shorts": []
                }

            # Aggregate from all playbooks (use most conservative)
            equity = min(p.equity_exposure_pct for p in self.active_playbooks)
            size_mult = min(p.position_size_multiplier for p in self.active_playbooks)

            activate = set()
            deactivate = set()
            longs = set()
            shorts = set()

            for p in self.active_playbooks:
                activate.update(p.activate_strategies)
                deactivate.update(p.deactivate_strategies)
                longs.update(p.long_instruments)
                shorts.update(p.short_instruments)

            return {
                "equity_exposure": equity,
                "position_size_mult": size_mult,
                "activate": list(activate),
                "deactivate": list(deactivate),
                "longs": list(longs),
                "shorts": list(shorts),
                "alert_level": self.current_alert_level.value,
                "crisis_probability": self.crisis_probability
            }

    def simulate_from_data(
        self,
        returns: float,
        volume_ratio: float = 1.0,
        volatility: float = 0.15
    ) -> Dict[str, Any]:
        """
        Simulate crisis scan from basic market data.

        For use when granular indicator data unavailable.
        """
        # Estimate VIX from realized vol
        estimated_vix = volatility * 100 * np.sqrt(252)

        # Run scan
        return self.scan_market(
            vix=estimated_vix,
            sp500_1d_return=returns,
            volume_ratio=volume_ratio
        )


# Singleton
_scanner: Optional[CrisisAlphaScanner] = None


def get_crisis_scanner() -> CrisisAlphaScanner:
    """Get or create the Crisis Alpha Scanner."""
    global _scanner
    if _scanner is None:
        _scanner = CrisisAlphaScanner()
    return _scanner
