"""
regime/controller.py

Global Regime Controller Service (Ticket 10)

Central service for regime detection and risk overrides.
Combines multiple signals (HMM, VIX, realized vol) into a single regime label.
Provides position and execution overrides based on regime.

Regimes:
- RISK_ON: Normal conditions, full capital allowed
- RISK_OFF: Elevated caution, reduced position sizes
- CRISIS: Extreme conditions, minimal positions
- LIQUIDITY_STRESS: Liquidity concerns, sliced execution only
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger("REGIME_CONTROLLER")


class RegimeLabel(str, Enum):
    """Canonical regime labels."""
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    CRISIS = "CRISIS"
    LIQUIDITY_STRESS = "LIQUIDITY_STRESS"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeOverrides:
    """Position and execution overrides for current regime."""
    max_position_pct: float      # Max single position as % of NAV
    max_leverage: float          # Max total leverage
    execution_tactic: str        # "NORMAL", "CONSERVATIVE", "SLICED", "HALT"
    new_positions_allowed: bool  # Can open new positions?
    scale_factor: float          # Multiplier for all position sizes

    @staticmethod
    def for_regime(regime: RegimeLabel) -> "RegimeOverrides":
        """Get overrides for a specific regime."""
        overrides_map = {
            RegimeLabel.RISK_ON: RegimeOverrides(
                max_position_pct=0.10,    # 10% max single position
                max_leverage=1.5,
                execution_tactic="NORMAL",
                new_positions_allowed=True,
                scale_factor=1.0
            ),
            RegimeLabel.RISK_OFF: RegimeOverrides(
                max_position_pct=0.05,    # 5% max single position
                max_leverage=1.0,
                execution_tactic="CONSERVATIVE",
                new_positions_allowed=True,
                scale_factor=0.7
            ),
            RegimeLabel.CRISIS: RegimeOverrides(
                max_position_pct=0.01,    # 1% max single position
                max_leverage=0.5,
                execution_tactic="HALT",
                new_positions_allowed=False,  # No new positions in crisis
                scale_factor=0.25
            ),
            RegimeLabel.LIQUIDITY_STRESS: RegimeOverrides(
                max_position_pct=0.02,    # 2% max single position
                max_leverage=0.75,
                execution_tactic="SLICED",  # Only sliced orders
                new_positions_allowed=True,
                scale_factor=0.5
            ),
            RegimeLabel.UNKNOWN: RegimeOverrides(
                max_position_pct=0.05,
                max_leverage=1.0,
                execution_tactic="CONSERVATIVE",
                new_positions_allowed=True,
                scale_factor=0.6
            )
        }
        return overrides_map.get(regime, overrides_map[RegimeLabel.UNKNOWN])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeState:
    """Current regime state with full context."""
    regime: RegimeLabel
    confidence: float            # [0, 1]
    explanation: str             # Human-readable rationale
    overrides: RegimeOverrides
    indicators: Dict[str, Any]   # Raw indicator values
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "overrides": self.overrides.to_dict(),
            "indicators": self.indicators,
            "timestamp": self.timestamp
        }


class RegimeController:
    """
    Global Regime Controller.

    Detects market regime using multiple signals:
    1. Realized volatility (short vs long)
    2. VIX level
    3. HMM state probabilities
    4. Trend indicators
    5. Liquidity metrics

    Outputs a single regime label with confidence and overrides.
    All components MUST consult this controller before execution.
    """

    # Thresholds
    VIX_RISK_OFF = 25.0
    VIX_CRISIS = 35.0
    VOL_RATIO_ELEVATED = 1.5
    VOL_RATIO_EXTREME = 2.5
    PANIC_PROB_THRESHOLD = 0.3

    # Hysteresis (prevent rapid switching)
    HYSTERESIS_CYCLES = 3

    def __init__(self, db_manager=None):
        """
        Initialize RegimeController.

        Args:
            db_manager: DatabaseManager instance (lazy loaded if None)
        """
        self._db = db_manager
        self._current_state: Optional[RegimeState] = None
        self._hmm_model = None
        self._regime_history: List[RegimeLabel] = []
        self._transition_count = 0

    @property
    def db(self):
        if self._db is None:
            from database.manager import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    @property
    def hmm(self):
        if self._hmm_model is None:
            from regime.markov import RegimeModel
            self._hmm_model = RegimeModel()
        return self._hmm_model

    def detect_regime(
        self,
        prices: pd.Series,
        vix: float = 20.0,
        spread_bps: float = 10.0,
        volume_ratio: float = 1.0
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            prices: Recent price series (min 60 days)
            vix: Current VIX level
            spread_bps: Average bid-ask spread in bps
            volume_ratio: Current volume / average volume

        Returns:
            RegimeState with label, confidence, and overrides
        """
        now = datetime.utcnow().isoformat() + 'Z'
        indicators = {}

        # ========== VOLATILITY ANALYSIS ==========

        if len(prices) >= 60:
            returns = prices.pct_change().dropna()
            short_vol = returns.tail(10).std() * np.sqrt(252)
            long_vol = returns.tail(60).std() * np.sqrt(252)
            vol_ratio = short_vol / max(0.01, long_vol)

            # Trend
            ma_short = prices.tail(10).mean()
            ma_long = prices.tail(60).mean()
            is_uptrend = ma_short > ma_long
            trend_strength = (ma_short - ma_long) / ma_long if ma_long > 0 else 0
        else:
            vol_ratio = 1.0
            is_uptrend = True
            trend_strength = 0.0
            short_vol = 0.20
            long_vol = 0.20

        indicators.update({
            "vol_ratio": round(vol_ratio, 3),
            "short_vol": round(short_vol, 4),
            "long_vol": round(long_vol, 4),
            "is_uptrend": is_uptrend,
            "trend_strength": round(trend_strength, 4),
            "vix": vix,
            "spread_bps": spread_bps,
            "volume_ratio": round(volume_ratio, 2)
        })

        # ========== HMM UPDATE ==========

        if len(prices) >= 20:
            returns = prices.pct_change().dropna()
            hmm_probs = self.hmm.update(returns, vix)
            panic_prob = hmm_probs.get("panic_prob", 0.0)
            bear_prob = hmm_probs.get("bear_prob", 0.0)
        else:
            panic_prob = 0.0
            bear_prob = 0.0
            hmm_probs = {"panic_prob": 0.0, "bear_prob": 0.0, "bull_prob": 1.0}

        indicators["hmm_probs"] = hmm_probs

        # ========== LIQUIDITY ANALYSIS ==========

        is_illiquid = spread_bps > 50 or volume_ratio < 0.3
        indicators["is_illiquid"] = is_illiquid

        # ========== REGIME CLASSIFICATION ==========

        # Score each regime
        scores = {
            RegimeLabel.RISK_ON: 0.0,
            RegimeLabel.RISK_OFF: 0.0,
            RegimeLabel.CRISIS: 0.0,
            RegimeLabel.LIQUIDITY_STRESS: 0.0
        }

        # RISK_ON conditions
        if vix < self.VIX_RISK_OFF and vol_ratio < self.VOL_RATIO_ELEVATED and is_uptrend:
            scores[RegimeLabel.RISK_ON] += 1.0
        if panic_prob < 0.1 and bear_prob < 0.2:
            scores[RegimeLabel.RISK_ON] += 0.5

        # RISK_OFF conditions
        if self.VIX_RISK_OFF <= vix < self.VIX_CRISIS:
            scores[RegimeLabel.RISK_OFF] += 1.0
        if self.VOL_RATIO_ELEVATED <= vol_ratio < self.VOL_RATIO_EXTREME:
            scores[RegimeLabel.RISK_OFF] += 0.5
        if not is_uptrend and vol_ratio < self.VOL_RATIO_EXTREME:
            scores[RegimeLabel.RISK_OFF] += 0.5

        # CRISIS conditions
        if vix >= self.VIX_CRISIS:
            scores[RegimeLabel.CRISIS] += 1.5
        if panic_prob >= self.PANIC_PROB_THRESHOLD:
            scores[RegimeLabel.CRISIS] += 1.0
        if vol_ratio >= self.VOL_RATIO_EXTREME:
            scores[RegimeLabel.CRISIS] += 0.5

        # LIQUIDITY_STRESS conditions
        if is_illiquid:
            scores[RegimeLabel.LIQUIDITY_STRESS] += 1.5
        if volume_ratio < 0.5 and spread_bps > 30:
            scores[RegimeLabel.LIQUIDITY_STRESS] += 0.5

        # Select highest scoring regime
        best_regime = max(scores, key=scores.get)
        max_score = scores[best_regime]

        # Calculate confidence
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = max_score / total_score
        else:
            confidence = 0.5
            best_regime = RegimeLabel.UNKNOWN

        # ========== HYSTERESIS ==========

        # Prevent rapid switching
        if self._current_state and self._current_state.regime != best_regime:
            # Check if we've been in the new regime for enough cycles
            recent_same = 0
            for r in reversed(self._regime_history[-self.HYSTERESIS_CYCLES:]):
                if r == best_regime:
                    recent_same += 1

            if recent_same < self.HYSTERESIS_CYCLES // 2:
                # Not enough evidence to switch, keep current
                if confidence < 0.7:  # Unless very confident
                    best_regime = self._current_state.regime
                    confidence = max(0.4, self._current_state.confidence - 0.1)

        self._regime_history.append(best_regime)
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-50:]

        # ========== BUILD STATE ==========

        explanation = self._build_explanation(best_regime, indicators, scores)
        overrides = RegimeOverrides.for_regime(best_regime)

        state = RegimeState(
            regime=best_regime,
            confidence=round(confidence, 3),
            explanation=explanation,
            overrides=overrides,
            indicators=indicators,
            timestamp=now
        )

        # Log if regime changed
        if self._current_state and self._current_state.regime != best_regime:
            self._transition_count += 1
            logger.warning(json.dumps({
                "event": "REGIME_TRANSITION",
                "old_regime": self._current_state.regime.value,
                "new_regime": best_regime.value,
                "confidence": confidence,
                "transition_count": self._transition_count,
                "explanation": explanation
            }))

        self._current_state = state

        # Persist to database
        self._persist_regime(state)

        return state

    def _build_explanation(
        self,
        regime: RegimeLabel,
        indicators: Dict[str, Any],
        scores: Dict[RegimeLabel, float]
    ) -> str:
        """Build human-readable explanation."""
        parts = [f"Regime: {regime.value}"]

        if regime == RegimeLabel.RISK_ON:
            parts.append(f"VIX={indicators.get('vix', 'N/A'):.1f} (low)")
            parts.append(f"Vol ratio={indicators.get('vol_ratio', 1):.2f} (normal)")
            if indicators.get('is_uptrend'):
                parts.append("Uptrend confirmed")

        elif regime == RegimeLabel.RISK_OFF:
            parts.append(f"VIX={indicators.get('vix', 'N/A'):.1f} (elevated)")
            parts.append(f"Vol ratio={indicators.get('vol_ratio', 1):.2f}")

        elif regime == RegimeLabel.CRISIS:
            parts.append(f"VIX={indicators.get('vix', 'N/A'):.1f} (EXTREME)")
            hmm = indicators.get('hmm_probs', {})
            parts.append(f"Panic prob={hmm.get('panic_prob', 0):.2f}")

        elif regime == RegimeLabel.LIQUIDITY_STRESS:
            parts.append(f"Spread={indicators.get('spread_bps', 'N/A')} bps")
            parts.append(f"Volume ratio={indicators.get('volume_ratio', 1):.2f}")

        return " | ".join(parts)

    def _persist_regime(self, state: RegimeState):
        """Persist regime state to database."""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO regime_state
                    (timestamp, regime_label, confidence, explanation, indicators_json, overrides_applied)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    state.timestamp,
                    state.regime.value,
                    state.confidence,
                    state.explanation,
                    json.dumps(state.indicators),
                    json.dumps(state.overrides.to_dict())
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist regime state: {e}")

    def get_current_state(self) -> Optional[RegimeState]:
        """Get current regime state without re-detecting."""
        return self._current_state

    def get_overrides(self) -> RegimeOverrides:
        """Get current regime overrides."""
        if self._current_state:
            return self._current_state.overrides
        return RegimeOverrides.for_regime(RegimeLabel.UNKNOWN)

    def should_halt_trading(self) -> Tuple[bool, str]:
        """
        Check if trading should be halted.

        Returns:
            Tuple of (should_halt, reason)
        """
        if not self._current_state:
            return False, ""

        overrides = self._current_state.overrides

        if overrides.execution_tactic == "HALT":
            return True, f"Regime is {self._current_state.regime.value}"

        if not overrides.new_positions_allowed:
            return True, "New positions not allowed in current regime"

        return False, ""

    def apply_position_limits(
        self,
        proposed_position_pct: float,
        symbol: str = None
    ) -> float:
        """
        Apply regime-based position limits.

        Args:
            proposed_position_pct: Proposed position as % of NAV
            symbol: Optional symbol for logging

        Returns:
            Adjusted position size
        """
        if not self._current_state:
            return proposed_position_pct

        overrides = self._current_state.overrides

        # Scale by regime factor
        scaled = proposed_position_pct * overrides.scale_factor

        # Cap at max position
        capped = min(scaled, overrides.max_position_pct)

        if capped < proposed_position_pct:
            logger.info(json.dumps({
                "event": "POSITION_CAPPED_BY_REGIME",
                "symbol": symbol,
                "proposed": round(proposed_position_pct, 4),
                "capped": round(capped, 4),
                "regime": self._current_state.regime.value
            }))

        return capped

    def get_execution_tactic(self) -> str:
        """Get recommended execution tactic."""
        if self._current_state:
            return self._current_state.overrides.execution_tactic
        return "CONSERVATIVE"

    def get_regime_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent regime history from database."""
        results = []
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM regime_state ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
                for row in cursor.fetchall():
                    results.append({
                        "timestamp": row['timestamp'],
                        "regime": row['regime_label'],
                        "confidence": row['confidence'],
                        "explanation": row['explanation']
                    })
        except Exception as e:
            logger.warning(f"Failed to get regime history: {e}")
        return results


# Singleton instance
_instance: Optional[RegimeController] = None


def get_regime_controller() -> RegimeController:
    """Get singleton RegimeController instance."""
    global _instance
    if _instance is None:
        _instance = RegimeController()
    return _instance
