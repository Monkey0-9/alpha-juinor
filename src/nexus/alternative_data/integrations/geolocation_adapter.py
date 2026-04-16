"""
Geolocation Mobility Data Adapter
==================================

Integrate foot traffic and mobility data from:
- SafeGraph
- Foursquare
- PlaceIQ
- Cuebiq

Use cases:
- Store visit trends
- Dwell time analysis
- Trade area analysis
- Competitive intelligence
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class MobilitySignal:
    """Signal from geolocation mobility data."""

    symbol: str
    timestamp: datetime
    metric_type: str  # 'foot_traffic', 'dwell_time', 'visit_frequency', etc.
    value: float
    change_pct: float
    confidence: float
    metadata: Dict


class GeolocationAdapter:
    """
    Adapter for geolocation and mobility data providers.

    Tracks human movement patterns for retail and consumer intelligence.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "safegraph"):
        self.api_key = api_key
        self.provider = provider
        self.base_url = self._get_base_url(provider)
        self._cache: Dict[str, MobilitySignal] = {}

    def _get_base_url(self, provider: str) -> str:
        """Get API base URL for provider."""
        urls = {
            "safegraph": "https://api.safegraph.com/v2",
            "foursquare": "https://api.foursquare.com/v3",
            "placeiq": "https://api.placeiq.com/v1",
        }
        return urls.get(provider, "")

    def get_foot_traffic(
        self, symbol: str, location_id: str, lookback_days: int = 30
    ) -> Optional[MobilitySignal]:
        """
        Get foot traffic trends for a location.

        Args:
            symbol: Stock symbol
            location_id: Location identifier
            lookback_days: Days of historical data

        Returns:
            MobilitySignal or None
        """
        cache_key = f"{symbol}:{location_id}:foot_traffic"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_foot_traffic(symbol, location_id)

        try:
            endpoint = f"{self.base_url}/foot-traffic"
            params = {
                "location_id": location_id,
                "start_date": (datetime.now() - timedelta(days=lookback_days)).isoformat(),
                "end_date": datetime.now().isoformat(),
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            visits = data.get("total_visits", 0)
            wow_change = data.get("wow_change_pct", 0)
            confidence = data.get("confidence", 0.7)

            signal = MobilitySignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="foot_traffic",
                value=visits,
                change_pct=wow_change,
                confidence=confidence,
                metadata={"location_id": location_id, "provider": self.provider},
            )

            self._cache[cache_key] = signal
            return signal

        except Exception as e:
            logger.error(f"Mobility API error: {e}")
            return None

    def get_dwell_time(
        self, symbol: str, location_id: str
    ) -> Optional[MobilitySignal]:
        """
        Get average dwell time at a location.

        Longer dwell time often correlates with higher spending.

        Args:
            symbol: Stock symbol
            location_id: Location identifier

        Returns:
            MobilitySignal or None
        """
        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_dwell_time(symbol, location_id)

        try:
            endpoint = f"{self.base_url}/dwell-time"
            params = {"location_id": location_id}
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            avg_dwell_minutes = data.get("avg_dwell_minutes", 20)
            change_pct = data.get("wow_change_pct", 0)

            signal = MobilitySignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="dwell_time",
                value=avg_dwell_minutes,
                change_pct=change_pct,
                confidence=data.get("confidence", 0.65),
                metadata={"location_id": location_id, "provider": self.provider},
            )

            return signal

        except Exception as e:
            logger.error(f"Dwell time API error: {e}")
            return None

    def get_visit_frequency(
        self, symbol: str, location_id: str
    ) -> Optional[MobilitySignal]:
        """
        Get visit frequency (returning vs. new visitors).

        High returning visitor rate indicates strong loyalty.

        Args:
            symbol: Stock symbol
            location_id: Location identifier

        Returns:
            MobilitySignal or None
        """
        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_visit_frequency(symbol, location_id)

        try:
            endpoint = f"{self.base_url}/visit-frequency"
            params = {"location_id": location_id}
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            returning_pct = data.get("returning_visitor_pct", 40)
            change_pts = data.get("change_pts", 0)

            signal = MobilitySignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="visit_frequency",
                value=returning_pct,
                change_pct=change_pts,
                confidence=data.get("confidence", 0.6),
                metadata={"location_id": location_id, "provider": self.provider},
            )

            return signal

        except Exception as e:
            logger.error(f"Visit frequency API error: {e}")
            return None

    def get_trade_area_penetration(
        self, symbol: str, location_id: str, radius_km: float = 5.0
    ) -> Optional[MobilitySignal]:
        """
        Get penetration within trade area.

        Measures what % of nearby population visits the location.

        Args:
            symbol: Stock symbol
            location_id: Location identifier
            radius_km: Trade area radius in kilometers

        Returns:
            MobilitySignal or None
        """
        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_trade_area(symbol, location_id)

        try:
            endpoint = f"{self.base_url}/trade-area"
            params = {"location_id": location_id, "radius_km": radius_km}
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            penetration_pct = data.get("penetration_pct", 5)
            change_pts = data.get("mom_change_pts", 0)

            signal = MobilitySignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="trade_area_penetration",
                value=penetration_pct,
                change_pct=change_pts,
                confidence=data.get("confidence", 0.7),
                metadata={
                    "location_id": location_id,
                    "radius_km": radius_km,
                    "provider": self.provider,
                },
            )

            return signal

        except Exception as e:
            logger.error(f"Trade area API error: {e}")
            return None

    def _simulate_foot_traffic(self, symbol: str, location_id: str) -> MobilitySignal:
        """Simulate foot traffic for testing."""
        np.random.seed(hash(symbol + location_id) % 2**32)
        visits = 1000 + np.random.uniform(-300, 500)
        wow_change = np.random.uniform(-10, 15)

        return MobilitySignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="foot_traffic",
            value=visits,
            change_pct=wow_change,
            confidence=0.70,
            metadata={"location_id": location_id, "simulated": True},
        )

    def _simulate_dwell_time(self, symbol: str, location_id: str) -> MobilitySignal:
        """Simulate dwell time for testing."""
        np.random.seed((hash(symbol + location_id) + 1) % 2**32)
        dwell_minutes = 15 + np.random.uniform(0, 30)
        change_pct = np.random.uniform(-5, 10)

        return MobilitySignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="dwell_time",
            value=dwell_minutes,
            change_pct=change_pct,
            confidence=0.65,
            metadata={"location_id": location_id, "simulated": True},
        )

    def _simulate_visit_frequency(self, symbol: str, location_id: str) -> MobilitySignal:
        """Simulate visit frequency for testing."""
        np.random.seed((hash(symbol + location_id) + 2) % 2**32)
        returning_pct = 30 + np.random.uniform(0, 30)
        change_pts = np.random.uniform(-3, 5)

        return MobilitySignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="visit_frequency",
            value=returning_pct,
            change_pct=change_pts,
            confidence=0.60,
            metadata={"location_id": location_id, "simulated": True},
        )

    def _simulate_trade_area(self, symbol: str, location_id: str) -> MobilitySignal:
        """Simulate trade area penetration for testing."""
        np.random.seed((hash(symbol + location_id) + 3) % 2**32)
        penetration_pct = 3 + np.random.uniform(0, 7)
        change_pts = np.random.uniform(-0.5, 1.0)

        return MobilitySignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="trade_area_penetration",
            value=penetration_pct,
            change_pct=change_pts,
            confidence=0.70,
            metadata={"location_id": location_id, "simulated": True},
        )

    def get_alpha_signal(self, signal: MobilitySignal) -> int:
        """
        Convert mobility signal to trading signal.

        Args:
            signal: MobilitySignal

        Returns:
            Trading signal: +1 (bullish), 0 (neutral), -1 (bearish)
        """
        if signal.confidence < 0.5:
            return 0

        # Foot traffic signal
        if signal.metric_type == "foot_traffic":
            if signal.change_pct > 7:
                return 1
            elif signal.change_pct < -7:
                return -1

        # Dwell time signal
        elif signal.metric_type == "dwell_time":
            if signal.change_pct > 5:  # Longer visits = bullish
                return 1
            elif signal.change_pct < -5:
                return -1

        # Visit frequency signal
        elif signal.metric_type == "visit_frequency":
            if signal.change_pct > 2:  # More returning visitors
                return 1
            elif signal.change_pct < -2:
                return -1

        # Trade area penetration signal
        elif signal.metric_type == "trade_area_penetration":
            if signal.change_pct > 0.5:  # Growing penetration
                return 1
            elif signal.change_pct < -0.5:
                return -1

        return 0
