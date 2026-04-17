"""
Satellite Imagery Adapter
==========================

Integrate satellite imagery analysis for:
- Retail parking lot traffic
- Industrial activity monitoring
- Oil storage tank levels
- Agriculture crop health

Supports APIs from:
- Orbital Insight
- SpaceKnow
- Descartes Labs
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
class SatelliteSignal:
    """A signal generated from satellite imagery analysis."""

    symbol: str
    timestamp: datetime
    metric_type: str  # 'parking_lot_traffic', 'industrial_activity', etc.
    value: float
    change_pct: float  # Week-over-week change
    confidence: float  # 0-1
    metadata: Dict


class SatelliteAdapter:
    """
    Adapter for satellite imagery data providers.

    Standardizes satellite imagery insights into trading signals.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "orbital_insight"):
        self.api_key = api_key
        self.provider = provider
        self.base_url = self._get_base_url(provider)
        self._cache: Dict[str, SatelliteSignal] = {}

    def _get_base_url(self, provider: str) -> str:
        """Get API base URL for provider."""
        urls = {
            "orbital_insight": "https://api.orbitalinsight.com/v1",
            "spaceknow": "https://api.spaceknow.com/v1",
            "descartes": "https://api.descarteslabs.com/v1",
        }
        return urls.get(provider, "")

    def get_parking_lot_traffic(
        self, symbol: str, location: str, lookback_days: int = 30
    ) -> Optional[SatelliteSignal]:
        """
        Get parking lot traffic analysis for a retail location.

        Args:
            symbol: Stock symbol
            location: Store location identifier
            lookback_days: Days of historical data

        Returns:
            SatelliteSignal or None if unavailable
        """
        cache_key = f"{symbol}:{location}:parking"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.api_key:
            # Return simulated data for testing
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_parking_traffic(symbol, location)

        try:
            # Real API call
            endpoint = f"{self.base_url}/parking-lot-traffic"
            params = {
                "location": location,
                "start_date": (datetime.now() - timedelta(days=lookback_days)).isoformat(),
                "end_date": datetime.now().isoformat(),
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Parse response
            traffic_index = data.get("traffic_index", 0)
            change_pct = data.get("wow_change_pct", 0)
            confidence = data.get("confidence", 0.5)

            signal = SatelliteSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="parking_lot_traffic",
                value=traffic_index,
                change_pct=change_pct,
                confidence=confidence,
                metadata={"location": location, "provider": self.provider},
            )

            self._cache[cache_key] = signal
            return signal

        except Exception as e:
            logger.error(f"Satellite API error: {e}")
            return None

    def get_industrial_activity(
        self, symbol: str, facility_id: str
    ) -> Optional[SatelliteSignal]:
        """
        Get industrial facility activity level.

        Useful for commodities, industrials, logistics companies.

        Args:
            symbol: Stock symbol
            facility_id: Facility identifier

        Returns:
            SatelliteSignal or None
        """
        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_industrial_activity(symbol, facility_id)

        try:
            endpoint = f"{self.base_url}/industrial-activity"
            params = {"facility_id": facility_id}
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            activity_score = data.get("activity_score", 50)
            change_pct = data.get("mom_change_pct", 0)

            signal = SatelliteSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="industrial_activity",
                value=activity_score,
                change_pct=change_pct,
                confidence=data.get("confidence", 0.6),
                metadata={"facility_id": facility_id, "provider": self.provider},
            )

            return signal

        except Exception as e:
            logger.error(f"Industrial activity API error: {e}")
            return None

    def get_oil_storage_levels(self, symbol: str, tank_id: str) -> Optional[SatelliteSignal]:
        """
        Get oil storage tank fill levels.

        Critical for energy trading.

        Args:
            symbol: Stock symbol (e.g., XOM, CVX)
            tank_id: Storage tank identifier

        Returns:
            SatelliteSignal or None
        """
        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_oil_storage(symbol, tank_id)

        try:
            endpoint = f"{self.base_url}/oil-storage"
            params = {"tank_id": tank_id}
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            fill_level_pct = data.get("fill_level_pct", 50)
            change_pct = data.get("wow_change_pct", 0)

            signal = SatelliteSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="oil_storage_level",
                value=fill_level_pct,
                change_pct=change_pct,
                confidence=data.get("confidence", 0.7),
                metadata={"tank_id": tank_id, "provider": self.provider},
            )

            return signal

        except Exception as e:
            logger.error(f"Oil storage API error: {e}")
            return None

    def _simulate_parking_traffic(self, symbol: str, location: str) -> SatelliteSignal:
        """Simulate parking lot traffic for testing."""
        np.random.seed(hash(symbol + location) % 2**32)
        base_traffic = 100 + np.random.uniform(-20, 20)
        change_pct = np.random.uniform(-15, 15)

        return SatelliteSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="parking_lot_traffic",
            value=base_traffic,
            change_pct=change_pct,
            confidence=0.75,
            metadata={"location": location, "simulated": True},
        )

    def _simulate_industrial_activity(
        self, symbol: str, facility_id: str
    ) -> SatelliteSignal:
        """Simulate industrial activity for testing."""
        np.random.seed(hash(symbol + facility_id) % 2**32)
        activity_score = 50 + np.random.uniform(-25, 25)
        change_pct = np.random.uniform(-10, 10)

        return SatelliteSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="industrial_activity",
            value=activity_score,
            change_pct=change_pct,
            confidence=0.65,
            metadata={"facility_id": facility_id, "simulated": True},
        )

    def _simulate_oil_storage(self, symbol: str, tank_id: str) -> SatelliteSignal:
        """Simulate oil storage levels for testing."""
        np.random.seed(hash(symbol + tank_id) % 2**32)
        fill_level_pct = 30 + np.random.uniform(0, 70)
        change_pct = np.random.uniform(-5, 5)

        return SatelliteSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="oil_storage_level",
            value=fill_level_pct,
            change_pct=change_pct,
            confidence=0.70,
            metadata={"tank_id": tank_id, "simulated": True},
        )

    def get_alpha_signal(self, signal: SatelliteSignal) -> int:
        """
        Convert satellite signal to trading signal.

        Args:
            signal: SatelliteSignal

        Returns:
            Trading signal: +1 (bullish), 0 (neutral), -1 (bearish)
        """
        # Apply confidence threshold
        if signal.confidence < 0.5:
            return 0

        # Interpret change based on metric type
        if signal.metric_type == "parking_lot_traffic":
            # More traffic = bullish for retail
            if signal.change_pct > 5:
                return 1
            elif signal.change_pct < -5:
                return -1
        elif signal.metric_type == "industrial_activity":
            # More activity = bullish for industrials
            if signal.change_pct > 3:
                return 1
            elif signal.change_pct < -3:
                return -1
        elif signal.metric_type == "oil_storage_level":
            # Rising storage = bearish for oil (oversupply)
            if signal.change_pct > 2:
                return -1
            elif signal.change_pct < -2:
                return 1

        return 0
