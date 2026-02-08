"""
Credit Card Transaction Data Adapter
=====================================

Integrate consumer spending data from credit card transactions.

Use cases:
- Revenue nowcasting
- Consumer spending trends
- Category-level insights
- Geographic penetration

Supports APIs from:
- YipitData
- Second Measure
- Affinity Solutions
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
class TransactionSignal:
    """Signal from credit card transaction data."""

    symbol: str
    timestamp: datetime
    metric_type: str  # 'revenue_growth', 'transaction_volume', 'aov', etc.
    value: float
    change_pct: float  # YoY or MoM change
    confidence: float
    metadata: Dict


class CreditCardAdapter:
    """
    Adapter for credit card transaction data providers.

    Provides early revenue signals ahead of earnings.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "yipitdata"):
        self.api_key = api_key
        self.provider = provider
        self.base_url = self._get_base_url(provider)
        self._cache: Dict[str, TransactionSignal] = {}

    def _get_base_url(self, provider: str) -> str:
        """Get API base URL for provider."""
        urls = {
            "yipitdata": "https://api.yipitdata.com/v1",
            "second_measure": "https://api.secondmeasure.com/v1",
            "affinity": "https://api.affinitysolutions.com/v1",
        }
        return urls.get(provider, "")

    def get_revenue_growth(
        self, symbol: str, lookback_days: int = 30
    ) -> Optional[TransactionSignal]:
        """
        Get revenue growth signal from transaction data.

        Args:
            symbol: Stock symbol
            lookback_days: Days of historical comparison

        Returns:
            TransactionSignal or None
        """
        cache_key = f"{symbol}:revenue_growth"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_revenue_growth(symbol)

        try:
            endpoint = f"{self.base_url}/revenue"
            params = {
                "symbol": symbol,
                "start_date": (datetime.now() - timedelta(days=lookback_days)).isoformat(),
                "end_date": datetime.now().isoformat(),
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Parse response
            revenue_index = data.get("revenue_index", 100)
            yoy_growth = data.get("yoy_growth_pct", 0)
            confidence = data.get("confidence", 0.7)

            signal = TransactionSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="revenue_growth",
                value=revenue_index,
                change_pct=yoy_growth,
                confidence=confidence,
                metadata={"provider": self.provider, "lookback_days": lookback_days},
            )

            self._cache[cache_key] = signal
            return signal

        except Exception as e:
            logger.error(f"Credit card API error: {e}")
            return None

    def get_transaction_volume(self, symbol: str) -> Optional[TransactionSignal]:
        """
        Get transaction volume trends.

        Higher volume often correlates with strong consumer demand.

        Args:
            symbol: Stock symbol

        Returns:
            TransactionSignal or None
        """
        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_transaction_volume(symbol)

        try:
            endpoint = f"{self.base_url}/transactions"
            params = {"symbol": symbol}
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            volume_index = data.get("volume_index", 100)
            mom_change = data.get("mom_change_pct", 0)

            signal = TransactionSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="transaction_volume",
                value=volume_index,
                change_pct=mom_change,
                confidence=data.get("confidence", 0.65),
                metadata={"provider": self.provider},
            )

            return signal

        except Exception as e:
            logger.error(f"Transaction volume API error: {e}")
            return None

    def get_average_order_value(self, symbol: str) -> Optional[TransactionSignal]:
        """
        Get average order value (AOV) trends.

        Rising AOV can indicate premium product mix or pricing power.

        Args:
            symbol: Stock symbol

        Returns:
            TransactionSignal or None
        """
        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_aov(symbol)

        try:
            endpoint = f"{self.base_url}/aov"
            params = {"symbol": symbol}
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            aov = data.get("aov", 50)
            yoy_change = data.get("yoy_change_pct", 0)

            signal = TransactionSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="average_order_value",
                value=aov,
                change_pct=yoy_change,
                confidence=data.get("confidence", 0.6),
                metadata={"provider": self.provider},
            )

            return signal

        except Exception as e:
            logger.error(f"AOV API error: {e}")
            return None

    def get_category_share(
        self, symbol: str, category: str
    ) -> Optional[TransactionSignal]:
        """
        Get market share within a spending category.

        Args:
            symbol: Stock symbol
            category: Spending category (e.g., 'fast_food', 'electronics')

        Returns:
            TransactionSignal or None
        """
        if not self.api_key:
            logger.warning("No API key provided. Using simulated data.")
            return self._simulate_category_share(symbol, category)

        try:
            endpoint = f"{self.base_url}/market-share"
            params = {"symbol": symbol, "category": category}
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            share_pct = data.get("share_pct", 10)
            change_pts = data.get("change_pts", 0)

            signal = TransactionSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                metric_type="category_share",
                value=share_pct,
                change_pct=change_pts,
                confidence=data.get("confidence", 0.75),
                metadata={"category": category, "provider": self.provider},
            )

            return signal

        except Exception as e:
            logger.error(f"Category share API error: {e}")
            return None

    def _simulate_revenue_growth(self, symbol: str) -> TransactionSignal:
        """Simulate revenue growth for testing."""
        np.random.seed(hash(symbol) % 2**32)
        revenue_index = 100 + np.random.uniform(-10, 10)
        yoy_growth = np.random.uniform(-5, 15)

        return TransactionSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="revenue_growth",
            value=revenue_index,
            change_pct=yoy_growth,
            confidence=0.70,
            metadata={"simulated": True},
        )

    def _simulate_transaction_volume(self, symbol: str) -> TransactionSignal:
        """Simulate transaction volume for testing."""
        np.random.seed((hash(symbol) + 1) % 2**32)
        volume_index = 100 + np.random.uniform(-15, 15)
        mom_change = np.random.uniform(-8, 12)

        return TransactionSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="transaction_volume",
            value=volume_index,
            change_pct=mom_change,
            confidence=0.65,
            metadata={"simulated": True},
        )

    def _simulate_aov(self, symbol: str) -> TransactionSignal:
        """Simulate AOV for testing."""
        np.random.seed((hash(symbol) + 2) % 2**32)
        aov = 50 + np.random.uniform(-10, 20)
        yoy_change = np.random.uniform(-3, 8)

        return TransactionSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="average_order_value",
            value=aov,
            change_pct=yoy_change,
            confidence=0.60,
            metadata={"simulated": True},
        )

    def _simulate_category_share(self, symbol: str, category: str) -> TransactionSignal:
        """Simulate category share for testing."""
        np.random.seed((hash(symbol + category)) % 2**32)
        share_pct = 5 + np.random.uniform(0, 20)
        change_pts = np.random.uniform(-2, 3)

        return TransactionSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            metric_type="category_share",
            value=share_pct,
            change_pct=change_pts,
            confidence=0.75,
            metadata={"category": category, "simulated": True},
        )

    def get_alpha_signal(self, signal: TransactionSignal) -> int:
        """
        Convert transaction signal to trading signal.

        Args:
            signal: TransactionSignal

        Returns:
            Trading signal: +1 (bullish), 0 (neutral), -1 (bearish)
        """
        if signal.confidence < 0.5:
            return 0

        # Revenue growth signal
        if signal.metric_type == "revenue_growth":
            if signal.change_pct > 10:  # Strong YoY growth
                return 1
            elif signal.change_pct < -5:  # Declining revenues
                return -1

        # Transaction volume signal
        elif signal.metric_type == "transaction_volume":
            if signal.change_pct > 5:
                return 1
            elif signal.change_pct < -5:
                return -1

        # AOV signal
        elif signal.metric_type == "average_order_value":
            if signal.change_pct > 3:  # Rising AOV = pricing power
                return 1
            elif signal.change_pct < -3:  # Falling AOV = weakness
                return -1

        # Category share signal
        elif signal.metric_type == "category_share":
            if signal.change_pct > 1:  # Gaining share
                return 1
            elif signal.change_pct < -1:  # Losing share
                return -1

        return 0
