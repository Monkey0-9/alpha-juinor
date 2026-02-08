"""
Alternative Data Integrations
==============================

Framework for integrating premium alternative data sources.

Supported providers:
- Satellite imagery (Orbital Insight, SpaceKnow)
- Credit card transactions (YipitData, Second Measure)
- Geolocation mobility (SafeGraph, Foursquare)

Each adapter provides standardized signals that can be consumed
by the alpha engine.
"""

from alternative_data.integrations.credit_card_adapter import CreditCardAdapter
from alternative_data.integrations.geolocation_adapter import GeolocationAdapter
from alternative_data.integrations.satellite_adapter import SatelliteAdapter

__all__ = [
    "SatelliteAdapter",
    "CreditCardAdapter",
    "Geolocation Adapter",
]
