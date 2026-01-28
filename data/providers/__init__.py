"""
Data Provider Registry.

Central access point for all data providers with intelligent selection.
"""

from typing import Dict, Type, Optional
from .base import DataProvider
from .yahoo import YahooDataProvider
from .polygon import PolygonDataProvider
from .alpha_vantage import AlphaVantageProvider
from .stooq import StooqProvider


class ProviderRegistry:
    """
    Registry for data providers with discovery and instantiation.
    """

    _registry: Dict[str, Type[DataProvider]] = {
        'yahoo': YahooDataProvider,
        'polygon': PolygonDataProvider,
        'alpha_vantage': AlphaVantageProvider,
        'stooq': StooqProvider,
    }

    @classmethod
    def register(cls, name: str, provider_class: Type[DataProvider]):
        """Register a new provider"""
        cls._registry[name.lower()] = provider_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[DataProvider]]:
        """Get provider class by name"""
        return cls._registry.get(name.lower())

    @classmethod
    def create(cls, name: str, **kwargs) -> DataProvider:
        """Create provider instance"""
        provider_class = cls.get(name)
        if provider_class is None:
            raise ValueError(f"Unknown provider: {name}")
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> Dict[str, Dict]:
        """List all registered providers with capabilities"""
        return {
            name: {
                'class': cls.__name__,
                'supports_ohlcv': cls.supports_ohlcv,
                'supports_latest_quote': cls.supports_latest_quote,
            }
            for name, cls in cls._registry.items()
        }


# Convenience function
def get_provider(name: str, **kwargs) -> DataProvider:
    """Get a provider instance by name"""
    return ProviderRegistry.create(name, **kwargs)

