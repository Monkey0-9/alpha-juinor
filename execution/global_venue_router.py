"""
PACK 5C: GLOBAL VENUE ROUTING
Intelligent ordering to best venues based on liquidity, spreads, and market impact
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GlobalVenueRouter:
    """Route orders to optimal global venues"""

    VENUE_MAP = {
        # US Equities (3 venues)
        "NYSE": {
            "country": "US",
            "region": "North America",
            "asset_class": ["EQUITIES"],
            "timezone": "US/Eastern",
            "opens": "09:30",
            "closes": "16:00",
        },
        "NASDAQ": {
            "country": "US",
            "region": "North America",
            "asset_class": ["EQUITIES", "ETF"],
            "timezone": "US/Eastern",
            "opens": "09:30",
            "closes": "16:00",
        },
        # EU Equities (3 venues)
        "LSE": {
            "country": "GB",
            "region": "Europe",
            "asset_class": ["EQUITIES"],
            "timezone": "Europe/London",
            "opens": "08:00",
            "closes": "16:30",
        },
        "EURONEXT": {
            "country": "NL",
            "region": "Europe",
            "asset_class": ["EQUITIES"],
            "timezone": "Europe/Paris",
            "opens": "09:00",
            "closes": "17:30",
        },
        "XETRA": {
            "country": "DE",
            "region": "Europe",
            "asset_class": ["EQUITIES"],
            "timezone": "Europe/Berlin",
            "opens": "08:00",
            "closes": "20:00",
        },
        # APAC Equities (3 venues)
        "JPX": {
            "country": "JP",
            "region": "APAC",
            "asset_class": ["EQUITIES"],
            "timezone": "Asia/Tokyo",
            "opens": "09:00",
            "closes": "15:00",
        },
        "HKEx": {
            "country": "HK",
            "region": "APAC",
            "asset_class": ["EQUITIES"],
            "timezone": "Asia/Hong_Kong",
            "opens": "09:30",
            "closes": "16:00",
        },
        "ASX": {
            "country": "AU",
            "region": "APAC",
            "asset_class": ["EQUITIES"],
            "timezone": "Australia/Sydney",
            "opens": "10:00",
            "closes": "16:00",
        },
        # Crypto (2 venues - global)
        "BINANCE": {
            "country": "Global",
            "region": "Global",
            "asset_class": ["CRYPTO"],
            "timezone": "UTC",
            "opens": "00:00",
            "closes": "24:00",
        },
        "COINBASE": {
            "country": "US",
            "region": "North America",
            "asset_class": ["CRYPTO"],
            "timezone": "UTC",
            "opens": "00:00",
            "closes": "24:00",
        },
    }

    # Implied liquidity scores (1-10)
    LIQUIDITY_SCORES = {
        "NYSE": 10,
        "NASDAQ": 9.5,
        "LSE": 8.5,
        "EURONEXT": 8.0,
        "XETRA": 8.0,
        "JPX": 7.5,
        "HKEx": 8.0,
        "ASX": 7.0,
        "BINANCE": 9.0,
        "COINBASE": 8.5,
    }

    # Implied spread scores (1-10, higher = better/tighter)
    SPREAD_SCORES = {
        "NYSE": 9.5,
        "NASDAQ": 9.0,
        "LSE": 8.0,
        "EURONEXT": 7.5,
        "XETRA": 8.5,
        "JPX": 7.0,
        "HKEx": 7.5,
        "ASX": 6.5,
        "BINANCE": 8.0,
        "COINBASE": 7.5,
    }

    def __init__(self):
        self.venues = self.VENUE_MAP
        self.liquidity_scores = self.LIQUIDITY_SCORES
        self.spread_scores = self.SPREAD_SCORES

    def get_primary_venues_for_symbol(self, symbol: str) -> List[str]:
        """Get candidate venues for a symbol"""
        # Symbol to venue mapping
        symbol_venue_map = {
            # US symbols
            "SPY": ["NYSE", "NASDAQ"],
            "AAPL": ["NASDAQ"],
            "TSLA": ["NASDAQ"],
            # EU symbols
            "ASML.AS": ["EURONEXT"],
            "SHELL": ["LSE"],
            # Japan
            "9984.T": ["JPX"],
            # Hong Kong
            "0005.HK": ["HKEx"],
            # Australia
            "CBA": ["ASX"],
            # Crypto
            "BTC-USD": ["BINANCE", "COINBASE"],
            "ETH-USD": ["BINANCE", "COINBASE"],
        }

        if symbol in symbol_venue_map:
            return symbol_venue_map[symbol]

        # Infer from suffix
        if ".T" in symbol:
            return ["JPX"]
        elif ".HK" in symbol:
            return ["HKEx"]
        elif ".AS" in symbol:
            return ["EURONEXT"]
        elif symbol.endswith(".AX"):
            return ["ASX"]
        elif "-USD" in symbol or "-USDT" in symbol:
            return ["BINANCE", "COINBASE"]
        else:
            # Default to US venues
            return ["NYSE", "NASDAQ"]

    def select_optimal_venue(
        self,
        symbol: str,
        order_size_usd: float,
        liquidity_weight: float = 0.3,
        spread_weight: float = 0.3,
        impact_weight: float = 0.4,
    ) -> Tuple[str, float]:
        """
        Select best venue based on weighted scoring.
        Returns (venue, score)
        """
        candidates = self.get_primary_venues_for_symbol(symbol)

        if not candidates:
            return "NYSE", 0.0

        best_venue = None
        best_score = -1

        for venue in candidates:
            liquidity_score = self.liquidity_scores.get(venue, 5) / 10.0
            spread_score = self.spread_scores.get(venue, 5) / 10.0

            # Impact score: lower order size relative to venue = better
            # Assume higher liquidity = handles larger orders better
            impact_score = min(1.0, order_size_usd / (10_000_000 * liquidity_score))

            composite_score = (
                liquidity_score * liquidity_weight
                + spread_score * spread_weight
                + impact_score * impact_weight
            )

            logger.debug(
                f"[ROUTING] {venue} score: {composite_score:.3f} "
                f"(liquidity: {liquidity_score:.2f}, spread: {spread_score:.2f}, impact: {impact_score:.2f})"
            )

            if composite_score > best_score:
                best_score = composite_score
                best_venue = venue

        logger.info(
            f"[ROUTING] Selected {best_venue} for {symbol} "
            f"(order ${order_size_usd:,.0f}, score {best_score:.3f})"
        )

        return best_venue, best_score

    def get_alternative_venues(self, symbol: str) -> List[str]:
        """Get alternative venues for a symbol"""
        primary = self.get_primary_venues_for_symbol(symbol)
        all_venues = list(self.venues.keys())
        alternatives = [v for v in all_venues if v not in primary]
        return alternatives[:3]  # Top 3 alternatives

    def estimate_market_impact(
        self, venue: str, symbol: str, order_size_usd: float
    ) -> float:
        """
        Estimate market impact in basis points.
        Based on: order_size / venue_liquidity
        """
        # Average daily liquidity estimate (stub)
        venue_liquidity_usd = {
            "NYSE": 50_000_000_000,  # $50B/day
            "NASDAQ": 40_000_000_000,  # $40B/day
            "LSE": 20_000_000_000,  # $20B/day
            "EURONEXT": 15_000_000_000,  # $15B/day
            "XETRA": 18_000_000_000,  # $18B/day
            "JPX": 10_000_000_000,  # $10B/day
            "HKEx": 12_000_000_000,  # $12B/day
            "ASX": 8_000_000_000,  # $8B/day
            "BINANCE": 30_000_000_000,  # $30B/day
            "COINBASE": 5_000_000_000,  # $5B/day
        }

        daily_liquidity = venue_liquidity_usd.get(venue, 10_000_000_000)
        participation_rate = order_size_usd / daily_liquidity

        # Rough impact formula: impact = 0.1 * (participation_rate)^1.5 * 10000 bps
        impact_bps = 0.1 * (participation_rate**1.5) * 10000
        impact_bps = min(impact_bps, 500)  # Cap at 500 bps

        return impact_bps

    def split_large_order(
        self, symbol: str, order_size_usd: float, max_impact_bps: float = 50.0
    ) -> Dict:
        """
        Split large orders across venues to limit impact.
        Returns route plan with venue assignments.
        """
        venues = self.get_primary_venues_for_symbol(symbol)
        route_plan = {"symbol": symbol, "original_size": order_size_usd, "routes": []}

        remaining = order_size_usd

        for venue in venues:
            impact = self.estimate_market_impact(venue, symbol, remaining)

            if impact > max_impact_bps:
                # Reduce allocation to this venue
                max_size = order_size_usd * (max_impact_bps / 100.0) / (impact / 100.0)
                allocation = min(max_size, remaining)
            else:
                allocation = remaining

            if allocation > 0:
                route_plan["routes"].append(
                    {
                        "venue": venue,
                        "size_usd": allocation,
                        "estimated_impact_bps": self.estimate_market_impact(
                            venue, symbol, allocation
                        ),
                    }
                )

            remaining -= allocation

            if remaining <= 0:
                break

        if remaining > 0:
            logger.warning(
                f"[ROUTING] Could not fully route {symbol}: "
                f"${remaining:,.0f} unrouted due to impact limits"
            )

        return route_plan

    def get_venue_comparison(self, symbol: str, order_size_usd: float) -> Dict:
        """Get comparison of all candidate venues"""
        venues = self.get_primary_venues_for_symbol(symbol)

        comparison = {"symbol": symbol, "order_size": order_size_usd, "venues": {}}

        for venue in venues:
            liquidity_score = self.liquidity_scores.get(venue, 5)
            spread_score = self.spread_scores.get(venue, 5)
            impact = self.estimate_market_impact(venue, symbol, order_size_usd)

            comparison["venues"][venue] = {
                "liquidity_score": liquidity_score,
                "spread_score": spread_score,
                "estimated_impact_bps": impact,
                "recommended": venue == venues[0],  # First is best
            }

        return comparison


if __name__ == "__main__":
    import json

    router = GlobalVenueRouter()

    # Test routing
    venue, score = router.select_optimal_venue("AAPL", 1_000_000)
    print(f"Selected venue: {venue} (score: {score:.3f})")

    print("\nVenue Comparison:")
    print(json.dumps(router.get_venue_comparison("AAPL", 5_000_000), indent=2))

    print("\nLarge Order Split:")
    print(json.dumps(router.split_large_order("SPY", 100_000_000), indent=2))
