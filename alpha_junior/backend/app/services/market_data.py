"""
Market Data Service
Fetches financial data from public APIs for fund NAV calculations,
benchmark comparisons, and performance tracking.

APIs Used:
- Alpha Vantage (stocks, forex, crypto)
- CoinGecko (crypto prices)
- ExchangeRate-API (currency conversion)
- FRED (economic indicators)
- NewsAPI (financial news)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class StockQuote:
    """Stock price quote"""
    symbol: str
    price: Decimal
    change: Decimal
    change_percent: Decimal
    volume: int
    timestamp: datetime


@dataclass
class BenchmarkData:
    """Benchmark index data"""
    symbol: str
    name: str
    price: Decimal
    ytd_return: Decimal
    daily_return: Decimal
    timestamp: datetime


@dataclass
class EconomicIndicator:
    """Economic data from FRED"""
    series_id: str
    name: str
    value: Decimal
    date: datetime
    frequency: str


class MarketDataService:
    """
    Service for fetching market data from public APIs
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # API Keys (should be in environment variables)
        self.alpha_vantage_key = getattr(settings, 'ALPHA_VANTAGE_API_KEY', 'demo')
        self.news_api_key = getattr(settings, 'NEWS_API_KEY', '')
        self.fred_api_key = getattr(settings, 'FRED_API_KEY', '')
        
        # Cache storage (in production, use Redis)
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
    
    async def _fetch(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Generic fetch with error handling"""
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"API fetch error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        if key in self._cache and key in self._cache_ttl:
            if datetime.utcnow() < self._cache_ttl[key]:
                return self._cache[key]
        return None
    
    def _set_cached(self, key: str, data: Any, ttl_minutes: int = 5) -> None:
        """Cache data with TTL"""
        self._cache[key] = data
        self._cache_ttl[key] = datetime.utcnow() + timedelta(minutes=ttl_minutes)
    
    # ==================== ALPHA VANTAGE (Stocks) ====================
    
    async def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get real-time stock quote from Alpha Vantage
        https://www.alphavantage.co/documentation/#latestprice
        """
        cache_key = f"av_quote_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.alpha_vantage_key,
        }
        
        data = await self._fetch(url, params)
        if not data or "Global Quote" not in data:
            return None
        
        quote = data["Global Quote"]
        
        result = StockQuote(
            symbol=symbol,
            price=Decimal(quote.get("05. price", "0")),
            change=Decimal(quote.get("09. change", "0")),
            change_percent=Decimal(quote.get("10. change percent", "0").replace("%", "")),
            volume=int(quote.get("06. volume", "0")),
            timestamp=datetime.utcnow(),
        )
        
        self._set_cached(cache_key, result, ttl_minutes=1)  # 1 min cache for quotes
        return result
    
    async def get_historical_prices(
        self,
        symbol: str,
        days: int = 30
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical daily prices
        https://www.alphavantage.co/documentation/#daily
        """
        cache_key = f"av_history_{symbol}_{days}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.alpha_vantage_key,
        }
        
        data = await self._fetch(url, params)
        if not data or "Time Series (Daily)" not in data:
            return None
        
        time_series = data["Time Series (Daily)"]
        
        prices = []
        for date_str, values in list(time_series.items())[:days]:
            prices.append({
                "date": date_str,
                "open": Decimal(values["1. open"]),
                "high": Decimal(values["2. high"]),
                "low": Decimal(values["3. low"]),
                "close": Decimal(values["4. close"]),
                "volume": int(values["5. volume"]),
            })
        
        self._set_cached(cache_key, prices, ttl_minutes=60)  # 1 hour cache
        return prices
    
    async def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company fundamentals (P/E, market cap, etc.)
        https://www.alphavantage.co/documentation/#fundamentals
        """
        cache_key = f"av_fundamentals_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.alpha_vantage_key,
        }
        
        data = await self._fetch(url, params)
        if not data or "Symbol" not in data:
            return None
        
        self._set_cached(cache_key, data, ttl_minutes=1440)  # 24 hour cache
        return data
    
    # ==================== COINGECKO (Crypto) ====================
    
    async def get_crypto_price(
        self,
        coin_id: str = "bitcoin",
        currency: str = "usd"
    ) -> Optional[Decimal]:
        """
        Get cryptocurrency price from CoinGecko
        https://api.coingecko.com/api/v3/simple/price
        """
        cache_key = f"cg_price_{coin_id}_{currency}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": currency,
            "include_24hr_change": "true",
        }
        
        data = await self._fetch(url, params)
        if not data or coin_id not in data:
            return None
        
        price = Decimal(str(data[coin_id].get(currency, 0)))
        
        self._set_cached(cache_key, price, ttl_minutes=1)
        return price
    
    async def get_crypto_market_data(
        self,
        vs_currency: str = "usd",
        per_page: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get top cryptocurrencies by market cap
        https://api.coingecko.com/api/v3/coins/markets
        """
        cache_key = f"cg_market_{vs_currency}_{per_page}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": 1,
            "sparkline": "false",
        }
        
        data = await self._fetch(url, params)
        if not data or not isinstance(data, list):
            return None
        
        self._set_cached(cache_key, data, ttl_minutes=5)
        return data
    
    # ==================== EXCHANGE RATE API ====================
    
    async def get_exchange_rate(
        self,
        base: str = "USD",
        target: str = "EUR"
    ) -> Optional[Decimal]:
        """
        Get exchange rate from ExchangeRate-API
        https://api.exchangerate-api.com/v4/latest/USD
        """
        cache_key = f"fx_{base}_{target}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url = f"https://api.exchangerate-api.com/v4/latest/{base}"
        
        data = await self._fetch(url)
        if not data or "rates" not in data:
            return None
        
        rate = Decimal(str(data["rates"].get(target, 0)))
        
        self._set_cached(cache_key, rate, ttl_minutes=60)
        return rate
    
    async def convert_currency(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str
    ) -> Optional[Decimal]:
        """Convert amount between currencies"""
        if from_currency == to_currency:
            return amount
        
        rate = await self.get_exchange_rate(from_currency, to_currency)
        if rate:
            return amount * rate
        return None
    
    # ==================== FRED (Economic Data) ====================
    
    async def get_fred_series(
        self,
        series_id: str,
        limit: int = 10
    ) -> Optional[List[EconomicIndicator]]:
        """
        Get economic data from FRED (Federal Reserve)
        https://fred.stlouisfed.org/docs/api/fred/series_observations.html
        
        Common series IDs:
        - DFF: Federal Funds Effective Rate
        - T10Y2Y: 10-Year Treasury Constant Maturity Minus 2-Year
        - UNRATE: Unemployment Rate
        - CPIAUCSL: Consumer Price Index
        - GDP: Gross Domestic Product
        """
        if not self.fred_api_key:
            return None
        
        cache_key = f"fred_{series_id}_{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        
        data = await self._fetch(url, params)
        if not data or "observations" not in data:
            return None
        
        # Get series info for name
        series_name = data.get("series", {}).get("title", series_id)
        
        indicators = []
        for obs in data["observations"][:limit]:
            if obs.get("value") != ".":  # FRED uses '.' for missing data
                indicators.append(EconomicIndicator(
                    series_id=series_id,
                    name=series_name,
                    value=Decimal(obs["value"]),
                    date=datetime.strptime(obs["date"], "%Y-%m-%d"),
                    frequency=data.get("series", {}).get("frequency", "Unknown"),
                ))
        
        self._set_cached(cache_key, indicators, ttl_minutes=360)  # 6 hour cache
        return indicators
    
    async def get_key_economic_indicators(self) -> Dict[str, Any]:
        """Fetch key economic indicators for fund analytics"""
        indicators = {}
        
        # Federal Funds Rate
        fed_funds = await self.get_fred_series("DFF", limit=1)
        if fed_funds:
            indicators["federal_funds_rate"] = {
                "value": float(fed_funds[0].value),
                "date": fed_funds[0].date.isoformat(),
            }
        
        # 10Y-2Y Spread (yield curve)
        yield_spread = await self.get_fred_series("T10Y2Y", limit=1)
        if yield_spread:
            indicators["yield_curve_spread"] = {
                "value": float(yield_spread[0].value),
                "date": yield_spread[0].date.isoformat(),
            }
        
        # Unemployment Rate
        unemployment = await self.get_fred_series("UNRATE", limit=1)
        if unemployment:
            indicators["unemployment_rate"] = {
                "value": float(unemployment[0].value),
                "date": unemployment[0].date.isoformat(),
            }
        
        # Inflation (CPI)
        cpi = await self.get_fred_series("CPIAUCSL", limit=12)  # 12 months
        if cpi and len(cpi) >= 2:
            # Calculate YoY inflation
            current = cpi[0].value
            year_ago = cpi[-1].value if len(cpi) >= 12 else cpi[-1].value
            inflation = ((current - year_ago) / year_ago) * 100
            indicators["inflation_yoy"] = {
                "value": float(inflation),
                "date": cpi[0].date.isoformat(),
            }
        
        return indicators
    
    # ==================== BENCHMARK DATA ====================
    
    async def get_benchmark_quotes(self) -> List[BenchmarkData]:
        """
        Get major benchmark index quotes
        S&P 500, NASDAQ, Dow Jones, VIX
        """
        benchmarks = []
        
        # S&P 500
        spy = await self.get_stock_quote("SPY")
        if spy:
            benchmarks.append(BenchmarkData(
                symbol="SPY",
                name="S&P 500 ETF",
                price=spy.price,
                ytd_return=Decimal("0"),  # Would need historical data
                daily_return=spy.change_percent,
                timestamp=spy.timestamp,
            ))
        
        # NASDAQ-100
        qqq = await self.get_stock_quote("QQQ")
        if qqq:
            benchmarks.append(BenchmarkData(
                symbol="QQQ",
                name="NASDAQ-100 ETF",
                price=qqq.price,
                ytd_return=Decimal("0"),
                daily_return=qqq.change_percent,
                timestamp=qqq.timestamp,
            ))
        
        # Dow Jones
        dia = await self.get_stock_quote("DIA")
        if dia:
            benchmarks.append(BenchmarkData(
                symbol="DIA",
                name="Dow Jones ETF",
                price=dia.price,
                ytd_return=Decimal("0"),
                daily_return=dia.change_percent,
                timestamp=dia.timestamp,
            ))
        
        # VIX (Volatility Index)
        vix = await self.get_stock_quote("VIX")
        if vix:
            benchmarks.append(BenchmarkData(
                symbol="VIX",
                name="VIX Volatility Index",
                price=vix.price,
                ytd_return=Decimal("0"),
                daily_return=vix.change_percent,
                timestamp=vix.timestamp,
            ))
        
        return benchmarks
    
    # ==================== NEWS API ====================
    
    async def get_financial_news(
        self,
        query: str = "finance OR investing OR stock market",
        language: str = "en",
        page_size: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get financial news from NewsAPI
        https://newsapi.org/docs/endpoints/everything
        """
        if not self.news_api_key:
            return None
        
        cache_key = f"news_{query}_{page_size}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self.news_api_key,
        }
        
        data = await self._fetch(url, params)
        if not data or data.get("status") != "ok":
            return None
        
        articles = data.get("articles", [])
        
        self._set_cached(cache_key, articles, ttl_minutes=30)
        return articles
    
    # ==================== FUND NAV CALCULATION ====================
    
    async def calculate_fund_nav(
        self,
        holdings: List[Dict[str, Any]]  # List of {symbol, quantity, asset_type}
    ) -> Optional[Decimal]:
        """
        Calculate fund NAV from holdings
        
        Args:
            holdings: List of holdings with symbol, quantity, and asset_type
                     asset_type: 'stock', 'crypto', 'cash'
        
        Returns:
            Total NAV in USD
        """
        total_value = Decimal("0")
        
        for holding in holdings:
            symbol = holding["symbol"]
            quantity = Decimal(str(holding["quantity"]))
            asset_type = holding.get("asset_type", "stock")
            
            price = None
            
            if asset_type == "stock":
                quote = await self.get_stock_quote(symbol)
                if quote:
                    price = quote.price
            
            elif asset_type == "crypto":
                # Map symbol to CoinGecko ID
                crypto_map = {
                    "BTC": "bitcoin",
                    "ETH": "ethereum",
                    "SOL": "solana",
                }
                coin_id = crypto_map.get(symbol.upper(), symbol.lower())
                price = await self.get_crypto_price(coin_id, "usd")
            
            elif asset_type == "cash":
                price = Decimal("1")
            
            if price:
                total_value += price * quantity
            else:
                logger.warning(f"Could not fetch price for {symbol}")
        
        return total_value
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Singleton instance
market_data_service = MarketDataService()
