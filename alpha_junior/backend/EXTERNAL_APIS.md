# External APIs Documentation
## Alpha Junior - Public API Integrations

This document lists all external APIs used by the Alpha Junior platform for market data, economic indicators, and financial information.

---

## Free Tier APIs (No API Key Required)

### 1. ExchangeRate-API
**URL:** https://api.exchangerate-api.com  
**Usage:** Currency conversion and exchange rates  
**Endpoints:**
- `GET /v4/latest/{base}` - Get all exchange rates for a base currency
- `GET /v4/{date}/{base}` - Historical rates for specific date

**Rate Limits:** Free tier: 1,500 requests/month  
**Pricing:** Free tier available, paid plans from $10/month  
**Implementation:** `app/services/market_data.py::get_exchange_rate()`

---

### 2. CoinGecko
**URL:** https://api.coingecko.com  
**Usage:** Cryptocurrency prices and market data  
**Endpoints:**
- `GET /api/v3/simple/price` - Current price for multiple coins
- `GET /api/v3/coins/markets` - Market data with market cap, volume
- `GET /api/v3/coins/{id}/history` - Historical data

**Rate Limits:** 10-30 calls/minute (free tier)  
**Pricing:** Free tier available, paid plans from $129/month  
**Implementation:** `app/services/market_data.py::get_crypto_price()`

---

### 3. FRED (Federal Reserve Economic Data)
**URL:** https://api.stlouisfed.org/fred  
**Usage:** Economic indicators (interest rates, inflation, unemployment)  
**Endpoints:**
- `GET /fred/series/observations` - Get time series data
- `GET /fred/series` - Series metadata

**Common Series IDs:**
- `DFF` - Federal Funds Effective Rate
- `T10Y2Y` - 10-Year Treasury Constant Maturity Minus 2-Year
- `UNRATE` - Unemployment Rate
- `CPIAUCSL` - Consumer Price Index for All Urban Consumers
- `GDP` - Gross Domestic Product

**Rate Limits:** 120 requests/minute (free with API key)  
**Pricing:** Free tier available (requires API key)  
**Implementation:** `app/services/market_data.py::get_fred_series()`

---

## APIs Requiring API Key

### 4. Alpha Vantage
**URL:** https://www.alphavantage.co  
**Usage:** Stock quotes, historical prices, fundamentals  
**Endpoints:**
- `GET /query?function=GLOBAL_QUOTE` - Real-time stock quote
- `GET /query?function=TIME_SERIES_DAILY` - Historical daily prices
- `GET /query?function=OVERVIEW` - Company fundamentals
- `GET /query?function=EARNINGS` - Earnings data
- `GET /query?function=INCOME_STATEMENT` - Financial statements

**Rate Limits:** 25 requests/day (free tier), 75 requests/minute (paid)  
**Pricing:** Free tier available, paid from $49.99/month  
**Get API Key:** https://www.alphavantage.co/support/#api-key  
**Implementation:** `app/services/market_data.py::get_stock_quote()`

---

### 5. NewsAPI
**URL:** https://newsapi.org  
**Usage:** Financial news and market sentiment  
**Endpoints:**
- `GET /v2/everything` - Search all articles
- `GET /v2/top-headlines` - Top headlines
- `GET /v2/sources` - News sources

**Rate Limits:** 100 requests/day (free developer tier)  
**Pricing:** Free tier available, paid from $449/month  
**Get API Key:** https://newsapi.org/register  
**Implementation:** `app/services/market_data.py::get_financial_news()`

---

## Premium APIs (Paid Only)

### 6. IEX Cloud (Stock Data)
**URL:** https://iexcloud.io  
**Usage:** Real-time US stock market data  
**Pricing:** From $9/month for basic plan  

---

### 7. Polygon.io
**URL:** https://polygon.io  
**Usage:** Stocks, forex, crypto market data  
**Pricing:** From $49/month  

---

### 8. Twelve Data
**URL:** https://twelvedata.com  
**Usage:** Real-time and historical stock data  
**Pricing:** From $29/month  

---

### 9. Financial Modeling Prep
**URL:** https://financialmodelingprep.com  
**Usage:** Financial statements, ratios, market data  
**Pricing:** Free tier (limited), paid from $19/month  

---

## Setup Instructions

### 1. Get Free API Keys

#### Alpha Vantage
```bash
# Visit: https://www.alphavantage.co/support/#api-key
# Fill out the form to get free API key
```

#### NewsAPI
```bash
# Visit: https://newsapi.org/register
# Create account and get API key
```

#### FRED
```bash
# Visit: https://fred.stlouisfed.org/docs/api/api_key.html
# Request API key for free tier
```

### 2. Configure Environment Variables

Add to your `.env` file:

```env
# Alpha Vantage (free tier - 25 requests/day)
ALPHA_VANTAGE_API_KEY=your_key_here

# NewsAPI (free tier - 100 requests/day)
NEWS_API_KEY=your_key_here

# FRED (free tier - requires key)
FRED_API_KEY=your_key_here

# Optional: Paid APIs for higher limits
IEX_CLOUD_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
```

### 3. Rate Limiting Strategy

The platform implements caching to respect API rate limits:

| API | Cache TTL | Max Calls/Min |
|-----|-----------|---------------|
| Alpha Vantage | 60 minutes | 1 |
| CoinGecko | 5 minutes | 6 |
| ExchangeRate | 60 minutes | 1 |
| FRED | 360 minutes | 1 |
| NewsAPI | 30 minutes | 2 |

---

## Fallback Strategy

If primary APIs fail or rate limits are exceeded:

1. **Use cached data** (stale data is better than no data)
2. **Try alternative APIs** (e.g., use IEX if Alpha Vantage fails)
3. **Return null with error message** (frontend shows "Data unavailable")
4. **Alert admin** (log error for monitoring)

---

## Celery Tasks Schedule

Market data is fetched periodically via Celery tasks:

| Task | Schedule | API Used |
|------|----------|----------|
| `update_all_fund_navs` | 4:30 PM ET daily | Alpha Vantage |
| `fetch_benchmark_data` | Every 5 minutes | Alpha Vantage |
| `fetch_crypto_data` | Every 5 minutes | CoinGecko |
| `fetch_economic_indicators` | Every hour | FRED |
| `fetch_financial_news` | Every 30 minutes | NewsAPI |

---

## Testing Without API Keys

For development without API keys:

```python
# In .env
ALPHA_VANTAGE_API_KEY=demo
NEWS_API_KEY=
FRED_API_KEY=
```

Alpha Vantage provides a `demo` key with limited functionality.
Other services will return mock data or None.

---

## Monitoring & Alerts

Monitor API usage to prevent hitting limits:

```python
# Log API calls
logger.info(f"Alpha Vantage API call: {endpoint}, Remaining: {remaining_calls}")

# Alert when approaching limit
if remaining_calls < 10:
    send_alert(f"Alpha Vantage API limit approaching: {remaining_calls} left")
```

---

## Additional Resources

### Public APIs GitHub Repository
- **URL:** https://github.com/public-apis/public-apis
- **Description:** A collective list of free APIs for use in software and web development
- **Categories:** Finance, Currency, Blockchain, News, etc.

### API Status Pages
- Alpha Vantage: https://alphavantage.co/support/#api-status
- CoinGecko: https://status.coingecko.com/
- FRED: https://fred.stlouisfed.org/docs/api/

### Community
- Alpha Vantage: https://stackoverflow.com/questions/tagged/alpha-vantage
- CoinGecko: https://discord.gg/Er7RM3Zkqg

---

## License & Attribution

Always comply with API terms of service:

1. **Alpha Vantage**: Attribution required for displayed data
2. **CoinGecko**: Link back to CoinGecko for price data
3. **FRED**: Public domain data, cite as source
4. **NewsAPI**: Attribution to original news source required

---

**Last Updated:** 2024
**Maintainer:** Alpha Junior Engineering Team
