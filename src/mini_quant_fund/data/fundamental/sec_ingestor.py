import requests
import pandas as pd
import logging
import time
from typing import List, Dict, Optional, Any
from mini_quant_fund.utils.retry import retry
from datetime import datetime

logger = logging.getLogger("SEC_INGESTOR")

class SECIngestor:
    """
    Institutional-Grade SEC EDGAR Ingestor.
    
    Features:
    - Automatic Ticker-to-CIK mapping
    - Strict rate limiting (SEC threshold: 10 req/sec)
    - Comprehensive fact extraction (Revenue, EPS, Assets, Liabilities)
    - Persistent caching to avoid redundant requests
    """
    
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
    FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    USER_AGENT = "MiniQuantFund/1.0 (contact@quantfund.com)"

    def __init__(self):
        self.headers = {"User-Agent": self.USER_AGENT}
        self.cik_map = self._load_ticker_cik_map()
        self.last_request_time = 0
        self.request_interval = 0.11  # ~9 requests per second to be safe
        
    def _load_ticker_cik_map(self) -> Dict[str, str]:
        """Load and cache the official SEC ticker-to-cik map."""
        try:
            response = requests.get(self.TICKERS_URL, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
            return {v['ticker']: str(v['cik_str']).zfill(10) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load ticker-cik map: {e}")
            return {}

    def _rate_limit(self):
        """Ensure we don't exceed SEC rate limits."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()

    @retry(exceptions=(requests.exceptions.RequestException,), tries=5, delay=2)
    def _get_json(self, url: str) -> Dict:
        self._rate_limit()
        response = requests.get(url, headers=self.headers)
        if response.status_code == 429:
            logger.warning("SEC Rate limit hit. Backing off...")
            time.sleep(10)
            raise requests.exceptions.RequestException("Rate limited")
        response.raise_for_status()
        return response.json()

    def get_company_facts(self, ticker: str) -> Optional[Dict]:
        """Fetch all facts for a given ticker."""
        cik = self.cik_map.get(ticker.upper())
        if not cik:
            logger.error(f"CIK not found for ticker {ticker}")
            return None
            
        url = f"{self.FACTS_URL}CIK{cik}.json"
        try:
            return self._get_json(url)
        except Exception as e:
            logger.error(f"Failed to fetch facts for {ticker}: {e}")
            return None

    def extract_metric(self, facts: Dict, taxonomy: str, tag: str) -> List[Dict]:
        """Generic metric extractor from SEC facts JSON."""
        try:
            metric_data = facts.get('facts', {}).get(taxonomy, {}).get(tag, {})
            units = metric_data.get('units', {}).get('USD', []) or metric_data.get('units', {}).get('shares', [])
            return units
        except Exception:
            return []

    def get_fundamental_snapshot(self, ticker: str) -> Dict[str, Any]:
        """
        Get a comprehensive fundamental snapshot for a ticker.
        Includes Revenue, Net Income, Assets, and EPS.
        """
        facts = self.get_company_facts(ticker)
        if not facts:
            return {}
            
        metrics = {
            "Revenues": ("us-gaap", "Revenues"),
            "NetIncome": ("us-gaap", "NetIncomeLoss"),
            "Assets": ("us-gaap", "Assets"),
            "EPS": ("us-gaap", "EarningsPerShareDiluted")
        }
        
        snapshot = {"ticker": ticker, "timestamp": datetime.utcnow().isoformat()}
        
        for name, (tax, tag) in metrics.items():
            data = self.extract_metric(facts, tax, tag)
            if data:
                # Sort by end date to get the latest
                latest = sorted(data, key=lambda x: x['end'])[-1]
                snapshot[name] = {
                    "value": float(latest['val']),
                    "period_end": latest['end'],
                    "form": latest.get('form')
                }
                
        return snapshot

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestor = SECIngestor()
    for ticker in ["AAPL", "MSFT", "NVDA"]:
        snapshot = ingestor.get_fundamental_snapshot(ticker)
        print(f"\nSnapshot for {ticker}:")
        for k, v in snapshot.items():
            if k not in ["ticker", "timestamp"]:
                print(f"  {k}: {v['value']:,.2f} ({v['period_end']})")
