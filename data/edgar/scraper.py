import requests
import time
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger("EdgarScraper")

class EdgarScraper:
    """
    Scraper for SEC EDGAR API.
    Fetches company filings (Form 4 - Insider Trading, 13F - Institutional Holdings).
    """
    def __init__(self, user_agent: str = "MiniQuantFund/1.0 (admin@miniquant.com)"):
        self.base_url = "https://data.sec.gov/submissions"
        self.headers = {"User-Agent": user_agent}
        self.rate_limit_delay = 0.1 # Max 10 requests/sec allowed by SEC

    def _get_cik(self, ticker: str) -> str:
        """
        Convert ticker to CIK (Central Index Key).
        For now, this is a stub or simple lookup.
        In prod, use the official company_tickers.json from SEC.
        """
        # Load CIK map if available (omitted for brevity, returning dummy or looking up online)
        # For simplicity, we might just assume we have a map or pass CIK directly.
        # Here we'll just try to fetch the company metadata which usually requires CIK.
        # Let's assume input IS CIK or we skip mapping logic for this MVP step.
        return ticker # Placeholder

    def get_company_submissions(self, cik: str) -> dict:
        """
        Get recent submissions for a company (CIK).
        CIK must be 10 digits (padded with zeros).
        """
        cik_padded = str(cik).zfill(10)
        url = f"{self.base_url}/CIK{cik_padded}.json"

        try:
            time.sleep(self.rate_limit_delay)
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(f"Failed to fetch submissions for {cik}: {resp.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching SEC data: {e}")
            return {}

    def fetch_insider_transactions(self, cik: str, lookback_days: int = 30) -> list:
        """
        Filter submissions for '4' (Statement of Changes in Beneficial Ownership).
        """
        data = self.get_company_submissions(cik)
        filings = data.get("filings", {}).get("recent", {})

        results = []
        if not filings:
            return results

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accession_nums = filings.get("accessionNumber", [])

        # Simple iterate
        today = datetime.now()
        for i, form in enumerate(forms):
            if form == "4":
                f_date_str = dates[i]
                f_date = datetime.strptime(f_date_str, "%Y-%m-%d")
                days_diff = (today - f_date).days

                if days_diff <= lookback_days:
                    results.append({
                        "date": f_date_str,
                        "accession": accession_nums[i],
                        "form": form,
                        "cik": cik
                    })

        return results

if __name__ == "__main__":
    # Test with Apple CIK (0000320193)
    scraper = EdgarScraper()
    # Note: This requires valid CIK. Apple is 320193.
    print("Testing EDGAR Scraper with Apple (CIK 320193)...")
    txs = scraper.fetch_insider_transactions("320193", lookback_days=90)
    print(f"Found {len(txs)} Form 4 filings in last 90 days.")
    for tx in txs[:3]:
        print(tx)
