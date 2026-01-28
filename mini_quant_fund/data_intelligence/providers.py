import yfinance as yf
import pandas as pd
import datetime
from typing import List, Dict, Any, Optional
from mini_quant_fund.data_intelligence.contracts import ProviderAdapter
import structlog

logger = structlog.get_logger()

class YFinanceAdapter(ProviderAdapter):
    """
    Primary free provider using yfinance.
    """
    async def fetch_price_history(self, symbols: List[str], start: datetime.date, end: datetime.date) -> Dict[str, pd.DataFrame]:
        results = {}
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start, end=end, progress=False)
                if not data.empty:
                    # Robust flattening for yfinance >= 0.2
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                    results[symbol] = data
            except Exception as e:
                logger.error("YFinance fetch failed", symbol=symbol, error=str(e))
        return results

    async def fetch_corporate_actions(self, symbol: str) -> List[Dict[str, Any]]:
        ticker = yf.Ticker(symbol)
        actions = ticker.actions
        if actions.empty:
            return []
        return actions.reset_index().to_dict(orient="records")

class AlphaVantageAdapter(ProviderAdapter):
    """
    Fallback provider skeleton for AlphaVantage.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    async def fetch_price_history(self, symbols: List[str], start: datetime.date, end: datetime.date) -> Dict[str, pd.DataFrame]:
        # Implementation placeholder
        logger.info("AlphaVantage fallback requested (stub)")
        return {}

    async def fetch_corporate_actions(self, symbol: str) -> List[Dict[str, Any]]:
        return []

class PaidProviderAdapter(ProviderAdapter):
    """
    Placeholder for institutional providers (Bloomberg/Refinitiv).
    """
    async def fetch_price_history(self, symbols: List[str], start: datetime.date, end: datetime.date) -> Dict[str, pd.DataFrame]:
        logger.warning("Paid provider requested but not configured.")
        return {}

    async def fetch_corporate_actions(self, symbol: str) -> List[Dict[str, Any]]:
        return []
