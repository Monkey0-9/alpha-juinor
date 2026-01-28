import asyncio
import datetime
import numpy as np
import pandas as pd
import structlog
import yaml
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from mini_quant_fund.data_intelligence.providers import YFinanceAdapter, AlphaVantageAdapter
from mini_quant_fund.db.models import PriceHistory, DataQuality, Base
from sqlalchemy import create_engine
from mini_quant_fund.data_intelligence.contracts import DataQualityResult, ProviderAdapter
from mini_quant_fund.data_intelligence.quality_agent import QualityAgent
from mini_quant_fund.data_intelligence.bandit import ProviderBandit

logger = structlog.get_logger()

class DataIngestor:
    def __init__(self, db_url: str = "sqlite:///mini_quant.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.provider_instances: Dict[str, ProviderAdapter] = {
            "YFinance": YFinanceAdapter(),
            "AlphaVantage": AlphaVantageAdapter()
        }
        self.universe = self._load_universe()
        self.quality_agent = QualityAgent()
        self.bandit = ProviderBandit(list(self.provider_instances.keys()))

    def _load_universe(self) -> List[str]:
        try:
            with open("mini_quant_fund/config/universe.yaml", "r") as f:
                config = yaml.safe_load(f)
                return config.get("universe", {}).get("symbols", [])
        except FileNotFoundError:
            return ["AAPL", "MSFT"]

    async def run_ingestion(self) -> List[DataQualityResult]:
        logger.info("Running institutional live-ready ingestion", symbol_count=len(self.universe))
        results = []

        provider_name = self.bandit.select_provider()
        provider = self.provider_instances[provider_name]

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365 * 5)

        logger.info("Bandit selected provider", provider=provider_name)
        data_map = {}
        try:
            data_map = await provider.fetch_price_history(self.universe, start_date, end_date)
            self.bandit.update(provider_name, success=True)
        except Exception as e:
            logger.error("Primary provider failed", provider=provider_name, error=str(e))
            self.bandit.update(provider_name, success=False)
            # Fallback to YFinance as stable baseline
            provider_name = "YFinance"
            data_map = await self.provider_instances["YFinance"].fetch_price_history(self.universe, start_date, end_date)

        for symbol in self.universe:
            df = data_map.get(symbol)

            # Use QualityAgent for institutional validation
            score = self.quality_agent.compute_quality_score(df)
            errors = self.quality_agent.get_validation_report(df) if df is not None else ["NO_DATA"]

            quality = DataQualityResult(
                symbol=symbol,
                score=score,
                is_valid=score >= 0.6,
                errors=errors
            )

            if quality.is_valid and df is not None:
                self._store_data(symbol, df, provider=provider_name)

            self._store_quality(quality, provider=provider_name)
            results.append(quality)

        return results

    def _store_data(self, symbol: str, df: pd.DataFrame, provider: str):
        with Session(self.engine) as session:
            for index, row in df.iterrows():
                ph = PriceHistory(
                    symbol=symbol,
                    date=index,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    adjusted_close=float(row.get("Adj Close", row["Close"])),
                    data_source=provider
                )
                session.add(ph)
            session.commit()

    def _store_quality(self, quality: DataQualityResult, provider: str):
        with Session(self.engine) as session:
            dq = DataQuality(
                symbol=quality.symbol,
                data_quality_score=quality.score,
                validation_errors=quality.errors,
                provider_used=provider
            )
            session.add(dq)
            session.commit()
