import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from mini_quant_fund.meta_intelligence.pm_brain import AgentOutput

class AlphaAgent(ABC):
    def __init__(self, symbol: str, seed: int = 42):
        self.symbol = symbol
        self.seed = seed
        self.model_version = "1.0.0"

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> AgentOutput:
        pass

class MomentumAgent(AlphaAgent):
    def generate_signal(self, data: pd.DataFrame) -> AgentOutput:
        if len(data) < 30:
            return AgentOutput(
                symbol=self.symbol, mu=0.0, sigma=0.1, confidence=0.0,
                model_id="mom_lgb_01", model_version=self.model_version,
                debug={"reason": "insufficient_data"}
            )

        returns = data["Close"].pct_change().dropna()
        mu = float(returns.tail(30).mean() * 252)
        sigma = float(returns.tail(30).std() * np.sqrt(252))

        return AgentOutput(
            symbol=self.symbol,
            mu=mu,
            sigma=sigma,
            confidence=0.75,
            model_id="mom_lgb_01",
            model_version=self.model_version,
            tail_params={"xi": 0.12, "beta": 0.02}, # Stub EVT params
            debug={"window": 30}
        )

class MeanReversionAgent(AlphaAgent):
    def generate_signal(self, data: pd.DataFrame) -> AgentOutput:
        if len(data) < 60:
            return AgentOutput(
                symbol=self.symbol, mu=0.0, sigma=0.1, confidence=0.0,
                model_id="mr_bayesian_01", model_version=self.model_version
            )

        last_price = data["Close"].iloc[-1]
        moving_avg = data["Close"].tail(60).mean()
        std = data["Close"].tail(60).std()

        z_score = (last_price - moving_avg) / (std + 1e-9)
        mu = float(-z_score * 0.02)

        return AgentOutput(
            symbol=self.symbol,
            mu=mu,
            sigma=0.015,
            confidence=0.65,
            model_id="mr_bayesian_01",
            model_version=self.model_version,
            tail_params={"xi": 0.08, "beta": 0.01},
            debug={"z_score": z_score}
        )
