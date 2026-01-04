
import logging
from typing import List, Dict, Any
from .technical_agent import TechnicalAgent
from .fundamental_agent import FundamentalAgent
from .sentiment_agent import SentimentAgent
from .valuation_agent import ValuationAgent
from .risk_agent import RiskAgent
from .portfolio_agent import PortfolioAgent

logger = logging.getLogger(__name__)

class HeadOfTrading:
    """
    Orchestrator that coordinates the full AI Investment Committee.
    """
    def __init__(self):
        # 1. Analytical Agents
        self.analyzers = [
            TechnicalAgent(),
            FundamentalAgent(),
            SentimentAgent(),
            ValuationAgent()
        ]
        # 2. Control Agents
        self.risk_manager = RiskAgent()
        self.portfolio_manager = PortfolioAgent()

    def get_consensus_signal(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full committee workflow:
        Analyzers -> Consensus -> Risk Scaling -> Portfolio Sizing
        """
        logger.info(f"--- [COMMITTEE] Analyzing {ticker} ---")
        
        # Step 1: Gather analytical signals
        signals = []
        for agent in self.analyzers:
            try:
                sig = agent.analyze(ticker, data)
                signals.append(sig)
                logger.info(f"  > [{agent.name}] Signal: {sig['signal']} | {sig['reason']}")
            except Exception as e:
                logger.error(f"  ! Error in {agent.name}: {e}")

        if not signals:
            return {"ticker": ticker, "signal": 0.0, "reason": "No agent signals available"}

        # Step 2: Unweighted average for base signal
        base_signal = sum(s['signal'] for s in signals) / len(signals)
        
        # Step 3: Risk Scaling (Qualitative check)
        risk_check = self.risk_manager.analyze(ticker, data)
        scaled_signal = base_signal * risk_check['signal']
        logger.info(f"  > [RiskManager] Scaled {base_signal:.2f} to {scaled_signal:.2f} ({risk_check['reason']})")

        # Step 4: Portfolio Sizing
        p_data = {"avg_signal": scaled_signal}
        sizing = self.portfolio_manager.analyze(ticker, p_data)
        final_weight = sizing['signal']
        
        # Step 5: Final Report
        combined_reason = " | ".join([f"{s['agent']}: {s['reason']}" for s in signals])
        
        logger.info(f"--- [DECISION] {ticker}: Target Weight {final_weight:.1%} ---")
        
        return {
            "ticker": ticker,
            "signal": final_weight, # Return final target weight
            "raw_consensus": base_signal,
            "reason": f"Risk-adjusted weight {final_weight:.1%}. {combined_reason}"
        }
