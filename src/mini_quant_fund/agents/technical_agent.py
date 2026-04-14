
import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class TechnicalAgent(BaseAgent):
    """
    Specialized agent for Technical Analysis.
    Analyzes price patterns, indicators, and trends.
    """
    def __init__(self, model_name: str = "gpt-4-turbo"):
        super().__init__("TechnicalAnalyzer", model_name)

    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes OHLCV data and returns a technical signal.
        """
        prices = data.get("prices") # Expected as pd.DataFrame
        
        if prices is None or prices.empty:
            return {"ticker": ticker, "signal": 0.0, "confidence": 0.0, "reason": "No price data available"}

        # 1. Qualitative Prompt Construction
        # In a real system, we'd pass statistics like RSI, SMA, MACD values
        current_price = prices['Close'].iloc[-1]
        last_5_days = prices['Close'].tail(5).tolist()
        
        prompt = f"""
        Analyze the price action for {ticker}:
        Current Price: ${current_price:.2f}
        Last 5 days close: {last_5_days}
        
        Provide a trading signal (BUY/SELL/HOLD) and brief reasoning.
        """
        
        # 2. Call LLM (Simulated with Technical Indicators)
        # In production, use ta-lib or similar
        mavg_20 = prices['Close'].rolling(20).mean().iloc[-1]
        mavg_50 = prices['Close'].rolling(50).mean().iloc[-1]
        rsi = 65 # Mock RSI
        
        signal_val = 0.0
        if current_price > mavg_20 and rsi < 70:
            signal_val = 1.0 # Bullish
        elif current_price < mavg_20 or rsi > 80:
            signal_val = -1.0 # Bearish
        
        reasoning = f"Price ${current_price:.2f} is {'above' if current_price > mavg_20 else 'below'} 20d MA. RSI is {rsi}."
        
        return {
            "ticker": ticker,
            "signal": signal_val,
            "confidence": 0.75,
            "reason": reasoning,
            "agent": self.name
        }
