# strategies/deepseek_analyst.py
import os
import requests
import json
import logging
import pandas as pd
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class DeepSeekAnalyst:
    """
    LLM-powered Macro Analyst using DeepSeek.
    Analyzes technical summaries to provide a qualitative sentiment score.
    """
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.cache: Dict[str, float] = {} # ticker_date -> score

    def analyze_market_regime(self, ticker: str, history: pd.DataFrame) -> float:
        """
        Summarizes price action and asks DeepSeek for a sentiment score (0..1).
        0 = Extreme Bearish, 0.5 = Neutral, 1 = Extreme Bullish.
        """
        if not self.api_key:
            return 0.5

        if history.empty or len(history) < 20:
            return 0.5

        # Create a compact summary for the LLM
        last_close = history['Close'].iloc[-1]
        prev_close = history['Close'].iloc[-2]
        change_1d = (last_close / prev_close) - 1
        change_20d = (last_close / history['Close'].iloc[-20]) - 1
        
        # Simple trend
        trend = "Uptrend" if change_20d > 0 else "Downtrend"
        
        # Check cache
        cache_key = f"{ticker}_{history.index[-1].strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
        You are a senior macro quantitative analyst. 
        Analyze the following technical snapshot for {ticker}:
        - Current Price: {last_close:.2f}
        - 1-Day Change: {change_1d:.2%}
        - 20-Day Change: {change_20d:.2%}
        - Short-term Trend: {trend}

        Based on these technicals and your general market knowledge, provide a 'Market Sentiment Score' between 0.0 and 1.0.
        - 1.0: Extremely Bullish
        - 0.5: Neutral
        - 0.0: Extremely Bearish

        Respond ONLY with the numerical score. No text.
        """

        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides numerical trading sentiment scores."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 10
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            score_text = result['choices'][0]['message']['content'].strip()
            score = float(score_text)
            
            # Sanity clip
            score = max(0.0, min(1.0, score))
            
            self.cache[cache_key] = score
            logger.info(f"[DeepSeek] {ticker} Sentiment: {score:.2f}")
            return score

        except Exception as e:
            logger.warning(f"DeepSeek analysis failed for {ticker}: {e}")
            return 0.5
