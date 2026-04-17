#!/usr/bin/env python3
"""
Voice NLP Intent Engine (Mock)
==============================
Converts speech-to-text strings into actionable trading intents.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class VoiceNLPEngine:
    """
    Simple keyword-based NLP engine for voice-activated trading.
    Example: "buy 100 shares of Apple" -> {intent: 'buy', symbol: 'AAPL', qty: 100}
    """
    
    TICKER_MAP = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "google": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "bitcoin": "BTC/USD"
    }

    def __init__(self):
        # Patterns
        self.buy_pattern = re.compile(r"(buy|purchase|get)\s+(\d+)\s+(shares of\s+)?([a-zA-Z\s]+)", re.IGNORECASE)
        self.sell_pattern = re.compile(r"(sell|dump|liquidate)\s+(\d+)\s+(shares of\s+)?([a-zA-Z\s]+)", re.IGNORECASE)
        self.status_pattern = re.compile(r"(status|portfolio|balance|how am i doing)", re.IGNORECASE)

    def parse_intent(self, text: str) -> Dict[str, Any]:
        """
        Parses raw text into a structured intent.
        
        Args:
            text: The transcribed speech text.
            
        Returns:
            Dict containing intent, symbol, quantity, etc.
        """
        text = text.lower().strip()
        logger.info(f"Parsing voice input: '{text}'")

        # 1. Check Buy Intent
        buy_match = self.buy_pattern.search(text)
        if buy_match:
            qty = int(buy_match.group(2))
            raw_symbol = buy_match.group(4).strip()
            symbol = self._resolve_symbol(raw_symbol)
            return {
                "intent": "TRADE",
                "action": "BUY",
                "symbol": symbol,
                "quantity": qty,
                "confidence": 0.95
            }

        # 2. Check Sell Intent
        sell_match = self.sell_pattern.search(text)
        if sell_match:
            qty = int(sell_match.group(2))
            raw_symbol = sell_match.group(4).strip()
            symbol = self._resolve_symbol(raw_symbol)
            return {
                "intent": "TRADE",
                "action": "SELL",
                "symbol": symbol,
                "quantity": qty,
                "confidence": 0.95
            }

        # 3. Check Status Intent
        if self.status_pattern.search(text):
            return {
                "intent": "QUERY",
                "action": "GET_STATUS",
                "confidence": 0.90
            }

        return {
            "intent": "UNKNOWN",
            "text": text,
            "confidence": 0.0
        }

    def _resolve_symbol(self, raw_name: str) -> str:
        """Helper to resolve company names to tickers."""
        # Check if it's already a ticker
        if len(raw_name) <= 5 and raw_name.isalpha():
            return raw_name.upper()
        
        # Check map
        return self.TICKER_MAP.get(raw_name.lower(), raw_name.upper())

if __name__ == "__main__":
    # Test cases
    engine = VoiceNLPEngine()
    
    test_inputs = [
        "Buy 100 shares of Apple",
        "Sell 50 TSLA",
        "Show me my portfolio status",
        "What is my balance?",
        "Get 10 bitcoin"
    ]
    
    for inp in test_inputs:
        print(f"Input: {inp} -> {engine.parse_intent(inp)}")
