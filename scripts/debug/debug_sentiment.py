import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategy_factory.sentiment_engine import SentimentStrategy

try:
    s = SentimentStrategy()
    print("Successfully instantiated SentimentStrategy")
    print(f"Name: {s.name}")
except Exception as e:
    print(f"Failed to instantiate: {e}")
