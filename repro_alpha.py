
import pandas as pd
from mini_quant_fund.alpha_families.alternative_alpha import AlternativeAlpha

df = pd.DataFrame({'Close': [1, 2, 3], 'Open': [1, 2, 3], 'High': [1, 2, 3], 'Low': [1, 2, 3], 'Volume': [1, 2, 3]})
agent = AlternativeAlpha()

print("Testing positional call...")
try:
    res = agent.generate_signal(df, None, {})
    print(f"Success: {res}")
except Exception as e:
    print(f"Failed positional: {e}")

print("\nTesting keyword call (market_data)...")
try:
    res = agent.generate_signal(market_data=df, regime_context=None, features={})
    print(f"Success: {res}")
except Exception as e:
    print(f"Failed keyword (market_data): {e}")

print("\nTesting mixed call...")
try:
    res = agent.generate_signal(df, regime_context=None, features={})
    print(f"Success: {res}")
except Exception as e:
    print(f"Failed mixed: {e}")
