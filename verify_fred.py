
import os
import logging
from dotenv import load_dotenv
from data.providers.fred import FredDataProvider

logging.basicConfig(level=logging.INFO)
load_dotenv()

def verify():
    print("Testing FRED Connectivity...")
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY not found in .env")
        return

    try:
        provider = FredDataProvider(api_key=api_key)
        # Fetch VIX as a test
        vix = provider.fetch_series("VIXCLS")
        if not vix.empty:
            print(f"Successfully fetched VIX data. Latest value: {vix.iloc[-1]}")
            
            # Fetch full regime indicators
            indicators = provider.get_macro_regime_indicators()
            if not indicators.empty:
                print(f"Macro Regime Indicators Fetched: {indicators.columns.tolist()}")
                print(f"Latest Values:\n{indicators.tail(1)}")
                print("✅ FRED Verification SUCCESSFUL")
            else:
                print("❌ Macro Regime Indicators failed (Empty)")
        else:
            print("❌ VIX fetch failed (Empty)")
            
    except Exception as e:
        print(f"Verification FAILED: {e}")

if __name__ == "__main__":
    verify()
