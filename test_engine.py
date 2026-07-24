import asyncio
import os
import sys

# Ensure nexus is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from nexus.core.engine import NexusEngine

async def main():
    print("Initializing NexusEngine...")
    engine = NexusEngine()
    success = await engine.initialize()
    if not success:
        print("Failed to initialize engine. Alpaca backend may be unreachable.")
        # But we can still test the evaluation pipeline without connecting
    
    print("Loading test symbols and running cycle...")
    engine.symbols = ["AAPL", "MSFT", "NVDA"]
    try:
        await engine.run_cycle()
        print("NexusEngine cycle completed successfully.")
    except Exception as e:
        print(f"Engine cycle failed: {e}")
    finally:
        await engine.close()

if __name__ == "__main__":
    asyncio.run(main())
