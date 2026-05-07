import asyncio
import os
from dotenv import load_dotenv
from nexus.execution.alpaca import AlpacaClient

async def test_keys():
    load_dotenv()
    client = AlpacaClient()
    print(f"Testing keys: {os.getenv('ALPACA_API_KEY')[:5]}...")
    acc = await client.get_account()
    print(f"Account Response: {acc}")
    if acc.get("status") == "ACTIVE" and acc.get("simulated") is False:
        print("SUCCESS: Alpaca keys are valid!")
    else:
        print("FAILURE: Alpaca keys are still unauthorized.")

if __name__ == "__main__":
    asyncio.run(test_keys())
