import asyncio
from nexus.core.alpha import AlphaEngine


async def test_data():
    engine = AlphaEngine()
    print("Fetching SPY data...")
    df = await engine.fetch_market_data("SPY", timeframe="1Min", limit=120)
    print(f"Data received: {len(df)} rows")
    if not df.empty:
        print(df.tail(5))
        signal = engine.generate_signal(df)
        print(f"Signal: {signal}")
    else:
        print("FAILED to fetch data")


if __name__ == "__main__":
    asyncio.run(test_data())
