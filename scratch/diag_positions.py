import asyncio
import sys
import os

sys.path.insert(0, os.getcwd())

from nexus.execution.alpaca import get_client

async def main():
    client = get_client()
    positions = await client.get_positions()
    account = await client.get_account()
    print('SIMULATED', client.simulated)
    print('ACCOUNT', account)
    print('POSITIONS', positions)

if __name__ == '__main__':
    asyncio.run(main())
