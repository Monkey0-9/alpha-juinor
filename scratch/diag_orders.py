import asyncio
import sys
import os

sys.path.insert(0, os.getcwd())
from nexus.execution.alpaca import get_client

async def main():
    client = get_client()
    orders = await client.get_orders(status='all', limit=20)
    print('ORDERS', orders)

if __name__ == '__main__':
    asyncio.run(main())
