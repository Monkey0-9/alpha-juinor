import asyncio
import logging
from nexus.execution.alpaca import get_client
from nexus.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_alpaca_live():
    logger.info("Testing Alpaca Connectivity...")
    client = get_client()
    
    if not client.enabled:
        logger.error("Alpaca Client is NOT enabled (check API keys in .env)")
        return False
        
    try:
        account = await client.get_account()
        logger.info(f"SUCCESS: Connected to Alpaca Account!")
        logger.info(f"Account ID: {account.get('id')}")
        logger.info(f"Status: {account.get('status')}")
        logger.info(f"Equity: {account.get('equity')}")
        logger.info(f"Currency: {account.get('currency')}")
        
        clock = await client.get_clock()
        logger.info(f"Market Clock: {clock}")
        
        positions = await client.get_positions()
        logger.info(f"Current Positions: {len(positions)}")
        
        return True
    except Exception as e:
        logger.error(f"FAILURE: Could not connect to Alpaca API: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_alpaca_live())
