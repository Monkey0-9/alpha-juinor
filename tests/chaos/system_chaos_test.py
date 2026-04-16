import asyncio
import random
import logging
import time
from typing import Dict, Any
from mini_quant_fund.trading.production.live_trading_system import LiveTradingSystem

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SYSTEM_CHAOS_TEST")

class ChaosEngine:
    """
    Simulates real-world infrastructure instabilities for the Quant Platform.
    Targeting: Network Latency, API Failures, and Component Crashes.
    """
    def __init__(self, system: LiveTradingSystem):
        self.system = system
        self.is_running = False

    async def inject_network_latency(self):
        """Simulates varying network latency across broker connections."""
        logger.info("Starting Network Latency Injection Scenario...")
        while self.is_running:
            # We simulate latency by monkey-patching or wrapping the connection status
            # In a real distributed system, this would be done at the OS/Container level
            delay = random.uniform(0.5, 3.0)
            logger.debug(f"CHAOS: Injecting {delay:.2f}s latency into global state")
            # For simulation, we just wait here to represent system-wide slowness
            await asyncio.sleep(delay)
            await asyncio.sleep(random.randint(5, 15))

    async def simulate_broker_disconnects(self):
        """Randomly toggles broker connection states to test recovery logic."""
        logger.info("Starting Broker Disconnect Scenario...")
        while self.is_running:
            await asyncio.sleep(random.randint(10, 30))
            if not self.system.brokers:
                continue
            
            broker_name = random.choice(list(self.system.brokers.keys()))
            logger.warning(f"CHAOS: Component Failure - Disconnecting {broker_name}")
            self.system.brokers[broker_name].is_connected = False
            
            # Wait for system to react or for some time
            await asyncio.sleep(5)
            logger.info(f"CHAOS: Restoring connection to {broker_name}")
            self.system.brokers[broker_name].is_connected = True

    async def run_trading_load(self):
        """Generates continuous trading activity to observe system resilience."""
        logger.info("Starting Background Trading Load...")
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        while self.is_running:
            symbol = random.choice(symbols)
            side = random.choice(['buy', 'sell'])
            qty = random.randint(1, 100)
            
            logger.info(f"LOAD: Executing {side} for {qty} {symbol}")
            try:
                # Use a timeout to simulate response handling under stress
                start_time = time.time()
                result = await asyncio.wait_for(
                    self.system.execute_live_trade(symbol, side, qty),
                    timeout=5.0
                )
                latency = time.time() - start_time
                if result.get('success'):
                    logger.info(f"LOAD SUCCESS: {symbol} execution in {latency:.2f}s")
                else:
                    logger.warning(f"LOAD FAILURE: {symbol} failed: {result.get('error')} (latency: {latency:.2f}s)")
            except asyncio.TimeoutError:
                logger.error(f"LOAD TIMEOUT: {symbol} trade timed out!")
            except Exception as e:
                logger.error(f"LOAD CRITICAL ERROR: {e}")
                
            await asyncio.sleep(random.uniform(1.0, 5.0))

    async def start(self):
        self.is_running = True
        logger.info("--- SYSTEM CHAOS TEST INITIATED ---")
        
        # Initial deployment
        await self.system.deploy_live_trading_system()
        
        try:
            await asyncio.gather(
                self.inject_network_latency(),
                self.simulate_broker_disconnects(),
                self.run_trading_load()
            )
        except Exception as e:
            logger.error(f"Chaos engine encountered a critical error: {e}")
        finally:
            self.is_running = False
            logger.info("--- SYSTEM CHAOS TEST CONCLUDED ---")

if __name__ == "__main__":
    system = LiveTradingSystem()
    chaos = ChaosEngine(system)
    
    try:
        asyncio.run(chaos.start())
    except KeyboardInterrupt:
        logger.info("Test stopped by user.")
