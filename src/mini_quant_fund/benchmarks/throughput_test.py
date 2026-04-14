
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from mini_quant_fund.execution.ultimate_executor import get_ultimate_executor
from mini_quant_fund.brokers.mock_broker import MockBroker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("THROUGHPUT_BENCHMARK")

class ThroughputTester:
    def __init__(self, target_rps=1000, duration_sec=10):
        self.target_rps = target_rps
        self.duration_sec = duration_sec
        self.broker = MockBroker()
        self.executor = get_ultimate_executor(self.broker)
        self.success_count = 0
        self.error_count = 0

    async def single_request(self):
        try:
            # Simulate a trading decision and execution plan creation
            plan = self.executor.create_execution_plan(
                symbol="AAPL",
                action="BUY",
                quantity=10,
                current_price=150.0
            )
            # Execute (in MockBroker this is fast)
            result = self.executor.execute(plan)
            if not result.rejected:
                self.success_count += 1
            else:
                if self.error_count < 5:
                    logger.error(f"Execution rejected: {result.error_message}")
                self.error_count += 1
        except Exception as e:
            if self.error_count < 5:
                logger.error(f"Exception in request: {e}")
            self.error_count += 1

    async def run_benchmark(self):
        logger.info(f"Starting benchmark: Target {self.target_rps} RPS for {self.duration_sec}s")
        start_time = time.perf_counter()
        end_time = start_time + self.duration_sec
        
        while time.perf_counter() < end_time:
            batch_start = time.perf_counter()
            tasks = [self.single_request() for _ in range(self.target_rps)]
            await asyncio.gather(*tasks)
            
            # Rate limiting
            elapsed = time.perf_counter() - batch_start
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
        
        total_duration = time.perf_counter() - start_time
        total_requests = self.success_count + self.error_count
        actual_rps = total_requests / total_duration
        
        logger.info("=" * 40)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 40)
        logger.info(f"Total Requests: {total_requests}")
        logger.info(f"Successes: {self.success_count}")
        logger.info(f"Errors: {self.error_count}")
        logger.info(f"Actual RPS: {actual_rps:.2f}")
        logger.info(f"Target RPS: {self.target_rps}")
        logger.info(f"Throughput Efficiency: {(actual_rps/self.target_rps)*100:.2f}%")
        logger.info("=" * 40)

if __name__ == "__main__":
    tester = ThroughputTester(target_rps=1000, duration_sec=5)
    asyncio.run(tester.run_benchmark())
