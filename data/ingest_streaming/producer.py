
import json
import logging
import argparse
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketDataProducer")

class MarketDataProducer:
    """
    Produces real-time market data ticks to Kafka.
    """
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='market_data_ticks'):
        self.topic = topic
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=5
            )
            logger.info(f"Connected to Kafka at {bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self.producer = None

    def send_tick(self, symbol: str, price: float, volume: int, provider: str = "mock"):
        """Send a single price tick."""
        if not self.producer:
            logger.warning("Producer not connected, dropping message.")
            return

        tick = {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "provider": provider,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "trade"
        }

        try:
            # key by symbol to ensure ordering within partition
            future = self.producer.send(self.topic, key=symbol, value=tick)
            # define callback for async handling
            future.add_callback(self._on_send_success).add_errback(self._on_send_error)
        except Exception as e:
            logger.error(f"Error sending tick: {e}")

    def _on_send_success(self, record_metadata):
        logger.debug(f"Message sent to {record_metadata.topic} partition {record_metadata.partition} offset {record_metadata.offset}")

    def _on_send_error(self, exc):
        logger.error(f"Message delivery failed: {exc}")

    def close(self):
        if self.producer:
            self.producer.flush()
            self.producer.close()

def simulate_feed(producer, symbols=['AAPL', 'GOOGL', 'MSFT', 'SPY']):
    """Simulate a random market feed."""
    logger.info("Starting simulation feed...")
    prices = {s: 100.0 + random.random() * 50 for s in symbols}

    try:
        while True:
            for symbol in symbols:
                # Random walk
                change = (random.random() - 0.5) * 0.5
                prices[symbol] += change
                volume = random.randint(10, 1000)

                producer.send_tick(symbol, round(prices[symbol], 2), volume)

            time.sleep(0.5) # Simulate latency
    except KeyboardInterrupt:
        logger.info("Simulation stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Market Data Producer")
    parser.add_argument("--servers", default="localhost:9092", help="Bootstrap servers")
    parser.add_argument("--topic", default="market_data_ticks", help="Kafka topic")
    parser.add_argument("--test", action="store_true", help="Run simulation")

    args = parser.parse_args()

    producer = MarketDataProducer(bootstrap_servers=args.servers.split(','), topic=args.topic)

    if args.test:
        try:
            simulate_feed(producer)
        finally:
            producer.close()
    else:
        logger.info("Producer initialized. Import and use 'send_tick' method.")
