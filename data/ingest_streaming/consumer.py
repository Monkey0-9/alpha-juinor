
import json
import logging
import argparse
import sys
import os
from kafka import KafkaConsumer
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.manager import get_db
from database.schema import IntradayPriceRecord

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureBuilderConsumer")

class FeatureBuilderConsumer:
    """
    Consumes market data ticks, builds features (TODO), and persists to DB.
    """
    def __init__(self, bootstrap_servers=['localhost:9092'], topic='market_data_ticks', group_id='feature_builder_v1'):
        self.db = get_db()
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id=group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            logger.info(f"Subscribed to {topic} at {bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect consumer: {e}")
            self.consumer = None

    def run(self):
        """Main consumption loop."""
        if not self.consumer:
            logger.error("Consumer not initialized.")
            return

        logger.info("Starting consumption loop...")
        try:
            for message in self.consumer:
                tick = message.value
                self.process_tick(tick)
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
        finally:
            self.close()

    def process_tick(self, tick: dict):
        """Process a single tick."""
        try:
            symbol = tick.get('symbol')
            price = tick.get('price')
            volume = tick.get('volume')
            ts_str = tick.get('timestamp')
            provider = tick.get('provider', 'unknown')

            if not symbol or not price:
                return

            # parse timestamp
            if ts_str:
                dt = datetime.fromisoformat(ts_str)
            else:
                dt = datetime.utcnow()

            # 1. Update Intraday Price History (Persistence)
            # For simplicity, treating every tick as a minute bar update or trade insert.
            # Real impl would aggregate ticks into bars.
            # Here we just log the tick as a point entry.

            record = IntradayPriceRecord(
                symbol=symbol,
                date=dt.date().isoformat(),
                time=dt.time().isoformat(),
                datetime=dt.isoformat(),
                open=price, high=price, low=price, close=price, # Tick data
                volume=volume,
                source_provider=provider,
                raw_hash="stream",
                pulled_at=datetime.utcnow().isoformat()
            )

            self.db.upsert_intraday_price(record)
            logger.info(f"Processed {symbol} @ {price}")

            # 2. Real-time Feature Calculation (Placeholder)
            # feature_engine.update(symbol, price)

        except Exception as e:
            logger.error(f"Error processing tick: {e}")

    def close(self):
        if self.consumer:
            self.consumer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Feature Builder Consumer")
    parser.add_argument("--servers", default="localhost:9092", help="Bootstrap servers")
    parser.add_argument("--topic", default="market_data_ticks", help="Kafka topic")
    parser.add_argument("--group", default="feature_builder_v1", help="Consumer group ID")

    args = parser.parse_args()

    consumer = FeatureBuilderConsumer(bootstrap_servers=args.servers.split(','), topic=args.topic, group_id=args.group)
    consumer.run()
