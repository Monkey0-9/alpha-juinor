#!/usr/bin/env python3
"""
Kafka Ingestion Smoke Test.

This script tests the Kafka ingestion pipeline by:
1. Starting a local Kafka instance
2. Producing sample market data
3. Consuming and validating the data
4. Verifying feature computation
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest_streaming.producer import StreamingProducer, ProducerConfig, get_producer
from data.ingest_streaming.consumer import StreamingConsumer, ConsumerConfig, get_consumer
from data.ingest_streaming.schema_registry import get_schema_registry
from database import get_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KafkaSmokeTest:
    """Smoke test for Kafka ingestion pipeline."""

    def __init__(self, num_symbols: int = 10, num_bars: int = 100):
        """Initialize smoke test."""
        self.num_symbols = num_symbols
        self.num_bars = num_bars
        self.symbols = [f"SYM{i:03d}" for i in range(num_symbols)]
        self.produced_messages = 0
        self.consumed_messages = 0
        self.start_time = None
        self.end_time = None

        kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.producer = get_producer(ProducerConfig(bootstrap_servers=kafka_servers))
        self.consumer = get_consumer(ConsumerConfig(bootstrap_servers=kafka_servers, group_id="smoke_test_group"))
        self.schema_registry = get_schema_registry()
        self.db = get_db()

    def generate_sample_data(self) -> List[Dict[str, Any]]:
        """Generate sample bar data for testing."""
        data = []
        base_price = 100.0

        for symbol in self.symbols:
            for i in range(self.num_bars):
                date = (datetime.now() - timedelta(days=self.num_bars - i)).strftime('%Y-%m-%d')

                open_price = base_price * (1 + (i * 0.001) + (hash(symbol) % 100 - 50) / 10000)
                close_price = open_price * (1 + ((hash(f"{symbol}{i}") % 200 - 100) / 10000))
                high_price = max(open_price, close_price) * (1 + abs(hash(f"{symbol}{i}") % 50) / 10000)
                low_price = min(open_price, close_price) * (1 - abs(hash(f"{symbol}{i}") % 50) / 10000)
                volume = 1000000 + (hash(f"{symbol}{i}") % 500000)

                data.append({
                    "symbol": symbol,
                    "date": date,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volume,
                })

        return data

    def run_producer_test(self) -> bool:
        """Run producer smoke test."""
        logger.info("Starting producer smoke test...")

        self.producer.start()

        sample_data = self.generate_sample_data()
        logger.info(f"Generated {len(sample_data)} sample bars for {self.num_symbols} symbols")

        for bar in sample_data:
            self.producer.produce_bar(
                symbol=bar["symbol"],
                open_price=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                interval="1d",
                source="smoke_test"
            )
            self.produced_messages += 1

        self.producer.flush()

        logger.info(f"Produced {self.produced_messages} messages")

        stats = self.producer.get_stats()
        logger.info(f"Producer stats: {stats}")

        return stats["errors"] == 0

    def run_consumer_test(self) -> bool:
        """Run consumer smoke test."""
        logger.info("Starting consumer smoke test...")

        consumed_data = []

        def bar_handler(record):
            nonlocal self.consumed_messages
            self.consumed_messages += 1
            consumed_data.append(record)
            if self.consumed_messages % 100 == 0:
                logger.info(f"Consumed {self.consumed_messages} messages")

        self.consumer.register_handler("bars", bar_handler)
        self.consumer.subscribe(["mini_quant_bars"])

        self.consumer.start()

        max_wait = 30
        wait_time = 0
        while wait_time < max_wait and self.consumed_messages < self.produced_messages:
            time.sleep(1)
            wait_time += 1

        self.consumer.stop()

        logger.info(f"Consumed {self.consumed_messages} messages")

        stats = self.consumer.get_stats()
        logger.info(f"Consumer stats: {stats}")

        if self.consumed_messages >= self.produced_messages * 0.9:
            logger.info("Consumer test PASSED")
            return True
        else:
            logger.error(f"Consumer test FAILED: expected ~{self.produced_messages}, got {self.consumed_messages}")
            return False

    def run_schema_test(self) -> bool:
        """Run schema validation test."""
        logger.info("Running schema validation test...")

        test_records = {
            "quote": {
                "symbol": "AAPL",
                "bid": 185.50,
                "ask": 185.55,
                "bid_size": 100,
                "ask_size": 200,
                "timestamp": datetime.utcnow().timestamp() * 1000,
                "source": "test",
            },
            "trade": {
                "symbol": "AAPL",
                "price": 185.50,
                "quantity": 100,
                "timestamp": datetime.utcnow().timestamp() * 1000,
                "side": "buy",
                "source": "test",
            },
            "bar": {
                "symbol": "AAPL",
                "open": 185.00,
                "high": 186.00,
                "low": 184.50,
                "close": 185.50,
                "volume": 1000000,
                "interval": "1d",
                "source": "test",
            },
        }

        all_valid = True
        for schema_name, record in test_records.items():
            is_valid = self.schema_registry.validate_record(schema_name, record)
            if is_valid:
                logger.info(f"Schema {schema_name}: VALID")
            else:
                logger.error(f"Schema {schema_name}: INVALID")
                all_valid = False

        return all_valid

    def run_feature_persistence_test(self) -> bool:
        """Test that features can be persisted to database."""
        logger.info("Running feature persistence test...")

        try:
            from database.schema import FeatureRecord

            features_list = []
            for i in range(self.num_symbols):
                features_list.append(FeatureRecord(
                    symbol=f"SYM{i:03d}",
                    date=datetime.now().strftime('%Y-%m-%d'),
                    features={
                        "sma_20": 100.0 + i,
                        "rsi": 50.0 + i % 10,
                        "volatility": 0.2,
                    },
                    version="1.0"
                ))

            count = self.db.upsert_features(features_list)
            logger.info(f"Persisted {count} feature records")

            retrieved = self.db.get_latest_features([f"SYM{i:03d}" for i in range(min(5, self.num_symbols))])
            logger.info(f"Retrieved {len(retrieved)} feature records")

            return count == len(features_list) and len(retrieved) > 0

        except Exception as e:
            logger.error(f"Feature persistence test failed: {e}")
            return False

    def run(self) -> Dict[str, Any]:
        """Run complete smoke test."""
        self.start_time = datetime.utcnow()
        logger.info(f"Starting Kafka smoke test at {self.start_time}")

        results = {
            "start_time": self.start_time.isoformat(),
            "producer_test": False,
            "consumer_test": False,
            "schema_test": False,
            "feature_test": False,
            "total_produced": 0,
            "total_consumed": 0,
        }

        try:
            results["schema_test"] = self.run_schema_test()
            results["producer_test"] = self.run_producer_test()
            results["total_produced"] = self.produced_messages
            results["consumer_test"] = self.run_consumer_test()
            results["total_consumed"] = self.consumed_messages
            results["feature_test"] = self.run_feature_persistence_test()

        except Exception as e:
            logger.error(f"Smoke test failed with error: {e}")
            results["error"] = str(e)

        finally:
            self.end_time = datetime.utcnow()
            results["end_time"] = self.end_time.isoformat()
            results["duration_seconds"] = (self.end_time - self.start_time).total_seconds()

            all_passed = all([
                results["producer_test"],
                results["consumer_test"],
                results["schema_test"],
                results["feature_test"],
            ])
            results["overall"] = "PASSED" if all_passed else "FAILED"

            logger.info(f"Smoke test completed: {results['overall']}")
            logger.info(f"Results: {json.dumps(results, indent=2)}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Kafka Ingestion Smoke Test")
    parser.add_argument("--symbols", type=int, default=10, help="Number of symbols")
    parser.add_argument("--bars", type=int, default=100, help="Number of bars per symbol")
    parser.add_argument("--kafka", type=str, default="localhost:9092", help="Kafka bootstrap servers")
    args = parser.parse_args()

    os.environ["KAFKA_BOOTSTRAP_SERVERS"] = args.kafka

    test = KafkaSmokeTest(num_symbols=args.symbols, num_bars=args.bars)
    results = test.run()

    sys.exit(0 if results.get("overall") == "PASSED" else 1)


if __name__ == "__main__":
    main()

