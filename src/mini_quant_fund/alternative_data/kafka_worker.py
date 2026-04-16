"""
Alternative Data Kafka Worker
=============================

Consumes real-time alternative data feeds (Satellite, Credit Card) 
from Kafka and pushes them to the Alpha Engine for real-time 
alpha factor recalculation.

Part of the Phase 5 Real-Time Streaming Upgrade.
"""

import json
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any

from confluent_kafka import Consumer, KafkaError, KafkaException

# Assuming relative imports or that src is in PYTHONPATH
try:
    from mini_quant_fund.data.alternative_data import get_alt_data_engine, AltDataSignal
    from mini_quant_fund.strategies.alpha import CompositeAlpha
except ImportError:
    # Fallback/Mock for standalone testing or if paths differ
    def get_alt_data_engine():
        class MockEngine:
            def process_signal(self, sig):
                print(f"Processed signal: {sig}")
        return MockEngine()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlternativeDataWorker:
    def __init__(self, bootstrap_servers: str = "localhost:9092", group_id: str = "alt-data-worker"):
        self.conf = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        }
        self.consumer = Consumer(self.conf)
        self.engine = get_alt_data_engine()
        self.running = True
        
        # Define topics
        self.topics = ["satellite-data", "credit-card-data"]

    def start(self):
        try:
            self.consumer.subscribe(self.topics)
            logger.info(f"Subscribed to topics: {self.topics}")

            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"End of partition reached {msg.topic()}/{msg.partition()}")
                    else:
                        raise KafkaException(msg.error())
                else:
                    self.process_message(msg)

        except Exception as e:
            logger.error(f"Worker crashed: {e}")
        finally:
            self.stop()

    def process_message(self, msg):
        try:
            payload = json.loads(msg.value().decode('utf-8'))
            topic = msg.topic()
            
            logger.info(f"Received message from {topic}")
            
            # Normalize signal based on source
            symbol = payload.get("symbol", "UNKNOWN")
            raw_value = payload.get("value", 0.0)
            
            # Map raw data to AltDataSignal
            sig = AltDataSignal(
                timestamp=datetime.utcnow(),
                source=topic,
                symbol=symbol,
                signal_value=float(raw_value),
                confidence=payload.get("confidence", 0.8),
                metadata=payload.get("metadata", {})
            )
            
            # Feed to engine
            # Note: In a real system, the engine would trigger a re-run of alphas
            if hasattr(self.engine, "process_signal"):
                self.engine.process_signal(sig)
            else:
                logger.info(f"Signal ingested for {symbol}: {sig.signal_value}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def stop(self):
        self.running = False
        self.consumer.close()
        logger.info("Worker stopped")

def signal_handler(sig, frame):
    logger.info("Interrupt received, shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    worker = AlternativeDataWorker()
    worker.start()
