"""
Kafka Ingestion Consumer Tests.

Tests for Kafka streaming pipeline consumer functionality.
"""

import os
import sys
import pytest
import json
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSchemaRegistry:
    """Test cases for schema registry."""

    def test_default_schemas_registered(self):
        """Test that default schemas are registered."""
        from data.ingest_streaming.schema_registry import SchemaRegistry

        registry = SchemaRegistry()

        assert "quote" in registry.schemas
        assert "trade" in registry.schemas
        assert "bar" in registry.schemas
        assert "fundamental" in registry.schemas
        assert "news" in registry.schemas
        assert "order" in registry.schemas
        assert "fill" in registry.schemas

    def test_get_schema(self):
        """Test getting schema by name."""
        from data.ingest_streaming.schema_registry import SchemaRegistry

        registry = SchemaRegistry()

        schema = registry.get_schema("quote")

        assert schema is not None
        assert schema["type"] == "record"
        assert schema["name"] == "Quote"
        assert "fields" in schema

    def test_validate_valid_record(self):
        """Test validating a valid record."""
        from data.ingest_streaming.schema_registry import SchemaRegistry

        registry = SchemaRegistry()

        record = {
            "symbol": "AAPL",
            "bid": 185.50,
            "ask": 185.55,
            "bid_size": 100,
            "ask_size": 200,
            "timestamp": 1704067200000,
            "source": "test",
        }

        is_valid = registry.validate_record("quote", record)

        assert is_valid is True

    def test_validate_invalid_record(self):
        """Test validating an invalid record (missing required field)."""
        from data.ingest_streaming.schema_registry import SchemaRegistry

        registry = SchemaRegistry()

        record = {
            "symbol": "AAPL",
            "ask": 185.55,
            "timestamp": 1704067200000,
            "source": "test",
        }

        is_valid = registry.validate_record("quote", record)

        assert is_valid is False

    def test_serialize_deserialize(self):
        """Test serialization and deserialization."""
        from data.ingest_streaming.schema_registry import SchemaRegistry

        registry = SchemaRegistry()

        record = {
            "symbol": "AAPL",
            "bid": 185.50,
            "ask": 185.55,
            "bid_size": 100,
            "ask_size": 200,
            "timestamp": 1704067200000,
            "source": "test",
        }

        data = registry.serialize_record("quote", record)

        assert isinstance(data, bytes)

        deserialized = registry.deserialize_record("quote", data)

        assert deserialized["symbol"] == record["symbol"]
        assert deserialized["bid"] == record["bid"]
        assert deserialized["ask"] == record["ask"]


class TestStreamingProducer:
    """Test cases for streaming producer."""

    def test_producer_initialization(self):
        """Test producer initializes correctly."""
        from data.ingest_streaming.producer import StreamingProducer, ProducerConfig

        config = ProducerConfig(bootstrap_servers="localhost:9092")
        producer = StreamingProducer(config)

        assert producer.config.bootstrap_servers == "localhost:9092"
        assert producer._running is False
        assert producer._topics_created == set()

    def test_topic_name_generation(self):
        """Test topic name generation."""
        from data.ingest_streaming.producer import StreamingProducer

        producer = StreamingProducer()

        topic = producer._get_topic_name("quotes")

        assert topic == "mini_quant_quotes"

    def test_producer_start_stop(self):
        """Test producer start and stop."""
        from data.ingest_streaming.producer import StreamingProducer

        producer = StreamingProducer()

        producer.start()
        assert producer._running is True

        producer.stop()
        assert producer._running is False

    def test_produce_bar_with_mock(self):
        """Test producing bar messages with mock producer."""
        from data.ingest_streaming.producer import StreamingProducer, MockKafkaProducer

        producer = StreamingProducer()
        producer._use_confluent = False
        producer._producer = MockKafkaProducer()
        producer._running = True

        producer.produce_bar(
            symbol="AAPL",
            open_price=185.0,
            high=186.0,
            low=184.5,
            close=185.5,
            volume=1000000,
            interval="1d",
            source="test"
        )

        messages = producer._producer.get_messages()

        assert len(messages) == 1
        assert messages[0]["topic"] == "mini_quant_bars"
        assert messages[0]["key"] == "AAPL"
        assert messages[0]["value"]["symbol"] == "AAPL"
        assert messages[0]["value"]["close"] == 185.5


class TestStreamingConsumer:
    """Test cases for streaming consumer."""

    def test_consumer_initialization(self):
        """Test consumer initializes correctly."""
        from data.ingest_streaming.consumer import StreamingConsumer, ConsumerConfig

        config = ConsumerConfig(
            bootstrap_servers="localhost:9092",
            group_id="test_group"
        )
        consumer = StreamingConsumer(config)

        assert consumer.config.bootstrap_servers == "localhost:9092"
        assert consumer.config.group_id == "test_group"
        assert consumer._running is False

    def test_consumer_subscribe(self):
        """Test consumer subscription."""
        from data.ingest_streaming.consumer import StreamingConsumer

        consumer = StreamingConsumer()
        consumer._use_confluent = False
        consumer._consumer = MagicMock()

        consumer.subscribe(["mini_quant_bars", "mini_quant_quotes"])

        assert consumer._topics == {"mini_quant_bars", "mini_quant_quotes"}

    def test_handler_registration(self):
        """Test handler registration."""
        from data.ingest_streaming.consumer import StreamingConsumer

        consumer = StreamingConsumer()

        def test_handler(record):
            pass

        consumer.register_handler("bars", test_handler)

        assert "bars" in consumer._handler.handlers
        assert consumer._handler.handlers["bars"] == test_handler

    def test_consumer_with_mock_messages(self):
        """Test consuming messages with mock consumer."""
        from data.ingest_streaming.consumer import StreamingConsumer, MockKafkaConsumer

        consumer = StreamingConsumer()
        consumer._use_confluent = False
        mock_consumer = MockKafkaConsumer()
        consumer._consumer = mock_consumer

        mock_consumer.add_message(
            "mini_quant_bars",
            "AAPL",
            {
                "symbol": "AAPL",
                "open": 185.0,
                "high": 186.0,
                "low": 184.5,
                "close": 185.5,
                "volume": 1000000,
            }
        )

        consumed = []

        def bar_handler(record):
            consumed.append(record)

        consumer.register_handler("bars", bar_handler)

        messages = mock_consumer.consume(timeout=1.0)

        for msg in messages:
            data_type = msg["topic"].replace("mini_quant_", "")
            consumer._handler.handle(data_type, msg["value"])

        assert len(consumed) == 1
        assert consumed[0]["symbol"] == "AAPL"


class TestKafkaSmokeTest:
    """Test cases for Kafka smoke test script."""

    def test_smoke_test_initialization(self):
        """Test smoke test initializes correctly."""
        from scripts.ingest_live_kafka_smoke import KafkaSmokeTest

        test = KafkaSmokeTest(num_symbols=5, num_bars=10)

        assert test.num_symbols == 5
        assert test.num_bars == 10
        assert len(test.symbols) == 5

    def test_sample_data_generation(self):
        """Test sample data generation."""
        from scripts.ingest_live_kafka_smoke import KafkaSmokeTest

        test = KafkaSmokeTest(num_symbols=3, num_bars=5)

        data = test.generate_sample_data()

        assert len(data) == 15
        assert all("symbol" in d for d in data)
        assert all("open" in d for d in data)
        assert all("close" in d for d in data)

    def test_schema_test(self):
        """Test schema validation in smoke test."""
        from scripts.ingest_live_kafka_smoke import KafkaSmokeTest

        test = KafkaSmokeTest()

        result = test.run_schema_test()

        assert result is True


class TestMockProducers:
    """Test cases for mock producers/consumers."""

    def test_mock_producer_messages(self):
        """Test mock producer stores messages."""
        from data.ingest_streaming.producer import MockKafkaProducer

        producer = MockKafkaProducer()

        producer.produce("test_topic", "key1", {"value": 1})
        producer.produce("test_topic", "key2", {"value": 2})

        messages = producer.get_messages()

        assert len(messages) == 2
        assert messages[0]["topic"] == "test_topic"
        assert messages[0]["key"] == "key1"

    def test_mock_consumer_messages(self):
        """Test mock consumer stores and returns messages."""
        from data.ingest_streaming.consumer import MockKafkaConsumer

        consumer = MockKafkaConsumer()

        consumer.add_message("topic1", "key1", {"data": "value1"})
        consumer.add_message("topic1", "key2", {"data": "value2"})

        messages = consumer.consume(timeout=1.0)

        assert len(messages) == 1
        assert messages[0]["key"] == "key1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

