"""
streaming/kafka_consumer.py

Kafka consumer for real-time market data and event streaming.
Provides high-throughput, fault-tolerant message processing.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError, CommitFailedError
    HAS_KAFKA = True
except ImportError:
    KafkaConsumer = None
    KafkaProducer = None
    KafkaError = Exception
    CommitFailedError = Exception
    HAS_KAFKA = False
import redis

from monitoring.structured_logger import get_logger

logger = get_logger("kafka_consumer")


class MessageType(Enum):
    """Message types for Kafka topics."""
    MARKET_DATA = "MARKET_DATA"
    TRADE_EXECUTION = "TRADE_EXECUTION"
    ORDER_UPDATE = "ORDER_UPDATE"
    RISK_ALERT = "RISK_ALERT"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    STRATEGY_SIGNAL = "STRATEGY_SIGNAL"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"


@dataclass
class KafkaMessage:
    """Structured Kafka message."""
    topic: str
    partition: int
    offset: int
    key: str
    value: Dict[str, Any]
    timestamp: datetime
    message_type: MessageType
    correlation_id: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConsumerConfig:
    """Configuration for Kafka consumer."""
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    group_id: str = "quant-fund-consumer"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    auto_commit_interval_ms: int = 1000
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    fetch_min_bytes: int = 1
    fetch_max_wait_ms: int = 500
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None


@dataclass
class TopicConfig:
    """Configuration for Kafka topics."""
    name: str
    num_partitions: int = 3
    replication_factor: int = 1
    retention_ms: int = 86400000  # 24 hours
    cleanup_policy: str = "delete"
    compression_type: str = "snappy"


class KafkaMessageProcessor:
    """Base class for message processors."""
    
    async def process_message(self, message: KafkaMessage) -> bool:
        """Process a single message. Return True if successful."""
        raise NotImplementedError
    
    async def process_batch(self, messages: List[KafkaMessage]) -> List[bool]:
        """Process a batch of messages. Return list of success flags."""
        results = []
        for message in messages:
            try:
                result = await self.process_message(message)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                results.append(False)
        return results


class MarketDataProcessor(KafkaMessageProcessor):
    """Processor for market data messages."""
    
    def __init__(self, callback: Callable[[Dict[str, Any]], None]):
        self.callback = callback
        self.logger = get_logger("market_data_processor")
    
    async def process_message(self, message: KafkaMessage) -> bool:
        """Process market data message."""
        try:
            if message.message_type == MessageType.MARKET_DATA:
                # Validate market data
                data = message.value
                required_fields = ['symbol', 'price', 'volume', 'timestamp']
                
                if not all(field in data for field in required_fields):
                    self.logger.warning(f"Invalid market data message: missing required fields")
                    return False
                
                # Convert timestamp
                if isinstance(data['timestamp'], str):
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                
                # Call callback
                self.callback(data)
                
                self.logger.debug(
                    f"Processed market data",
                    symbol=data['symbol'],
                    price=data['price'],
                    volume=data['volume']
                )
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return False


class OrderProcessor(KafkaMessageProcessor):
    """Processor for order-related messages."""
    
    def __init__(self, execution_engine):
        self.execution_engine = execution_engine
        self.logger = get_logger("order_processor")
    
    async def process_message(self, message: KafkaMessage) -> bool:
        """Process order message."""
        try:
            if message.message_type in [MessageType.TRADE_EXECUTION, MessageType.ORDER_UPDATE]:
                data = message.value
                
                if message.message_type == MessageType.TRADE_EXECUTION:
                    # Handle trade execution
                    await self._handle_trade_execution(data)
                elif message.message_type == MessageType.ORDER_UPDATE:
                    # Handle order update
                    await self._handle_order_update(data)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing order message: {e}")
            return False
    
    async def _handle_trade_execution(self, data: Dict[str, Any]):
        """Handle trade execution message."""
        # Update execution engine with trade
        self.logger.info(
            f"Trade execution received",
            order_id=data.get('order_id'),
            symbol=data.get('symbol'),
            quantity=data.get('quantity'),
            price=data.get('price')
        )
    
    async def _handle_order_update(self, data: Dict[str, Any]):
        """Handle order update message."""
        # Update order status
        self.logger.info(
            f"Order update received",
            order_id=data.get('order_id'),
            status=data.get('status'),
            filled_quantity=data.get('filled_quantity')
        )


class RiskProcessor(KafkaMessageProcessor):
    """Processor for risk-related messages."""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.logger = get_logger("risk_processor")
    
    async def process_message(self, message: KafkaMessage) -> bool:
        """Process risk message."""
        try:
            if message.message_type == MessageType.RISK_ALERT:
                data = message.value
                
                # Handle risk alert
                await self._handle_risk_alert(data)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing risk message: {e}")
            return False
    
    async def _handle_risk_alert(self, data: Dict[str, Any]):
        """Handle risk alert message."""
        self.logger.warning(
            f"Risk alert received",
            alert_type=data.get('alert_type'),
            severity=data.get('severity'),
            description=data.get('description')
        )


class AdvancedKafkaConsumer:
    """Advanced Kafka consumer with high-throughput processing."""
    
    def __init__(self, config: ConsumerConfig):
        if not HAS_KAFKA:
            raise ImportError("kafka-python is required for AdvancedKafkaConsumer")
        self.config = config
        self.logger = logger
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        self.processors: Dict[str, KafkaMessageProcessor] = {}
        self.running = False
        
        # Redis for state management
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Metrics
        self.messages_processed = 0
        self.errors_count = 0
        self.last_commit_time = datetime.utcnow()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def start(self, topics: List[str], processors: Dict[str, KafkaMessageProcessor]):
        """Start consuming messages."""
        self.logger.info(f"Starting Kafka consumer for topics: {topics}")
        
        try:
            # Initialize consumer
            self.consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                auto_commit_interval_ms=self.config.auto_commit_interval_ms,
                session_timeout_ms=self.config.session_timeout_ms,
                heartbeat_interval_ms=self.config.heartbeat_interval_ms,
                max_poll_records=self.config.max_poll_records,
                max_poll_interval_ms=self.config.max_poll_interval_ms,
                fetch_min_bytes=self.config.fetch_min_bytes,
                fetch_max_wait_ms=self.config.fetch_max_wait_ms,
                security_protocol=self.config.security_protocol,
                sasl_mechanism=self.config.sasl_mechanism,
                sasl_username=self.config.sasl_username,
                sasl_password=self.config.sasl_password,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None,
                key_deserializer=lambda k: k.decode('utf-8') if k else None
            )
            
            # Initialize producer for dead letter queue
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                security_protocol=self.config.security_protocol,
                sasl_mechanism=self.config.sasl_mechanism,
                sasl_username=self.config.sasl_username,
                sasl_password=self.config.sasl_password,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            # Set processors
            self.processors = processors
            
            # Start consuming
            self.running = True
            await self._consume_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start Kafka consumer: {e}")
            raise
    
    async def stop(self):
        """Stop consuming messages."""
        self.logger.info("Stopping Kafka consumer")
        
        self.running = False
        
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        self.logger.info("Kafka consumer stopped")
    
    async def _consume_loop(self):
        """Main consumption loop."""
        self.logger.info("Starting Kafka consumption loop")
        
        while self.running:
            try:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    continue
                
                # Process messages by topic
                for topic_partition, messages in message_batch.items():
                    await self._process_topic_messages(topic_partition.topic, messages)
                
                # Commit offsets periodically
                if datetime.utcnow() - self.last_commit_time > timedelta(seconds=5):
                    await self._commit_offsets()
                
            except Exception as e:
                self.logger.error(f"Error in consumption loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_topic_messages(self, topic: str, messages: List):
        """Process messages for a specific topic."""
        try:
            # Convert messages to KafkaMessage objects
            kafka_messages = []
            for msg in messages:
                kafka_message = self._parse_message(msg, topic)
                if kafka_message:
                    kafka_messages.append(kafka_message)
            
            if not kafka_messages:
                return
            
            # Get processor for this topic
            processor = self.processors.get(topic)
            if not processor:
                self.logger.warning(f"No processor found for topic: {topic}")
                return
            
            # Process messages in parallel
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                processor.process_batch,
                kafka_messages
            )
            
            # Handle failed messages
            for i, (message, success) in enumerate(zip(kafka_messages, results)):
                if not success:
                    await self._handle_failed_message(message)
                else:
                    self.messages_processed += 1
            
            # Log processing stats
            if len(kafka_messages) > 0:
                success_rate = sum(results) / len(results)
                self.logger.debug(
                    f"Processed messages for topic {topic}",
                    message_count=len(kafka_messages),
                    success_rate=success_rate,
                    total_processed=self.messages_processed
                )
            
        except Exception as e:
            self.logger.error(f"Error processing messages for topic {topic}: {e}")
    
    def _parse_message(self, msg, topic: str) -> Optional[KafkaMessage]:
        """Parse raw Kafka message to KafkaMessage object."""
        try:
            if not msg.value:
                return None
            
            # Determine message type from headers or value
            message_type = self._determine_message_type(msg, topic)
            
            return KafkaMessage(
                topic=topic,
                partition=msg.partition,
                offset=msg.offset,
                key=msg.key,
                value=msg.value,
                timestamp=datetime.fromtimestamp(msg.timestamp / 1000) if msg.timestamp else datetime.utcnow(),
                message_type=message_type,
                correlation_id=msg.headers.get('correlation_id') if msg.headers else None,
                headers=dict(msg.headers) if msg.headers else {}
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing message: {e}")
            return None
    
    def _determine_message_type(self, msg, topic: str) -> MessageType:
        """Determine message type from headers or topic."""
        # Check headers first
        if msg.headers:
            for header_key, header_value in msg.headers:
                if header_key == 'message_type':
                    try:
                        return MessageType(header_value.decode('utf-8'))
                    except ValueError:
                        pass
        
        # Determine from topic name
        if 'market_data' in topic:
            return MessageType.MARKET_DATA
        elif 'trade_execution' in topic:
            return MessageType.TRADE_EXECUTION
        elif 'order_update' in topic:
            return MessageType.ORDER_UPDATE
        elif 'risk_alert' in topic:
            return MessageType.RISK_ALERT
        elif 'system_event' in topic:
            return MessageType.SYSTEM_EVENT
        elif 'strategy_signal' in topic:
            return MessageType.STRATEGY_SIGNAL
        elif 'portfolio_update' in topic:
            return MessageType.PORTFOLIO_UPDATE
        
        # Default
        return MessageType.SYSTEM_EVENT
    
    async def _handle_failed_message(self, message: KafkaMessage):
        """Handle failed message by sending to dead letter queue."""
        try:
            self.errors_count += 1
            
            # Send to dead letter queue
            dead_letter_topic = f"{message.topic}_dead_letter"
            
            dead_letter_message = {
                'original_topic': message.topic,
                'original_partition': message.partition,
                'original_offset': message.offset,
                'original_timestamp': message.timestamp.isoformat(),
                'message': message.value,
                'error_reason': 'processing_failed',
                'failed_at': datetime.utcnow().isoformat()
            }
            
            if self.producer:
                self.producer.send(
                    dead_letter_topic,
                    key=message.key,
                    value=dead_letter_message
                )
            
            self.logger.warning(
                f"Sent message to dead letter queue",
                original_topic=message.topic,
                dead_letter_topic=dead_letter_topic,
                offset=message.offset
            )
            
        except Exception as e:
            self.logger.error(f"Error handling failed message: {e}")
    
    async def _commit_offsets(self):
        """Commit consumer offsets."""
        try:
            if self.consumer:
                self.consumer.commit()
                self.last_commit_time = datetime.utcnow()
                
                self.logger.debug(
                    "Committed consumer offsets",
                    total_processed=self.messages_processed,
                    total_errors=self.errors_count
                )
                
        except CommitFailedError as e:
            self.logger.error(f"Failed to commit offsets: {e}")
        except Exception as e:
            self.logger.error(f"Error committing offsets: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return {
            'messages_processed': self.messages_processed,
            'errors_count': self.errors_count,
            'error_rate': self.errors_count / max(self.messages_processed, 1),
            'last_commit_time': self.last_commit_time.isoformat(),
            'running': self.running,
            'active_processors': list(self.processors.keys())
        }
    
    def create_topic(self, topic_config: TopicConfig):
        """Create Kafka topic with specified configuration."""
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
            
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.config.bootstrap_servers
            )
            
            topic = NewTopic(
                name=topic_config.name,
                num_partitions=topic_config.num_partitions,
                replication_factor=topic_config.replication_factor,
                topic_configs={
                    'retention.ms': str(topic_config.retention_ms),
                    'cleanup.policy': topic_config.cleanup_policy,
                    'compression.type': topic_config.compression_type
                }
            )
            
            admin_client.create_topics([topic])
            admin_client.close()
            
            self.logger.info(
                f"Created Kafka topic",
                topic=topic_config.name,
                partitions=topic_config.num_partitions,
                replication_factor=topic_config.replication_factor
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create topic {topic_config.name}: {e}")


# Global consumer instance
kafka_consumer = None

def get_kafka_consumer(config: ConsumerConfig = None) -> AdvancedKafkaConsumer:
    """Get or create Kafka consumer."""
    global kafka_consumer
    if kafka_consumer is None:
        config = config or ConsumerConfig()
        kafka_consumer = AdvancedKafkaConsumer(config)
    return kafka_consumer


# Topic configurations
TOPICS = {
    'market_data': TopicConfig(
        name='market_data',
        num_partitions=6,
        replication_factor=1,
        retention_ms=86400000,  # 24 hours
        cleanup_policy='delete',
        compression_type='snappy'
    ),
    'trade_execution': TopicConfig(
        name='trade_execution',
        num_partitions=3,
        replication_factor=1,
        retention_ms=604800000,  # 7 days
        cleanup_policy='delete',
        compression_type='snappy'
    ),
    'order_update': TopicConfig(
        name='order_update',
        num_partitions=3,
        replication_factor=1,
        retention_ms=86400000,  # 24 hours
        cleanup_policy='delete',
        compression_type='snappy'
    ),
    'risk_alert': TopicConfig(
        name='risk_alert',
        num_partitions=2,
        replication_factor=1,
        retention_ms=604800000,  # 7 days
        cleanup_policy='delete',
        compression_type='snappy'
    ),
    'system_event': TopicConfig(
        name='system_event',
        num_partitions=2,
        replication_factor=1,
        retention_ms=2592000000,  # 30 days
        cleanup_policy='delete',
        compression_type='snappy'
    ),
    'strategy_signal': TopicConfig(
        name='strategy_signal',
        num_partitions=3,
        replication_factor=1,
        retention_ms=86400000,  # 24 hours
        cleanup_policy='delete',
        compression_type='snappy'
    ),
    'portfolio_update': TopicConfig(
        name='portfolio_update',
        num_partitions=2,
        replication_factor=1,
        retention_ms=604800000,  # 7 days
        cleanup_policy='delete',
        compression_type='snappy'
    )
}
