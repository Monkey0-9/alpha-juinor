"""
Real-Time Data Lake
===================

Stream processing with Kafka, data quality monitoring, and lineage tracking.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DataRecord:
    """Data record in the lake."""

    source: str
    timestamp: datetime
    data_type: str
    payload: Dict
    quality_score: float


class KafkaStreamProcessor:
    """
    Apache Kafka stream processor for real-time data ingestion.

    Topics:
    - market-data
    - news-feed
    - social-sentiment
    - alternative-data
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.topics: Dict[str, List[Callable]] = {}
        self.simulated = True  # Simulation mode

    def subscribe(self, topic: str, callback: Callable[[Dict], None]):
        """
        Subscribe to topic with callback.

        Args:
            topic: Kafka topic name
            callback: Function to process message
        """
        if topic not in self.topics:
            self.topics[topic] = []

        self.topics[topic].append(callback)
        logger.info(f"Subscribed to topic: {topic}")

    def publish(self, topic: str, message: Dict):
        """
        Publish message to topic.

        Args:
            topic: Topic name
            message: Message payload
        """
        if self.simulated:
            # Simulate immediate delivery to subscribers
            if topic in self.topics:
                for callback in self.topics[topic]:
                    callback(message)

        logger.debug(f"Published to {topic}: {message}")

    def process_stream(self, topic: str, max_messages: Optional[int] = None):
        """
        Process messages from stream (blocking).

        Args:
            topic: Topic to process
            max_messages: Max messages to process
        """
        # In real implementation, would use kafka-python library
        logger.info(f"Processing stream: {topic}")


class DataQualityMonitor:
    """
    Monitors data quality in real-time.

    Checks:
    - Completeness
    - Timeliness
    - Accuracy
    - Consistency
    """

    def __init__(self):
        self.quality_scores: Dict[str, List[float]] = {}
        self.alerts: List[Dict] = []

    def assess_quality(self, record: DataRecord) -> float:
        """
        Assess data quality.

        Args:
            record: Data record

        Returns:
            Quality score [0, 1]
        """
        score = 1.0

        # Completeness check
        if not record.payload:
            score *= 0.5

        required_fields = {"symbol", "timestamp", "value"}
        if record.data_type == "market-data":
            missing = required_fields - set(record.payload.keys())
            score *= (len(required_fields) - len(missing)) / len(required_fields)

        # Timeliness check
        age_seconds = (datetime.now() - record.timestamp).total_seconds()
        if age_seconds > 60:  # Stale if > 1 minute
            score *= 0.7

        # Store score
        if record.source not in self.quality_scores:
            self.quality_scores[record.source] = []

        self.quality_scores[record.source].append(score)

        # Alert if quality drops
        if score < 0.6:
            self.alerts.append(
                {
                    "timestamp": datetime.now(),
                    "source": record.source,
                    "quality_score": score,
                    "reason": "Low quality data detected",
                }
            )

        return score

    def get_source_quality(self, source: str) -> Optional[float]:
        """Get average quality score for a source."""
        if source not in self.quality_scores:
            return None

        return sum(self.quality_scores[source]) / len(self.quality_scores[source])


class DataLineageTracker:
    """
    Tracks data lineage for compliance and debugging.

    Records:
    - Data sources
    - Transformations applied
    - Downstream consumers
    """

    def __init__(self):
        self.lineage_graph: Dict[str, Dict] = {}

    def record_source(self, data_id: str, source: str, metadata: Dict):
        """
        Record data source.

        Args:
            data_id: Unique data identifier
            source: Source system
            metadata: Additional metadata
        """
        self.lineage_graph[data_id] = {
            "source": source,
            "metadata": metadata,
            "transformations": [],
            "consumers": [],
            "created_at": datetime.now(),
        }

    def record_transformation(
        self, input_id: str, output_id: str, transform_type: str, params: Dict
    ):
        """
        Record data transformation.

        Args:
            input_id: Input data ID
            output_id: Output data ID
            transform_type: Type of transformation
            params: Transformation parameters
        """
        if input_id in self.lineage_graph:
            self.lineage_graph[input_id]["transformations"].append(
                {
                    "output_id": output_id,
                    "type": transform_type,
                    "params": params,
                    "timestamp": datetime.now(),
                }
            )

        # Create entry for output
        self.record_source(output_id, f"transform({input_id})", {"parent": input_id})

    def record_consumer(self, data_id: str, consumer: str):
        """Record downstream consumer of data."""
        if data_id in self.lineage_graph:
            self.lineage_graph[data_id]["consumers"].append(
                {"consumer": consumer, "timestamp": datetime.now()}
            )

    def get_lineage(self, data_id: str) -> Optional[Dict]:
        """Get full lineage for a data ID."""
        return self.lineage_graph.get(data_id)

    def trace_upstream(self, data_id: str) -> List[str]:
        """Trace upstream sources for a data ID."""
        upstream = []

        if data_id in self.lineage_graph:
            metadata = self.lineage_graph[data_id]["metadata"]
            if "parent" in metadata:
                parent = metadata["parent"]
                upstream.append(parent)
                upstream.extend(self.trace_upstream(parent))

        return upstream
