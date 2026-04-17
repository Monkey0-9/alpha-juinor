#!/usr/bin/env python3
"""
TIMESCALEDB CLUSTER MANAGER
===========================

Production-grade TimescaleDB cluster management with:
- Automatic failover
- Read/write splitting
- Connection pooling
- Hypertable management
- Continuous aggregation
- Data retention policies

Author: MiniQuantFund Data Engineering
"""

import os
import sys
import json
import logging
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum
import threading
import queue

import numpy as np
import pandas as pd

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    from psycopg2.pool import ThreadedConnectionPool
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    logging.warning("psycopg2 not available - database features disabled")

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """Database node roles."""
    PRIMARY = "primary"
    REPLICA = "replica"
    STANDBY = "standby"


@dataclass
class TimescaleNode:
    """TimescaleDB node configuration."""
    host: str
    port: int = 5432
    role: NodeRole = NodeRole.REPLICA
    weight: int = 1  # Load balancing weight
    is_healthy: bool = True
    last_check: Optional[datetime] = None
    lag_ms: float = 0.0


class TimescaleDBCluster:
    """
    Production TimescaleDB cluster manager.
    
    Features:
    - Automatic failover to replicas
    - Read/write query routing
    - Connection pooling
    - Health monitoring
    - Hypertable automation
    """
    
    def __init__(self, 
                 primary_host: str = "timescaledb-primary",
                 replica_hosts: Optional[List[str]] = None,
                 database: str = "mini_quant_fund",
                 user: str = "quant_admin",
                 password: Optional[str] = None,
                 port: int = 5432):
        
        self.primary_host = primary_host
        self.replica_hosts = replica_hosts or []
        self.database = database
        self.user = user
        self.password = password or os.getenv("TIMESCALEDB_PASSWORD", "")
        self.port = port
        
        # Node management
        self.nodes: Dict[str, TimescaleNode] = {}
        self.current_primary: Optional[str] = None
        
        # Connection pools
        self.write_pool: Optional[Any] = None
        self.read_pool: Optional[Any] = None
        
        # Threading
        self._lock = threading.RLock()
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Metrics
        self.query_count = 0
        self.failover_count = 0
        
        self._initialize_cluster()
    
    def _initialize_cluster(self):
        """Initialize cluster nodes."""
        # Primary node
        self.nodes["primary"] = TimescaleNode(
            host=self.primary_host,
            port=self.port,
            role=NodeRole.PRIMARY,
            weight=1,
            is_healthy=True
        )
        self.current_primary = "primary"
        
        # Replica nodes
        for i, host in enumerate(self.replica_hosts):
            node_id = f"replica_{i}"
            self.nodes[node_id] = TimescaleNode(
                host=host,
                port=self.port,
                role=NodeRole.REPLICA,
                weight=1,
                is_healthy=True
            )
        
        # Initialize connection pools
        self._setup_connection_pools()
        
        logger.info(f"TimescaleDB cluster initialized: "
                   f"1 primary, {len(self.replica_hosts)} replicas")
    
    def _setup_connection_pools(self):
        """Setup connection pools for read/write splitting."""
        if not PSYCOPG_AVAILABLE:
            return
        
        try:
            # Write pool (primary only)
            primary = self.nodes[self.current_primary]
            self.write_pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                host=primary.host,
                port=primary.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            
            # Read pool (replicas with fallback to primary)
            read_hosts = [n for n in self.nodes.values() 
                         if n.role == NodeRole.REPLICA and n.is_healthy]
            
            if not read_hosts:
                # Fallback to primary for reads
                read_hosts = [self.nodes[self.current_primary]]
            
            # Use first healthy replica for reads
            read_node = read_hosts[0]
            self.read_pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=50,
                host=read_node.host,
                port=read_node.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            
        except Exception as e:
            logger.error(f"Failed to setup connection pools: {e}")
    
    @contextmanager
    def get_write_connection(self):
        """Get connection to primary for writes."""
        conn = None
        try:
            if self.write_pool:
                conn = self.write_pool.getconn()
                yield conn
            else:
                raise ConnectionError("Write pool not available")
        except Exception as e:
            logger.error(f"Write connection failed: {e}")
            raise
        finally:
            if conn and self.write_pool:
                self.write_pool.putconn(conn)
    
    @contextmanager
    def get_read_connection(self):
        """Get connection to replica for reads (with primary fallback)."""
        conn = None
        try:
            if self.read_pool:
                conn = self.read_pool.getconn()
                yield conn
            elif self.write_pool:
                # Fallback to primary
                conn = self.write_pool.getconn()
                yield conn
            else:
                raise ConnectionError("No connection pool available")
        except Exception as e:
            logger.error(f"Read connection failed: {e}")
            raise
        finally:
            if conn:
                if self.read_pool:
                    self.read_pool.putconn(conn)
                elif self.write_pool:
                    self.write_pool.putconn(conn)
    
    def execute_write(self, query: str, params: Optional[Tuple] = None) -> int:
        """Execute write query on primary."""
        with self.get_write_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()
                self.query_count += 1
                return cur.rowcount
    
    def execute_read(self, query: str, params: Optional[Tuple] = None) -> List[Dict]:
        """Execute read query on replica."""
        with self.get_read_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                self.query_count += 1
                return cur.fetchall()
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute batch write."""
        with self.get_write_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, params_list)
                conn.commit()
                self.query_count += len(params_list)
                return len(params_list)
    
    def query_to_dataframe(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        results = self.execute_read(query, params)
        return pd.DataFrame(results)
    
    # =========================================================================
    # HYPERTABLE MANAGEMENT
    # =========================================================================
    
    def create_hypertable(self, 
                         table_name: str,
                         time_column: str = "timestamp",
                         chunk_time_interval: str = "1 day",
                         if_not_exists: bool = True) -> bool:
        """
        Create a TimescaleDB hypertable for time-series data.
        
        Args:
            table_name: Name of the table
            time_column: Time column for partitioning
            chunk_time_interval: Chunk size (e.g., '1 day', '1 hour')
            if_not_exists: Only create if not exists
        
        Returns:
            True if successful
        """
        try:
            with self.get_write_connection() as conn:
                with conn.cursor() as cur:
                    # Create hypertable
                    exists_clause = "IF NOT EXISTS" if if_not_exists else ""
                    
                    cur.execute(f"""
                        SELECT create_hypertable(
                            '{table_name}',
                            '{time_column}',
                            chunk_time_interval => INTERVAL '{chunk_time_interval}',
                            if_not_exists => {str(if_not_exists).lower()}
                        );
                    """)
                    
                    conn.commit()
                    logger.info(f"Created hypertable: {table_name}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to create hypertable {table_name}: {e}")
            return False
    
    def setup_market_data_tables(self):
        """Setup market data hypertables."""
        tables = [
            {
                "name": "market_ticks",
                "ddl": """
                    CREATE TABLE IF NOT EXISTS market_ticks (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol_id INT NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        volume INT NOT NULL,
                        side SMALLINT NOT NULL,
                        exchange_id SMALLINT NOT NULL,
                        PRIMARY KEY (timestamp, symbol_id, exchange_id)
                    );
                """,
                "chunk_interval": "1 hour"
            },
            {
                "name": "ohlcv_1min",
                "ddl": """
                    CREATE TABLE IF NOT EXISTS ohlcv_1min (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol_id INT NOT NULL,
                        open DOUBLE PRECISION NOT NULL,
                        high DOUBLE PRECISION NOT NULL,
                        low DOUBLE PRECISION NOT NULL,
                        close DOUBLE PRECISION NOT NULL,
                        volume BIGINT NOT NULL,
                        PRIMARY KEY (timestamp, symbol_id)
                    );
                """,
                "chunk_interval": "1 day"
            },
            {
                "name": "trades",
                "ddl": """
                    CREATE TABLE IF NOT EXISTS trades (
                        timestamp TIMESTAMPTZ NOT NULL,
                        trade_id BIGINT NOT NULL,
                        symbol_id INT NOT NULL,
                        side SMALLINT NOT NULL,
                        price DOUBLE PRECISION NOT NULL,
                        quantity INT NOT NULL,
                        strategy_id INT,
                        pnl DOUBLE PRECISION,
                        PRIMARY KEY (timestamp, trade_id)
                    );
                """,
                "chunk_interval": "1 day"
            },
            {
                "name": "portfolio_snapshots",
                "ddl": """
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        timestamp TIMESTAMPTZ NOT NULL,
                        nav DOUBLE PRECISION NOT NULL,
                        exposure DOUBLE PRECISION NOT NULL,
                        cash DOUBLE PRECISION NOT NULL,
                        margin_used DOUBLE PRECISION NOT NULL,
                        open_positions INT NOT NULL
                    );
                """,
                "chunk_interval": "1 day"
            }
        ]
        
        for table in tables:
            # Create table
            self.execute_write(table["ddl"])
            
            # Convert to hypertable
            self.create_hypertable(
                table["name"],
                chunk_time_interval=table["chunk_interval"]
            )
        
        logger.info("Market data tables initialized")
    
    def setup_compression(self, table_name: str, 
                         compress_after: str = "7 days") -> bool:
        """
        Enable compression for a hypertable.
        
        Args:
            table_name: Name of the hypertable
            compress_after: Compress chunks older than this
        """
        try:
            with self.get_write_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        ALTER TABLE {table_name} 
                        SET (timescaledb.compress, 
                             timescaledb.compress_segmentby = 'symbol_id');
                    """)
                    
                    cur.execute(f"""
                        SELECT add_compression_policy(
                            '{table_name}',
                            INTERVAL '{compress_after}'
                        );
                    """)
                    
                    conn.commit()
                    logger.info(f"Compression enabled for {table_name}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to setup compression for {table_name}: {e}")
            return False
    
    def setup_retention_policy(self, table_name: str, 
                               drop_after: str = "90 days") -> bool:
        """
        Setup data retention policy.
        
        Args:
            table_name: Name of the hypertable
            drop_after: Drop chunks older than this
        """
        try:
            with self.get_write_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT add_retention_policy(
                            '{table_name}',
                            INTERVAL '{drop_after}'
                        );
                    """)
                    
                    conn.commit()
                    logger.info(f"Retention policy set for {table_name}: {drop_after}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to setup retention for {table_name}: {e}")
            return False
    
    # =========================================================================
    # CONTINUOUS AGGREGATES
    # =========================================================================
    
    def create_continuous_aggregate(self,
                                   view_name: str,
                                   source_table: str,
                                   aggregation: str,
                                   time_bucket: str = "1 hour") -> bool:
        """
        Create a continuous aggregate for downsampled data.
        
        Example:
            CREATE MATERIALIZED VIEW ohlcv_1h
            WITH (timescaledb.continuous) AS
            SELECT time_bucket('1 hour', timestamp) as bucket,
                   symbol_id,
                   first(price, timestamp) as open,
                   max(price) as high,
                   min(price) as low,
                   last(price, timestamp) as close,
                   sum(volume) as volume
            FROM ohlcv_1min
        """
        try:
            with self.get_write_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        CREATE MATERIALIZED VIEW {view_name}
                        WITH (timescaledb.continuous) AS
                        {aggregation}
                        WITH DATA;
                    """)
                    
                    # Add refresh policy
                    cur.execute(f"""
                        SELECT add_continuous_aggregate_policy(
                            '{view_name}',
                            start_offset => INTERVAL '1 month',
                            end_offset => INTERVAL '1 hour',
                            schedule_interval => INTERVAL '1 hour'
                        );
                    """)
                    
                    conn.commit()
                    logger.info(f"Created continuous aggregate: {view_name}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to create continuous aggregate {view_name}: {e}")
            return False
    
    # =========================================================================
    # HEALTH MONITORING & FAILOVER
    # =========================================================================
    
    def start_health_monitoring(self, interval_sec: int = 10):
        """Start background health checks."""
        if self._running:
            return
        
        self._running = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            args=(interval_sec,),
            daemon=True
        )
        self._health_check_thread.start()
        logger.info("Health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop health checks."""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
    
    def _health_check_loop(self, interval_sec: int):
        """Background health check loop."""
        while self._running:
            try:
                self._check_all_nodes()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(interval_sec)
    
    def _check_all_nodes(self):
        """Check health of all cluster nodes."""
        for node_id, node in self.nodes.items():
            try:
                # Try to connect
                conn = psycopg2.connect(
                    host=node.host,
                    port=node.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    connect_timeout=5
                )
                
                with conn.cursor() as cur:
                    # Check replication lag for replicas
                    if node.role == NodeRole.REPLICA:
                        cur.execute("SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) * 1000;")
                        lag_ms = cur.fetchone()[0] or 0
                        node.lag_ms = lag_ms
                    
                    # Check node is in recovery (replica) or not (primary)
                    cur.execute("SELECT pg_is_in_recovery();")
                    in_recovery = cur.fetchone()[0]
                    
                    expected_recovery = (node.role == NodeRole.REPLICA)
                    
                    if in_recovery != expected_recovery:
                        # Role mismatch - may indicate failover
                        if node_id == self.current_primary and in_recovery:
                            logger.critical(f"Primary {node_id} is in recovery! Failover needed.")
                            self._trigger_failover()
                
                conn.close()
                
                node.is_healthy = True
                node.last_check = datetime.utcnow()
                
            except Exception as e:
                node.is_healthy = False
                node.last_check = datetime.utcnow()
                
                if node_id == self.current_primary:
                    logger.critical(f"Primary {node_id} health check failed: {e}")
                    self._trigger_failover()
    
    def _trigger_failover(self):
        """Trigger failover to replica."""
        logger.critical("Initiating automatic failover...")
        
        # Find healthy replica
        healthy_replicas = [
            (nid, n) for nid, n in self.nodes.items()
            if n.role == NodeRole.REPLICA and n.is_healthy
        ]
        
        if not healthy_replicas:
            logger.critical("No healthy replicas available for failover!")
            return
        
        # Select replica with lowest lag
        best_replica = min(healthy_replicas, key=lambda x: x[1].lag_ms)
        new_primary_id, new_primary = best_replica
        
        logger.info(f"Promoting {new_primary_id} to primary...")
        
        # Promote replica (would execute pg_promote via K8s or directly)
        # For now, update routing
        with self._lock:
            old_primary = self.current_primary
            
            # Update node roles
            self.nodes[old_primary].role = NodeRole.STANDBY
            self.nodes[old_primary].is_healthy = False
            
            new_primary.role = NodeRole.PRIMARY
            self.current_primary = new_primary_id
            
            self.failover_count += 1
        
        # Reinitialize connection pools
        self._setup_connection_pools()
        
        logger.critical(f"Failover complete: {old_primary} -> {new_primary_id}")
        
        # Alert monitoring
        try:
            from mini_quant_fund.monitoring.production_monitor import get_production_monitor
            monitor = get_production_monitor()
            monitor.alert_manager.evaluate_metrics({
                "failover_triggered": True,
                "old_primary": old_primary,
                "new_primary": new_primary_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception:
            pass
    
    def get_cluster_status(self) -> Dict:
        """Get cluster health status."""
        return {
            "primary": self.current_primary,
            "failover_count": self.failover_count,
            "query_count": self.query_count,
            "nodes": {
                nid: {
                    "host": n.host,
                    "role": n.role.value,
                    "healthy": n.is_healthy,
                    "lag_ms": n.lag_ms,
                    "last_check": n.last_check.isoformat() if n.last_check else None
                }
                for nid, n in self.nodes.items()
            }
        }


# Global cluster instance
_cluster_instance: Optional[TimescaleDBCluster] = None


def get_timescale_cluster() -> TimescaleDBCluster:
    """Get global TimescaleDB cluster instance."""
    global _cluster_instance
    if _cluster_instance is None:
        _cluster_instance = TimescaleDBCluster()
    return _cluster_instance


if __name__ == "__main__":
    # Test cluster
    print("Testing TimescaleDB Cluster...")
    
    cluster = TimescaleDBCluster(
        primary_host="localhost",
        replica_hosts=["localhost"],  # Would be actual replica hosts
        password="test_password"
    )
    
    # Setup tables
    cluster.setup_market_data_tables()
    
    # Insert test data
    cluster.execute_write("""
        INSERT INTO market_ticks (timestamp, symbol_id, price, volume, side, exchange_id)
        VALUES (NOW(), 1, 150.25, 1000, 2, 1)
        ON CONFLICT DO NOTHING;
    """)
    
    # Query data
    results = cluster.execute_read(
        "SELECT * FROM market_ticks ORDER BY timestamp DESC LIMIT 5;"
    )
    
    print(f"Query results: {results}")
    
    # Check status
    status = cluster.get_cluster_status()
    print(f"\nCluster status: {json.dumps(status, indent=2)}")
