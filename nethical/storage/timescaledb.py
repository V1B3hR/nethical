"""
TimescaleDB integration for time-series data storage.

This module provides TimescaleDB support for storing and querying time-series
data such as metrics, events, and audit logs with efficient time-based queries.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

logger = logging.getLogger(__name__)


class TimescaleDBStore:
    """
    TimescaleDB store for time-series data.

    Features:
    - Hypertable management for time-series data
    - Efficient time-based queries and aggregations
    - Automatic data retention policies
    - Connection pooling for performance
    - Continuous aggregates for pre-computed metrics
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "nethical_timeseries",
        user: str = "nethical",
        password: str = "",
        min_connections: int = 1,
        max_connections: int = 10,
        enabled: bool = True,
    ):
        """
        Initialize TimescaleDB store.

        Args:
            host: PostgreSQL/TimescaleDB host
            port: PostgreSQL/TimescaleDB port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
            enabled: Whether TimescaleDB is enabled
        """
        self.enabled = enabled and PSYCOPG2_AVAILABLE

        if not PSYCOPG2_AVAILABLE:
            logger.warning(
                "psycopg2 not available. Install with: pip install psycopg2-binary"
            )
            self.enabled = False
            return

        if not self.enabled:
            logger.info("TimescaleDB disabled by configuration")
            return

        try:
            # Create connection pool
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
            )

            # Test connection and initialize schema
            self._initialize_schema()
            logger.info(f"TimescaleDB connected to {host}:{port}/{database}")

        except Exception as e:
            logger.warning(f"Failed to connect to TimescaleDB: {e}")
            self.enabled = False

    def _get_connection(self):
        """Get connection from pool."""
        if not self.enabled:
            raise RuntimeError("TimescaleDB not enabled")
        return self.pool.getconn()

    def _return_connection(self, conn):
        """Return connection to pool."""
        if self.enabled:
            self.pool.putconn(conn)

    def _initialize_schema(self):
        """Initialize database schema with hypertables."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create metrics table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS governance_metrics (
                        time TIMESTAMPTZ NOT NULL,
                        agent_id TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value DOUBLE PRECISION,
                        region_id TEXT,
                        logical_domain TEXT,
                        metadata JSONB,
                        PRIMARY KEY (time, agent_id, metric_name)
                    )
                """
                )

                # Create hypertable if not exists
                cur.execute(
                    """
                    SELECT create_hypertable(
                        'governance_metrics',
                        'time',
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 day'
                    )
                """
                )

                # Create events table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS governance_events (
                        time TIMESTAMPTZ NOT NULL,
                        event_type TEXT NOT NULL,
                        agent_id TEXT,
                        severity TEXT,
                        data JSONB,
                        region_id TEXT,
                        logical_domain TEXT
                    )
                """
                )

                cur.execute(
                    """
                    SELECT create_hypertable(
                        'governance_events',
                        'time',
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 day'
                    )
                """
                )

                # Create audit logs table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        time TIMESTAMPTZ NOT NULL,
                        action_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        decision TEXT,
                        violations JSONB,
                        region_id TEXT,
                        logical_domain TEXT,
                        metadata JSONB
                    )
                """
                )

                cur.execute(
                    """
                    SELECT create_hypertable(
                        'audit_logs',
                        'time',
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 day'
                    )
                """
                )

                # Create indexes for common queries
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_metrics_agent_time
                    ON governance_metrics (agent_id, time DESC)
                """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_metrics_region_time
                    ON governance_metrics (region_id, time DESC)
                """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_events_type_time
                    ON governance_events (event_type, time DESC)
                """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_audit_agent_time
                    ON audit_logs (agent_id, time DESC)
                """
                )

                conn.commit()

        finally:
            self._return_connection(conn)

    def insert_metric(
        self,
        agent_id: str,
        metric_name: str,
        metric_value: float,
        timestamp: Optional[datetime] = None,
        region_id: Optional[str] = None,
        logical_domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Insert a metric data point.

        Args:
            agent_id: Agent identifier
            metric_name: Name of the metric
            metric_value: Metric value
            timestamp: Timestamp (uses current time if None)
            region_id: Region identifier
            logical_domain: Logical domain
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        timestamp = timestamp or datetime.now(timezone.utc)
        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO governance_metrics
                    (time, agent_id, metric_name, metric_value, region_id, logical_domain, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (time, agent_id, metric_name)
                    DO UPDATE SET metric_value = EXCLUDED.metric_value
                """,
                    (
                        timestamp,
                        agent_id,
                        metric_name,
                        metric_value,
                        region_id,
                        logical_domain,
                        psycopg2.extras.Json(metadata) if metadata else None,
                    ),
                )
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error inserting metric: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    def insert_event(
        self,
        event_type: str,
        agent_id: Optional[str] = None,
        severity: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        region_id: Optional[str] = None,
        logical_domain: Optional[str] = None,
    ) -> bool:
        """
        Insert an event.

        Args:
            event_type: Type of event
            agent_id: Agent identifier
            severity: Event severity
            data: Event data
            timestamp: Timestamp (uses current time if None)
            region_id: Region identifier
            logical_domain: Logical domain

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        timestamp = timestamp or datetime.now(timezone.utc)
        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO governance_events
                    (time, event_type, agent_id, severity, data, region_id, logical_domain)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        timestamp,
                        event_type,
                        agent_id,
                        severity,
                        psycopg2.extras.Json(data) if data else None,
                        region_id,
                        logical_domain,
                    ),
                )
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error inserting event: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    def insert_audit_log(
        self,
        action_id: str,
        agent_id: str,
        decision: str,
        violations: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[datetime] = None,
        region_id: Optional[str] = None,
        logical_domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Insert an audit log entry.

        Args:
            action_id: Action identifier
            agent_id: Agent identifier
            decision: Decision made
            violations: List of violations
            timestamp: Timestamp (uses current time if None)
            region_id: Region identifier
            logical_domain: Logical domain
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        timestamp = timestamp or datetime.now(timezone.utc)
        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_logs
                    (time, action_id, agent_id, decision, violations, region_id, logical_domain, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        timestamp,
                        action_id,
                        agent_id,
                        decision,
                        psycopg2.extras.Json(violations) if violations else None,
                        region_id,
                        logical_domain,
                        psycopg2.extras.Json(metadata) if metadata else None,
                    ),
                )
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error inserting audit log: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)

    def query_metrics(
        self,
        agent_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        region_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Query metrics with time-based filtering.

        Args:
            agent_id: Filter by agent ID
            metric_name: Filter by metric name
            start_time: Start of time range
            end_time: End of time range
            region_id: Filter by region
            limit: Maximum number of results

        Returns:
            List of metric records
        """
        if not self.enabled:
            return []

        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT * FROM governance_metrics WHERE 1=1"
                params = []

                if agent_id:
                    query += " AND agent_id = %s"
                    params.append(agent_id)

                if metric_name:
                    query += " AND metric_name = %s"
                    params.append(metric_name)

                if start_time:
                    query += " AND time >= %s"
                    params.append(start_time)

                if end_time:
                    query += " AND time <= %s"
                    params.append(end_time)

                if region_id:
                    query += " AND region_id = %s"
                    params.append(region_id)

                query += " ORDER BY time DESC LIMIT %s"
                params.append(limit)

                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]

        except Exception as e:
            logger.error(f"Error querying metrics: {e}")
            return []
        finally:
            self._return_connection(conn)

    def aggregate_metrics(
        self,
        metric_name: str,
        aggregation: str = "avg",
        time_bucket: str = "1 hour",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        region_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate metrics over time buckets.

        Args:
            metric_name: Metric to aggregate
            aggregation: Aggregation function (avg, sum, min, max, count)
            time_bucket: Time bucket size (e.g., '1 hour', '1 day')
            start_time: Start of time range
            end_time: End of time range
            agent_id: Filter by agent ID
            region_id: Filter by region

        Returns:
            List of aggregated results
        """
        if not self.enabled:
            return []

        valid_aggs = ["avg", "sum", "min", "max", "count"]
        if aggregation not in valid_aggs:
            raise ValueError(f"Invalid aggregation: {aggregation}")

        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Safe: aggregation is validated against whitelist above
                query = f"""
                    SELECT
                        time_bucket(%s, time) AS bucket,
                        {aggregation}(metric_value) AS value,
                        COUNT(*) AS count
                    FROM governance_metrics
                    WHERE metric_name = %s
                """
                params = [time_bucket, metric_name]

                if start_time:
                    query += " AND time >= %s"
                    params.append(start_time)

                if end_time:
                    query += " AND time <= %s"
                    params.append(end_time)

                if agent_id:
                    query += " AND agent_id = %s"
                    params.append(agent_id)

                if region_id:
                    query += " AND region_id = %s"
                    params.append(region_id)

                query += " GROUP BY bucket ORDER BY bucket DESC"

                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]

        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
            return []
        finally:
            self._return_connection(conn)

    def close(self):
        """Close all connections in pool."""
        if self.enabled and hasattr(self, "pool"):
            try:
                self.pool.closeall()
                logger.info("TimescaleDB connections closed")
            except Exception as e:
                logger.error(f"Error closing TimescaleDB connections: {e}")
