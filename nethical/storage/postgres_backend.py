"""
PostgreSQL storage backend for Nethical governance system.

This module provides a production-grade PostgreSQL storage backend with support for:
- Connection pooling (PgBouncer compatible)
- TimescaleDB hypertables for time-series data
- Transaction management
- Query building with parameterized queries
- Async support via asyncpg
"""

import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import uuid

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor, Json
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    pool = None
    sql = None
    RealDictCursor = None
    Json = None

logger = logging.getLogger(__name__)


@dataclass
class PostgresConfig:
    """PostgreSQL connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "nethical"
    user: str = "nethical"
    password: str = ""
    min_connections: int = 1
    max_connections: int = 20
    schema: str = "nethical"
    connect_timeout: int = 10
    application_name: str = "nethical"
    ssl_mode: str = "prefer"
    # PgBouncer compatibility
    pgbouncer_mode: bool = False


class PostgresBackend:
    """
    Production-grade PostgreSQL storage backend for Nethical.
    
    Features:
    - Connection pooling with configurable pool size
    - Schema-aware queries (defaults to 'nethical' schema)
    - Transaction support with context managers
    - Parameterized queries for SQL injection prevention
    - Batch operations for high throughput
    - TimescaleDB integration for time-series data
    
    Example:
        >>> config = PostgresConfig(host="localhost", database="nethical")
        >>> backend = PostgresBackend(config)
        >>> 
        >>> # Insert an agent
        >>> backend.insert_agent("agent-001", name="Test Agent")
        >>> 
        >>> # Query with transaction
        >>> with backend.transaction() as tx:
        ...     tx.insert_audit_event(...)
        ...     tx.insert_metric(...)
    """
    
    def __init__(self, config: Optional[PostgresConfig] = None, enabled: bool = True):
        """
        Initialize PostgreSQL backend.
        
        Args:
            config: PostgreSQL configuration. Uses defaults if not provided.
            enabled: Whether the backend is enabled. Set to False to disable.
        """
        self.config = config or PostgresConfig()
        self.enabled = enabled and PSYCOPG2_AVAILABLE
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        
        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not available. Install with: pip install psycopg2-binary")
            self.enabled = False
            return
            
        if not self.enabled:
            logger.info("PostgreSQL backend disabled by configuration")
            return
            
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
        try:
            dsn_params = {
                "host": self.config.host,
                "port": self.config.port,
                "dbname": self.config.database,
                "user": self.config.user,
                "password": self.config.password,
                "connect_timeout": self.config.connect_timeout,
                "application_name": self.config.application_name,
                "sslmode": self.config.ssl_mode,
            }
            
            # Create threaded connection pool
            self._pool = pool.ThreadedConnectionPool(
                self.config.min_connections,
                self.config.max_connections,
                **dsn_params
            )
            
            # Test connection and set search path
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET search_path TO {self.config.schema}, public")
                    conn.commit()
            
            logger.info(
                f"PostgreSQL connected to {self.config.host}:{self.config.port}"
                f"/{self.config.database}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            self.enabled = False
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool with automatic return."""
        if not self.enabled or not self._pool:
            raise RuntimeError("PostgreSQL backend not available")
            
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            if conn:
                self._pool.putconn(conn)
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        
        Usage:
            with backend.transaction() as tx:
                tx.insert_agent(...)
                tx.insert_audit_event(...)
            # Automatically commits on success, rolls back on exception
        """
        with self._get_connection() as conn:
            try:
                yield TransactionContext(conn, self.config.schema)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    # =========================================================================
    # AGENT OPERATIONS
    # =========================================================================
    
    def insert_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        agent_type: str = "general",
        trust_level: float = 0.5,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
        region_id: Optional[str] = None,
        logical_domain: str = "default"
    ) -> Optional[str]:
        """
        Insert or update an agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            agent_type: Type of agent (e.g., 'general', 'autonomous', 'robot')
            trust_level: Trust level between 0 and 1
            status: Agent status ('active', 'suspended', 'terminated', 'quarantine')
            metadata: Additional metadata as JSON
            region_id: Region identifier
            logical_domain: Logical domain for multi-tenancy
            
        Returns:
            UUID of the inserted/updated agent, or None on failure
        """
        if not self.enabled:
            return None
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        INSERT INTO {self.config.schema}.agents 
                        (agent_id, name, agent_type, trust_level, status, metadata, region_id, logical_domain)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (agent_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            agent_type = EXCLUDED.agent_type,
                            trust_level = EXCLUDED.trust_level,
                            status = EXCLUDED.status,
                            metadata = EXCLUDED.metadata,
                            region_id = EXCLUDED.region_id,
                            logical_domain = EXCLUDED.logical_domain,
                            updated_at = NOW()
                        RETURNING id
                        """,
                        (
                            agent_id, name, agent_type, trust_level, status,
                            Json(metadata or {}), region_id, logical_domain
                        )
                    )
                    result = cur.fetchone()
                    conn.commit()
                    return str(result[0]) if result else None
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error inserting agent: {e}")
                    return None
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by agent_id."""
        if not self.enabled:
            return None
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"SELECT * FROM {self.config.schema}.agents WHERE agent_id = %s",
                    (agent_id,)
                )
                result = cur.fetchone()
                return dict(result) if result else None
    
    def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update agent status."""
        if not self.enabled:
            return False
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        UPDATE {self.config.schema}.agents 
                        SET status = %s, updated_at = NOW()
                        WHERE agent_id = %s
                        """,
                        (status, agent_id)
                    )
                    conn.commit()
                    return cur.rowcount > 0
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error updating agent status: {e}")
                    return False
    
    def list_agents(
        self,
        status: Optional[str] = None,
        region_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List agents with optional filtering."""
        if not self.enabled:
            return []
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = f"SELECT * FROM {self.config.schema}.agents WHERE 1=1"
                params: List[Any] = []
                
                if status:
                    query += " AND status = %s"
                    params.append(status)
                    
                if region_id:
                    query += " AND region_id = %s"
                    params.append(region_id)
                    
                query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    
    # =========================================================================
    # MODEL REGISTRY OPERATIONS
    # =========================================================================
    
    def register_model(
        self,
        model_name: str,
        version: str,
        artifact_path: Optional[str] = None,
        artifact_hash: Optional[str] = None,
        model_type: Optional[str] = None,
        framework: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> Optional[str]:
        """Register a new model version."""
        if not self.enabled:
            return None
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        INSERT INTO {self.config.schema}.model_versions
                        (model_name, version, artifact_path, artifact_hash, model_type,
                         framework, metrics, metadata, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            model_name, version, artifact_path, artifact_hash,
                            model_type, framework, Json(metrics or {}),
                            Json(metadata or {}), created_by
                        )
                    )
                    result = cur.fetchone()
                    conn.commit()
                    return str(result[0]) if result else None
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error registering model: {e}")
                    return None
    
    def get_model_version(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get a specific model version."""
        if not self.enabled:
            return None
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT * FROM {self.config.schema}.model_versions 
                    WHERE model_name = %s AND version = %s
                    """,
                    (model_name, version)
                )
                result = cur.fetchone()
                return dict(result) if result else None
    
    def promote_model(self, model_name: str, version: str, status: str = "production") -> bool:
        """Promote a model to a new status."""
        if not self.enabled:
            return False
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        UPDATE {self.config.schema}.model_versions
                        SET status = %s, promoted_at = NOW()
                        WHERE model_name = %s AND version = %s
                        """,
                        (status, model_name, version)
                    )
                    conn.commit()
                    return cur.rowcount > 0
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error promoting model: {e}")
                    return False
    
    def list_model_versions(
        self,
        model_name: str,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        if not self.enabled:
            return []
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = f"""
                    SELECT * FROM {self.config.schema}.model_versions 
                    WHERE model_name = %s
                """
                params: List[Any] = [model_name]
                
                if status:
                    query += " AND status = %s"
                    params.append(status)
                    
                query += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    
    # =========================================================================
    # POLICY OPERATIONS
    # =========================================================================
    
    def create_policy_version(
        self,
        policy_id: str,
        version: str,
        content: Dict[str, Any],
        policy_type: str = "governance",
        priority: int = 100,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a new policy version."""
        if not self.enabled:
            return None
            
        # Generate content hash
        content_str = json.dumps(content, sort_keys=True)
        version_hash = hashlib.sha256(content_str.encode()).hexdigest()[:64]
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        INSERT INTO {self.config.schema}.policy_versions
                        (policy_id, version, version_hash, content, policy_type, 
                         priority, created_by, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            policy_id, version, version_hash, Json(content),
                            policy_type, priority, created_by, Json(metadata or {})
                        )
                    )
                    result = cur.fetchone()
                    conn.commit()
                    return str(result[0]) if result else None
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error creating policy version: {e}")
                    return None
    
    def activate_policy(self, policy_id: str, version: str) -> bool:
        """Activate a policy version."""
        if not self.enabled:
            return False
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    # Deprecate current active version
                    cur.execute(
                        f"""
                        UPDATE {self.config.schema}.policy_versions
                        SET status = 'deprecated', deprecated_at = NOW()
                        WHERE policy_id = %s AND status = 'active'
                        """,
                        (policy_id,)
                    )
                    
                    # Activate new version
                    cur.execute(
                        f"""
                        UPDATE {self.config.schema}.policy_versions
                        SET status = 'active', activated_at = NOW()
                        WHERE policy_id = %s AND version = %s
                        """,
                        (policy_id, version)
                    )
                    conn.commit()
                    return cur.rowcount > 0
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error activating policy: {e}")
                    return False
    
    def get_active_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get the active version of a policy."""
        if not self.enabled:
            return None
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT * FROM {self.config.schema}.policy_versions
                    WHERE policy_id = %s AND status = 'active'
                    """,
                    (policy_id,)
                )
                result = cur.fetchone()
                return dict(result) if result else None
    
    # =========================================================================
    # AUDIT EVENT OPERATIONS
    # =========================================================================
    
    def insert_audit_event(
        self,
        agent_id: str,
        decision: str,
        action_type: Optional[str] = None,
        action_content: Optional[str] = None,
        risk_score: Optional[float] = None,
        latency_ms: Optional[float] = None,
        violations: Optional[List[Dict[str, Any]]] = None,
        policies_applied: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        region_id: Optional[str] = None,
        logical_domain: Optional[str] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Insert an audit event."""
        if not self.enabled:
            return False
            
        timestamp = timestamp or datetime.now(timezone.utc)
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        INSERT INTO {self.config.schema}.audit_events
                        (time, agent_id, action_type, action_content, decision,
                         risk_score, latency_ms, violations, policies_applied,
                         context, metadata, region_id, logical_domain, request_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            timestamp, agent_id, action_type, action_content, decision,
                            risk_score, latency_ms, Json(violations or []),
                            Json(policies_applied or []), Json(context or {}),
                            Json(metadata or {}), region_id, logical_domain, request_id
                        )
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error inserting audit event: {e}")
                    return False
    
    def batch_insert_audit_events(
        self,
        events: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Batch insert multiple audit events for high throughput.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Tuple of (success_count, failure_count)
        """
        if not self.enabled:
            return (0, len(events))
            
        success = 0
        failed = 0
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for event in events:
                    try:
                        timestamp = event.get('time', datetime.now(timezone.utc))
                        cur.execute(
                            f"""
                            INSERT INTO {self.config.schema}.audit_events
                            (time, agent_id, action_type, action_content, decision,
                             risk_score, latency_ms, violations, policies_applied,
                             context, metadata, region_id, logical_domain, request_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                timestamp,
                                event.get('agent_id'),
                                event.get('action_type'),
                                event.get('action_content'),
                                event.get('decision'),
                                event.get('risk_score'),
                                event.get('latency_ms'),
                                Json(event.get('violations', [])),
                                Json(event.get('policies_applied', [])),
                                Json(event.get('context', {})),
                                Json(event.get('metadata', {})),
                                event.get('region_id'),
                                event.get('logical_domain'),
                                event.get('request_id')
                            )
                        )
                        success += 1
                    except Exception as e:
                        logger.error(f"Error inserting audit event in batch: {e}")
                        failed += 1
                        
                conn.commit()
                
        return (success, failed)
    
    def query_audit_events(
        self,
        agent_id: Optional[str] = None,
        decision: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        region_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query audit events with filtering."""
        if not self.enabled:
            return []
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = f"SELECT * FROM {self.config.schema}.audit_events WHERE 1=1"
                params: List[Any] = []
                
                if agent_id:
                    query += " AND agent_id = %s"
                    params.append(agent_id)
                    
                if decision:
                    query += " AND decision = %s"
                    params.append(decision)
                    
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
    
    # =========================================================================
    # METRICS OPERATIONS
    # =========================================================================
    
    def insert_metric(
        self,
        agent_id: str,
        metric_name: str,
        metric_value: float,
        metric_type: str = "gauge",
        tags: Optional[Dict[str, Any]] = None,
        region_id: Optional[str] = None,
        logical_domain: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Insert a governance metric."""
        if not self.enabled:
            return False
            
        timestamp = timestamp or datetime.now(timezone.utc)
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        INSERT INTO {self.config.schema}.governance_metrics
                        (time, agent_id, metric_name, metric_value, metric_type,
                         tags, region_id, logical_domain)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (time, agent_id, metric_name) 
                        DO UPDATE SET metric_value = EXCLUDED.metric_value
                        """,
                        (
                            timestamp, agent_id, metric_name, metric_value,
                            metric_type, Json(tags or {}), region_id, logical_domain
                        )
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error inserting metric: {e}")
                    return False
    
    def aggregate_metrics(
        self,
        metric_name: str,
        aggregation: str = "avg",
        time_bucket: str = "1 hour",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        region_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate metrics over time buckets using TimescaleDB."""
        if not self.enabled:
            return []
            
        valid_aggs = ["avg", "sum", "min", "max", "count"]
        if aggregation not in valid_aggs:
            raise ValueError(f"Invalid aggregation: {aggregation}. Must be one of {valid_aggs}")
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Validated aggregation used in query (not user input)
                query = f"""
                    SELECT
                        time_bucket(%s, time) AS bucket,
                        {aggregation}(metric_value) AS value,
                        COUNT(*) AS count
                    FROM {self.config.schema}.governance_metrics
                    WHERE metric_name = %s
                """
                params: List[Any] = [time_bucket, metric_name]
                
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
    
    # =========================================================================
    # SECURITY EVENT OPERATIONS
    # =========================================================================
    
    def insert_security_event(
        self,
        event_type: str,
        severity: str,
        source_ip: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        description: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Insert a security event."""
        if not self.enabled:
            return False
            
        timestamp = timestamp or datetime.now(timezone.utc)
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        INSERT INTO {self.config.schema}.security_events
                        (time, event_type, severity, source_ip, agent_id,
                         user_id, description, details)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            timestamp, event_type, severity, source_ip,
                            agent_id, user_id, description, Json(details or {})
                        )
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error inserting security event: {e}")
                    return False
    
    # =========================================================================
    # API KEY OPERATIONS
    # =========================================================================
    
    def create_api_key(
        self,
        key_hash: str,
        key_prefix: str,
        name: Optional[str] = None,
        agent_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        rate_limit: int = 1000,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a new API key record."""
        if not self.enabled:
            return None
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        INSERT INTO {self.config.schema}.api_keys
                        (key_hash, key_prefix, name, agent_id, scopes,
                         rate_limit, expires_at, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            key_hash, key_prefix, name, agent_id,
                            Json(scopes or []), rate_limit, expires_at,
                            Json(metadata or {})
                        )
                    )
                    result = cur.fetchone()
                    conn.commit()
                    return str(result[0]) if result else None
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error creating API key: {e}")
                    return None
    
    def validate_api_key(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return its details if valid."""
        if not self.enabled:
            return None
            
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT * FROM {self.config.schema}.api_keys
                    WHERE key_hash = %s 
                      AND revoked_at IS NULL
                      AND (expires_at IS NULL OR expires_at > NOW())
                    """,
                    (key_hash,)
                )
                result = cur.fetchone()
                
                if result:
                    # Update last used timestamp
                    cur.execute(
                        f"""
                        UPDATE {self.config.schema}.api_keys
                        SET last_used_at = NOW()
                        WHERE key_hash = %s
                        """,
                        (key_hash,)
                    )
                    conn.commit()
                    
                return dict(result) if result else None
    
    def revoke_api_key(self, key_hash: str, reason: Optional[str] = None) -> bool:
        """Revoke an API key."""
        if not self.enabled:
            return False
            
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        UPDATE {self.config.schema}.api_keys
                        SET revoked_at = NOW(), revoked_reason = %s
                        WHERE key_hash = %s
                        """,
                        (reason, key_hash)
                    )
                    conn.commit()
                    return cur.rowcount > 0
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error revoking API key: {e}")
                    return False
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health and return status."""
        if not self.enabled:
            return {"status": "disabled", "available": False}
            
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    return {
                        "status": "healthy",
                        "available": True,
                        "version": version,
                        "host": self.config.host,
                        "database": self.config.database
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e)
            }
    
    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            try:
                self._pool.closeall()
                logger.info("PostgreSQL connection pool closed")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL pool: {e}")


class TransactionContext:
    """Context for database transactions with helper methods."""
    
    def __init__(self, conn, schema: str):
        self.conn = conn
        self.schema = schema
        self.cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    def execute(self, query: str, params: Optional[tuple] = None) -> None:
        """Execute a query within the transaction."""
        self.cursor.execute(query, params)
    
    def fetchone(self) -> Optional[Dict[str, Any]]:
        """Fetch one result."""
        result = self.cursor.fetchone()
        return dict(result) if result else None
    
    def fetchall(self) -> List[Dict[str, Any]]:
        """Fetch all results."""
        return [dict(row) for row in self.cursor.fetchall()]
    
    def insert_audit_event(self, **kwargs) -> bool:
        """Insert audit event within transaction."""
        timestamp = kwargs.get('time', datetime.now(timezone.utc))
        try:
            self.cursor.execute(
                f"""
                INSERT INTO {self.schema}.audit_events
                (time, agent_id, action_type, action_content, decision,
                 risk_score, latency_ms, violations, policies_applied,
                 context, metadata, region_id, logical_domain, request_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    timestamp,
                    kwargs.get('agent_id'),
                    kwargs.get('action_type'),
                    kwargs.get('action_content'),
                    kwargs.get('decision'),
                    kwargs.get('risk_score'),
                    kwargs.get('latency_ms'),
                    Json(kwargs.get('violations', [])),
                    Json(kwargs.get('policies_applied', [])),
                    Json(kwargs.get('context', {})),
                    Json(kwargs.get('metadata', {})),
                    kwargs.get('region_id'),
                    kwargs.get('logical_domain'),
                    kwargs.get('request_id')
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error inserting audit event in transaction: {e}")
            return False
    
    def insert_metric(self, **kwargs) -> bool:
        """Insert metric within transaction."""
        timestamp = kwargs.get('time', datetime.now(timezone.utc))
        try:
            self.cursor.execute(
                f"""
                INSERT INTO {self.schema}.governance_metrics
                (time, agent_id, metric_name, metric_value, metric_type,
                 tags, region_id, logical_domain)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (time, agent_id, metric_name) 
                DO UPDATE SET metric_value = EXCLUDED.metric_value
                """,
                (
                    timestamp,
                    kwargs.get('agent_id'),
                    kwargs.get('metric_name'),
                    kwargs.get('metric_value'),
                    kwargs.get('metric_type', 'gauge'),
                    Json(kwargs.get('tags', {})),
                    kwargs.get('region_id'),
                    kwargs.get('logical_domain')
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error inserting metric in transaction: {e}")
            return False
