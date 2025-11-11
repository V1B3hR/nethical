"""
Elasticsearch integration for audit log search and analytics.

This module provides Elasticsearch integration for full-text search,
advanced filtering, and analytics on audit logs and governance events.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

try:
    from elasticsearch import Elasticsearch

    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    Elasticsearch = None

logger = logging.getLogger(__name__)


class ElasticsearchAuditStore:
    """
    Elasticsearch store for audit logs and governance events.

    Features:
    - Full-text search across audit logs
    - Advanced filtering and aggregations
    - Real-time indexing
    - Automatic index lifecycle management
    - Multi-field queries and faceted search
    """

    def __init__(
        self,
        hosts: List[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_prefix: str = "nethical",
        enabled: bool = True,
    ):
        """
        Initialize Elasticsearch audit store.

        Args:
            hosts: List of Elasticsearch hosts (default: ["localhost:9200"])
            username: Elasticsearch username
            password: Elasticsearch password
            index_prefix: Prefix for index names
            enabled: Whether Elasticsearch is enabled
        """
        self.enabled = enabled and ELASTICSEARCH_AVAILABLE
        self.index_prefix = index_prefix

        if not ELASTICSEARCH_AVAILABLE:
            logger.warning("Elasticsearch not available. Install with: pip install elasticsearch")
            self.enabled = False
            return

        if not self.enabled:
            logger.info("Elasticsearch disabled by configuration")
            return

        hosts = hosts or ["localhost:9200"]

        try:
            # Create Elasticsearch client
            self.client = Elasticsearch(
                hosts=hosts,
                basic_auth=(username, password) if username and password else None,
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
            )

            # Test connection
            if not self.client.ping():
                raise ConnectionError("Cannot connect to Elasticsearch")

            # Initialize indices
            self._initialize_indices()
            logger.info(f"Elasticsearch connected to {hosts}")

        except Exception as e:
            logger.warning(f"Failed to connect to Elasticsearch: {e}")
            self.enabled = False

    def _initialize_indices(self):
        """Initialize Elasticsearch indices with mappings."""
        # Audit logs index
        audit_index = f"{self.index_prefix}-audit-logs"
        if not self.client.indices.exists(index=audit_index):
            self.client.indices.create(
                index=audit_index,
                body={
                    "settings": {
                        "number_of_shards": 3,
                        "number_of_replicas": 1,
                        "index.lifecycle.name": "nethical-lifecycle",
                    },
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "action_id": {"type": "keyword"},
                            "agent_id": {"type": "keyword"},
                            "decision": {"type": "keyword"},
                            "action_type": {"type": "keyword"},
                            "stated_intent": {"type": "text"},
                            "actual_action": {"type": "text"},
                            "context": {"type": "text"},
                            "violations": {
                                "type": "nested",
                                "properties": {
                                    "type": {"type": "keyword"},
                                    "severity": {"type": "integer"},
                                    "confidence": {"type": "float"},
                                    "message": {"type": "text"},
                                },
                            },
                            "region_id": {"type": "keyword"},
                            "logical_domain": {"type": "keyword"},
                            "risk_score": {"type": "float"},
                            "metadata": {"type": "object", "enabled": False},
                        }
                    },
                },
            )
            logger.info(f"Created Elasticsearch index: {audit_index}")

        # Events index
        events_index = f"{self.index_prefix}-events"
        if not self.client.indices.exists(index=events_index):
            self.client.indices.create(
                index=events_index,
                body={
                    "settings": {"number_of_shards": 2, "number_of_replicas": 1},
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "event_type": {"type": "keyword"},
                            "agent_id": {"type": "keyword"},
                            "severity": {"type": "keyword"},
                            "message": {"type": "text"},
                            "region_id": {"type": "keyword"},
                            "logical_domain": {"type": "keyword"},
                            "data": {"type": "object", "enabled": False},
                        }
                    },
                },
            )
            logger.info(f"Created Elasticsearch index: {events_index}")

    def index_audit_log(
        self,
        action_id: str,
        agent_id: str,
        decision: str,
        action_data: Dict[str, Any],
        violations: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[datetime] = None,
        region_id: Optional[str] = None,
        logical_domain: Optional[str] = None,
        risk_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Index an audit log entry.

        Args:
            action_id: Action identifier
            agent_id: Agent identifier
            decision: Decision made
            action_data: Action data (stated_intent, actual_action, etc.)
            violations: List of violations
            timestamp: Timestamp (uses current time if None)
            region_id: Region identifier
            logical_domain: Logical domain
            risk_score: Risk score
            metadata: Additional metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        timestamp = timestamp or datetime.now(timezone.utc)

        try:
            doc = {
                "timestamp": timestamp.isoformat(),
                "action_id": action_id,
                "agent_id": agent_id,
                "decision": decision,
                "action_type": action_data.get("action_type"),
                "stated_intent": action_data.get("stated_intent"),
                "actual_action": action_data.get("actual_action"),
                "context": action_data.get("context"),
                "violations": violations or [],
                "region_id": region_id,
                "logical_domain": logical_domain,
                "risk_score": risk_score,
                "metadata": metadata or {},
            }

            index = f"{self.index_prefix}-audit-logs"
            self.client.index(index=index, document=doc, id=action_id)
            return True

        except Exception as e:
            logger.error(f"Error indexing audit log: {e}")
            return False

    def index_event(
        self,
        event_type: str,
        message: str,
        agent_id: Optional[str] = None,
        severity: str = "INFO",
        timestamp: Optional[datetime] = None,
        region_id: Optional[str] = None,
        logical_domain: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Index an event.

        Args:
            event_type: Type of event
            message: Event message
            agent_id: Agent identifier
            severity: Event severity
            timestamp: Timestamp (uses current time if None)
            region_id: Region identifier
            logical_domain: Logical domain
            data: Event data

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        timestamp = timestamp or datetime.now(timezone.utc)

        try:
            doc = {
                "timestamp": timestamp.isoformat(),
                "event_type": event_type,
                "agent_id": agent_id,
                "severity": severity,
                "message": message,
                "region_id": region_id,
                "logical_domain": logical_domain,
                "data": data or {},
            }

            index = f"{self.index_prefix}-events"
            self.client.index(index=index, document=doc)
            return True

        except Exception as e:
            logger.error(f"Error indexing event: {e}")
            return False

    def search_audit_logs(
        self,
        query: Optional[str] = None,
        agent_id: Optional[str] = None,
        decision: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        region_id: Optional[str] = None,
        logical_domain: Optional[str] = None,
        violation_types: Optional[List[str]] = None,
        min_risk_score: Optional[float] = None,
        size: int = 100,
        from_: int = 0,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        Search audit logs with advanced filtering.

        Args:
            query: Full-text search query
            agent_id: Filter by agent ID
            decision: Filter by decision
            start_time: Start of time range
            end_time: End of time range
            region_id: Filter by region
            logical_domain: Filter by logical domain
            violation_types: Filter by violation types
            min_risk_score: Minimum risk score
            size: Number of results to return
            from_: Starting offset
            sort_by: Field to sort by
            sort_order: Sort order (asc or desc)

        Returns:
            Search results with hits and aggregations
        """
        if not self.enabled:
            return {"hits": {"total": {"value": 0}, "hits": []}, "aggregations": {}}

        try:
            # Build query
            must_clauses = []

            if query:
                must_clauses.append(
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["stated_intent", "actual_action", "context"],
                        }
                    }
                )

            if agent_id:
                must_clauses.append({"term": {"agent_id": agent_id}})

            if decision:
                must_clauses.append({"term": {"decision": decision}})

            if region_id:
                must_clauses.append({"term": {"region_id": region_id}})

            if logical_domain:
                must_clauses.append({"term": {"logical_domain": logical_domain}})

            if start_time or end_time:
                range_clause = {"range": {"timestamp": {}}}
                if start_time:
                    range_clause["range"]["timestamp"]["gte"] = start_time.isoformat()
                if end_time:
                    range_clause["range"]["timestamp"]["lte"] = end_time.isoformat()
                must_clauses.append(range_clause)

            if violation_types:
                must_clauses.append(
                    {
                        "nested": {
                            "path": "violations",
                            "query": {"terms": {"violations.type": violation_types}},
                        }
                    }
                )

            if min_risk_score is not None:
                must_clauses.append({"range": {"risk_score": {"gte": min_risk_score}}})

            # Build search body
            body = {
                "query": {"bool": {"must": must_clauses if must_clauses else [{"match_all": {}}]}},
                "size": size,
                "from": from_,
                "sort": [{sort_by: {"order": sort_order}}],
                "aggs": {
                    "decisions": {"terms": {"field": "decision"}},
                    "agents": {"terms": {"field": "agent_id", "size": 10}},
                    "regions": {"terms": {"field": "region_id"}},
                    "violations": {
                        "nested": {"path": "violations"},
                        "aggs": {"types": {"terms": {"field": "violations.type"}}},
                    },
                },
            }

            index = f"{self.index_prefix}-audit-logs"
            result = self.client.search(index=index, body=body)
            return result

        except Exception as e:
            logger.error(f"Error searching audit logs: {e}")
            return {"hits": {"total": {"value": 0}, "hits": []}, "aggregations": {}}

    def get_audit_log_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        region_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get audit log statistics and aggregations.

        Args:
            start_time: Start of time range
            end_time: End of time range
            region_id: Filter by region

        Returns:
            Statistics dictionary
        """
        if not self.enabled:
            return {}

        try:
            must_clauses = []

            if start_time or end_time:
                range_clause = {"range": {"timestamp": {}}}
                if start_time:
                    range_clause["range"]["timestamp"]["gte"] = start_time.isoformat()
                if end_time:
                    range_clause["range"]["timestamp"]["lte"] = end_time.isoformat()
                must_clauses.append(range_clause)

            if region_id:
                must_clauses.append({"term": {"region_id": region_id}})

            body = {
                "size": 0,
                "query": {"bool": {"must": must_clauses if must_clauses else [{"match_all": {}}]}},
                "aggs": {
                    "total_actions": {"value_count": {"field": "action_id"}},
                    "decisions": {"terms": {"field": "decision"}},
                    "agents": {"cardinality": {"field": "agent_id"}},
                    "avg_risk_score": {"avg": {"field": "risk_score"}},
                    "max_risk_score": {"max": {"field": "risk_score"}},
                    "timeline": {
                        "date_histogram": {"field": "timestamp", "calendar_interval": "1h"}
                    },
                },
            }

            index = f"{self.index_prefix}-audit-logs"
            result = self.client.search(index=index, body=body)

            return {
                "total_actions": result["hits"]["total"]["value"],
                "decisions": result["aggregations"]["decisions"]["buckets"],
                "unique_agents": result["aggregations"]["agents"]["value"],
                "avg_risk_score": result["aggregations"]["avg_risk_score"]["value"],
                "max_risk_score": result["aggregations"]["max_risk_score"]["value"],
                "timeline": result["aggregations"]["timeline"]["buckets"],
            }

        except Exception as e:
            logger.error(f"Error getting audit log statistics: {e}")
            return {}

    def close(self):
        """Close Elasticsearch connection."""
        if self.enabled and hasattr(self, "client"):
            try:
                self.client.close()
                logger.info("Elasticsearch connection closed")
            except Exception as e:
                logger.error(f"Error closing Elasticsearch connection: {e}")
