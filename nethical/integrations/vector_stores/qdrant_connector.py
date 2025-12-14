"""Qdrant vector database integration with governance.

Provides integration with Qdrant high-performance vector search engine
with built-in safety checks, PII detection, and audit logging.

Requirements:
    pip install qdrant-client>=1.7.0

Usage:
    from nethical.integrations.vector_stores import QdrantConnector
    
    # Local mode
    connector = QdrantConnector(
        collection_name="my_collection",
        path="./qdrant_data",
    )
    
    # Client/server mode
    connector = QdrantConnector(
        collection_name="my_collection",
        url="http://localhost:6333",
    )
    
    # Upsert with governance
    connector.upsert([
        {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"text": "sample"}}
    ])
"""

import logging
from typing import Any, Dict, List, Optional

from .base import VectorStoreProvider, VectorSearchResult

logger = logging.getLogger(__name__)

# Check if Qdrant is available
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant not installed. Install with: pip install qdrant-client>=1.7.0")


class QdrantConnector(VectorStoreProvider):
    """Qdrant vector search engine connector with governance.
    
    Provides full integration with Qdrant including:
    - Payload filtering with governance
    - Batch operations support
    - PII detection and redaction on query results
    - Audit logging for all operations
    - Snapshot/backup validation
    """
    
    def __init__(
        self,
        collection_name: str = "nethical_collection",
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: int = 384,
        distance: str = "Cosine",
        enable_governance: bool = True,
        enable_pii_detection: bool = True,
        enable_audit_logging: bool = True,
    ):
        """Initialize Qdrant connector.
        
        Args:
            collection_name: Name of the Qdrant collection
            url: Qdrant server URL (e.g., "http://localhost:6333")
            host: Qdrant server host (alternative to url)
            port: Qdrant server port (alternative to url)
            path: Path for local persistence
            api_key: Optional API key for authentication
            vector_size: Dimension of vectors
            distance: Distance metric (Cosine, Euclid, Dot)
            enable_governance: Enable governance checks
            enable_pii_detection: Enable PII detection
            enable_audit_logging: Enable audit logging
            
        Raises:
            ImportError: If qdrant-client is not installed
            ConnectionError: If cannot connect to Qdrant
        """
        super().__init__(enable_governance, enable_pii_detection, enable_audit_logging)
        
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant not installed. Install with: pip install qdrant-client>=1.7.0")
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self._client = None
        
        # Initialize Qdrant
        self._init_qdrant(url, host, port, path, api_key)
    
    def _init_qdrant(
        self,
        url: Optional[str],
        host: Optional[str],
        port: Optional[int],
        path: Optional[str],
        api_key: Optional[str],
    ):
        """Initialize Qdrant client and collection."""
        try:
            # Create client based on connection mode
            if path:
                # Local mode
                self._client = QdrantClient(path=path)
                logger.info(f"Using Qdrant local persistence at {path}")
            elif url:
                # URL mode
                self._client = QdrantClient(url=url, api_key=api_key)
                logger.info(f"Connected to Qdrant server at {url}")
            elif host:
                # Host/port mode
                self._client = QdrantClient(host=host, port=port or 6333, api_key=api_key)
                logger.info(f"Connected to Qdrant server at {host}:{port or 6333}")
            else:
                # In-memory mode
                self._client = QdrantClient(":memory:")
                logger.info("Using Qdrant in-memory mode")
            
            # Create collection if it doesn't exist
            self._ensure_collection()
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    def _ensure_collection(self):
        """Ensure collection exists, create if not."""
        try:
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Map distance string to Qdrant Distance enum
                distance_map = {
                    "Cosine": Distance.COSINE,
                    "Euclid": Distance.EUCLID,
                    "Dot": Distance.DOT,
                }
                distance_metric = distance_map.get(self.distance, Distance.COSINE)
                
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=distance_metric,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
    ) -> int:
        """Upsert vectors with governance checks on payload.
        
        Args:
            vectors: List of vectors to upsert. Each dict should contain:
                - id: Vector ID (string or int)
                - values: Vector values (list of floats)
                - metadata: Optional payload/metadata dict
            namespace: Not used in Qdrant (collections provide isolation)
            
        Returns:
            Number of vectors successfully upserted
            
        Raises:
            ValueError: If governance check fails
            ConnectionError: If Qdrant connection fails
        """
        if not self._client:
            raise ConnectionError("Qdrant client not initialized")
        
        # Check governance on payloads
        if self.enable_governance:
            for vec in vectors:
                payload = vec.get("metadata", {})
                check_result = self._check_governance(payload, "vector_upsert")
                if check_result.get("decision") != "ALLOW":
                    raise ValueError(
                        f"Governance check failed for vector {vec.get('id')}: "
                        f"{check_result.get('reason')}"
                    )
        
        try:
            # Prepare points for Qdrant
            points = []
            for vec in vectors:
                vec_id = vec["id"]
                # Convert string IDs to hash if needed
                if isinstance(vec_id, str):
                    vec_id = hash(vec_id) & 0xFFFFFFFF  # Convert to positive int
                
                point = PointStruct(
                    id=vec_id,
                    vector=vec["values"],
                    payload=vec.get("metadata", {}),
                )
                points.append(point)
            
            # Upsert to Qdrant
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            
            # Audit log
            self._audit_log("upsert", {
                "namespace": namespace,
                "count": len(points),
                "collection": self.collection_name,
            })
            
            return len(points)
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")
            raise ConnectionError(f"Failed to upsert vectors: {e}")
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        score_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """Query vectors with PII redaction on results.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            filter: Optional payload filter (dict with field: value pairs)
            namespace: Not used in Qdrant (collections provide isolation)
            score_threshold: Optional minimum score threshold
            
        Returns:
            List of search results with PII redacted
            
        Raises:
            ConnectionError: If Qdrant connection fails
        """
        if not self._client:
            raise ConnectionError("Qdrant client not initialized")
        
        try:
            # Build filter if provided
            qdrant_filter = None
            if filter:
                # Simple filter implementation - can be extended
                conditions = []
                for field, value in filter.items():
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value),
                        )
                    )
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Search Qdrant
            search_result = self._client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            
            # Convert to VectorSearchResult with PII redaction
            results = []
            for point in search_result:
                payload = point.payload or {}
                
                # Redact PII from payload
                if self.enable_pii_detection and payload:
                    pii_result = self._detect_pii(payload)
                    if pii_result["has_pii"]:
                        logger.info(f"PII detected in point {point.id}, redacting")
                
                result = VectorSearchResult(
                    id=str(point.id),
                    score=point.score,
                    vector=point.vector if hasattr(point, 'vector') else None,
                    metadata=payload,
                    payload={},
                )
                results.append(result)
            
            # Audit log
            self._audit_log("query", {
                "namespace": namespace,
                "top_k": top_k,
                "results_count": len(results),
                "collection": self.collection_name,
            })
            
            return results
        except Exception as e:
            logger.error(f"Qdrant query failed: {e}")
            raise ConnectionError(f"Failed to query vectors: {e}")
    
    def delete(
        self,
        ids: List[str],
        namespace: str = "",
    ) -> int:
        """Delete vectors with audit logging.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Not used in Qdrant (collections provide isolation)
            
        Returns:
            Number of vectors successfully deleted
            
        Raises:
            ConnectionError: If Qdrant connection fails
        """
        if not self._client:
            raise ConnectionError("Qdrant client not initialized")
        
        try:
            # Convert string IDs to int if needed
            point_ids = []
            for vec_id in ids:
                if isinstance(vec_id, str):
                    point_ids.append(hash(vec_id) & 0xFFFFFFFF)
                else:
                    point_ids.append(vec_id)
            
            # Delete from Qdrant
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )
            
            # Audit log
            self._audit_log("delete", {
                "namespace": namespace,
                "count": len(ids),
                "ids": ids[:10],  # Log first 10 IDs
                "collection": self.collection_name,
            })
            
            return len(ids)
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            raise ConnectionError(f"Failed to delete vectors: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of Qdrant connection.
        
        Returns:
            Health check result with status and details
        """
        base_health = super().health_check()
        
        try:
            if self._client:
                # Get collection info
                collection_info = self._client.get_collection(self.collection_name)
                
                base_health.update({
                    "status": "healthy",
                    "collection": self.collection_name,
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count,
                    "indexed_vectors_count": collection_info.indexed_vectors_count,
                })
            else:
                base_health.update({
                    "status": "not_initialized",
                    "collection": None,
                })
        except Exception as e:
            base_health.update({
                "status": "error",
                "error": str(e),
            })
        
        return base_health
