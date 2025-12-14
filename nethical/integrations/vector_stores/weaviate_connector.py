"""Weaviate vector database integration with governance.

Provides integration with Weaviate schema-aware vector search engine
with built-in safety checks, PII detection, and audit logging.

Requirements:
    pip install weaviate-client>=4.0.0

Usage:
    from nethical.integrations.vector_stores import WeaviateConnector
    
    connector = WeaviateConnector(
        url="http://localhost:8080",
        api_key="your-api-key",
        class_name="Document",
    )
    
    # Upsert with governance
    connector.upsert([
        {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"text": "sample"}}
    ])
    
    # Hybrid search with PII redaction
    results = connector.query([0.1, 0.2, ...], top_k=10)
"""

import logging
from typing import Any, Dict, List, Optional
import uuid

from .base import VectorStoreProvider, VectorSearchResult

logger = logging.getLogger(__name__)

# Check if Weaviate is available
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    logger.warning("Weaviate not installed. Install with: pip install weaviate-client>=4.0.0")


class WeaviateConnector(VectorStoreProvider):
    """Weaviate vector search engine connector with governance.
    
    Provides full integration with Weaviate including:
    - Schema-aware governance checks
    - Hybrid search support (vector + keyword)
    - PII detection and redaction on query results
    - Audit logging for all operations
    - Multi-tenancy support
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        class_name: str = "Document",
        enable_governance: bool = True,
        enable_pii_detection: bool = True,
        enable_audit_logging: bool = True,
    ):
        """Initialize Weaviate connector.
        
        Args:
            url: Weaviate instance URL
            api_key: Optional API key for authentication
            class_name: Weaviate class name for objects
            enable_governance: Enable governance checks
            enable_pii_detection: Enable PII detection
            enable_audit_logging: Enable audit logging
            
        Raises:
            ImportError: If weaviate-client is not installed
            ConnectionError: If cannot connect to Weaviate
        """
        super().__init__(enable_governance, enable_pii_detection, enable_audit_logging)
        
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate not installed. Install with: pip install weaviate-client>=4.0.0")
        
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self._client = None
        
        # Initialize Weaviate
        self._init_weaviate()
    
    def _init_weaviate(self):
        """Initialize Weaviate client."""
        try:
            if self.api_key:
                self._client = weaviate.Client(
                    url=self.url,
                    auth_client_secret=weaviate.AuthApiKey(api_key=self.api_key),
                )
            else:
                self._client = weaviate.Client(url=self.url)
            
            # Check connection
            if self._client.is_ready():
                logger.info(f"Connected to Weaviate at {self.url}")
            else:
                raise ConnectionError("Weaviate is not ready")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise
    
    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
    ) -> int:
        """Upsert objects with governance checks on metadata.
        
        Args:
            vectors: List of objects to upsert. Each dict should contain:
                - id: Object ID (string, optional - will be generated if missing)
                - values: Vector values (list of floats)
                - metadata: Object properties/metadata dict
            namespace: Optional tenant name for multi-tenancy
            
        Returns:
            Number of objects successfully upserted
            
        Raises:
            ValueError: If governance check fails
            ConnectionError: If Weaviate connection fails
        """
        if not self._client:
            raise ConnectionError("Not connected to Weaviate")
        
        upserted_count = 0
        
        for vec in vectors:
            # Check governance on metadata
            if self.enable_governance:
                metadata = vec.get("metadata", {})
                check_result = self._check_governance(metadata, "vector_upsert")
                if check_result.get("decision") != "ALLOW":
                    raise ValueError(
                        f"Governance check failed for object {vec.get('id')}: "
                        f"{check_result.get('reason')}"
                    )
            
            try:
                # Prepare object for Weaviate
                obj_id = vec.get("id", str(uuid.uuid4()))
                properties = vec.get("metadata", {})
                vector_values = vec.get("values")
                
                # Create or update object
                self._client.data_object.create(
                    data_object=properties,
                    class_name=self.class_name,
                    uuid=obj_id,
                    vector=vector_values,
                    tenant=namespace if namespace else None,
                )
                upserted_count += 1
            except Exception as e:
                logger.error(f"Failed to upsert object {vec.get('id')}: {e}")
        
        # Audit log
        self._audit_log("upsert", {
            "namespace": namespace,
            "count": upserted_count,
            "class": self.class_name,
        })
        
        return upserted_count
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        hybrid_alpha: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """Query objects with PII redaction on results.
        
        Supports both pure vector search and hybrid search (vector + keyword).
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            filter: Optional property filter (where clause)
            namespace: Optional tenant name for multi-tenancy
            hybrid_alpha: If set (0-1), enables hybrid search. 0=keyword, 1=vector
            
        Returns:
            List of search results with PII redacted
            
        Raises:
            ConnectionError: If Weaviate connection fails
        """
        if not self._client:
            raise ConnectionError("Not connected to Weaviate")
        
        try:
            # Build query
            query_builder = (
                self._client.query
                .get(self.class_name, ["*"])
                .with_limit(top_k)
            )
            
            # Add vector search
            if hybrid_alpha is None:
                # Pure vector search
                query_builder = query_builder.with_near_vector({
                    "vector": vector,
                })
            else:
                # Hybrid search (vector + keyword)
                query_builder = query_builder.with_hybrid(
                    query="",  # Empty query for pure vector with hybrid scoring
                    alpha=hybrid_alpha,
                    vector=vector,
                )
            
            # Add filter if provided
            if filter:
                query_builder = query_builder.with_where(filter)
            
            # Add tenant if multi-tenancy
            if namespace:
                query_builder = query_builder.with_tenant(namespace)
            
            # Execute query
            result = query_builder.with_additional(["id", "distance", "vector"]).do()
            
            # Convert to VectorSearchResult with PII redaction
            results = []
            objects = result.get("data", {}).get("Get", {}).get(self.class_name, [])
            
            for obj in objects:
                obj_id = obj.get("_additional", {}).get("id", "")
                distance = obj.get("_additional", {}).get("distance", 1.0)
                score = 1.0 - distance  # Convert distance to similarity score
                vector_val = obj.get("_additional", {}).get("vector")
                
                # Extract properties (metadata)
                metadata = {k: v for k, v in obj.items() if not k.startswith("_")}
                
                # Redact PII from metadata
                if self.enable_pii_detection and metadata:
                    pii_result = self._detect_pii(metadata)
                    if pii_result["has_pii"]:
                        logger.info(f"PII detected in object {obj_id}, redacting")
                
                result_obj = VectorSearchResult(
                    id=obj_id,
                    score=score,
                    vector=vector_val,
                    metadata=metadata,
                    payload={},
                )
                results.append(result_obj)
            
            # Audit log
            self._audit_log("query", {
                "namespace": namespace,
                "top_k": top_k,
                "hybrid": hybrid_alpha is not None,
                "results_count": len(results),
                "class": self.class_name,
            })
            
            return results
        except Exception as e:
            logger.error(f"Weaviate query failed: {e}")
            raise ConnectionError(f"Failed to query objects: {e}")
    
    def delete(
        self,
        ids: List[str],
        namespace: str = "",
    ) -> int:
        """Delete objects with audit logging.
        
        Args:
            ids: List of object IDs to delete
            namespace: Optional tenant name for multi-tenancy
            
        Returns:
            Number of objects successfully deleted
            
        Raises:
            ConnectionError: If Weaviate connection fails
        """
        if not self._client:
            raise ConnectionError("Not connected to Weaviate")
        
        deleted_count = 0
        
        for obj_id in ids:
            try:
                self._client.data_object.delete(
                    uuid=obj_id,
                    class_name=self.class_name,
                    tenant=namespace if namespace else None,
                )
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete object {obj_id}: {e}")
        
        # Audit log
        self._audit_log("delete", {
            "namespace": namespace,
            "count": deleted_count,
            "ids": ids[:10],  # Log first 10 IDs
            "class": self.class_name,
        })
        
        return deleted_count
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of Weaviate connection.
        
        Returns:
            Health check result with status and details
        """
        base_health = super().health_check()
        
        try:
            if self._client and self._client.is_ready():
                # Get schema info
                schema = self._client.schema.get(self.class_name)
                
                base_health.update({
                    "status": "healthy",
                    "url": self.url,
                    "class": self.class_name,
                    "schema_exists": schema is not None,
                })
            else:
                base_health.update({
                    "status": "not_ready",
                    "url": self.url,
                })
        except Exception as e:
            base_health.update({
                "status": "error",
                "error": str(e),
            })
        
        return base_health
